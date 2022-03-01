import os
from telnetlib import X3PAD
from tkinter.tix import X_REGION
from pyrsistent import freeze
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Lambda, Add, Concatenate, Flatten
from tensorflow.keras.models import Model
import numpy as np
import matplotlib.pyplot as plt
from keras.utils.vis_utils import plot_model
from keras.models import load_model
from DLD_Utils import DLD_Utils as utl

np.random.seed(1234)
tf.random.set_seed(1234)

def display( original_img,DLD_img, num_data=5, streamline=True):
    """
    Displays ten random images from each one of the supplied arrays.
    """
    indices = np.random.randint(len(original_img), size=num_data)
    images1 = original_img[indices, :]
    images2 = DLD_img[indices, :]

    grid_size_oi = original_img[0].shape
    grid_size_dld = DLD_img[0].shape

    plt.figure(figsize=(10, 4))
    for i, (image1, image2) in enumerate(zip(images1, images2)):

        ax = plt.subplot(3, num_data, i + 1)
        plt.imshow(image1.reshape(grid_size_oi))
        plt.jet()
        if i == 0:
            plt.title("GT")
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        ax = plt.subplot(3, num_data, i + 1 + 1*num_data)
        plt.imshow(image2.reshape(grid_size_dld))
        plt.jet()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        if i == 0:
            plt.title("Output")

    plt.show()
    return indices

test_frac = 0.2
summary = True
epoch = 500
EPOCH = 25
batch_size = 2
lr = 0.0005

T2 = True

x_grid_size = 128
y_grid_size = 128

xx = np.linspace(0, 1, x_grid_size)
yy = np.linspace(0, 1, y_grid_size)
x_grid, y_grid = np.meshgrid(xx, yy)

dx = xx[1] - xx[0]
dy = yy[1] - yy[0]
# loading data
dataset = utl.load_data('tiny_dataset')


# Spiliting data to train test sections
train_ix = np.random.choice(len(dataset[0]), size=int(
    (1-test_frac)*len(dataset[0])), replace=False)
test_ix = np.setdiff1d(np.arange(len(dataset[0])), train_ix)

psi_train, label_train = np.nan_to_num(
    dataset[0][train_ix]), np.nan_to_num(dataset[3][train_ix])

psi_test, label_test = np.nan_to_num(
    dataset[0][test_ix]), np.nan_to_num(dataset[3][test_ix])

# Normilizing data and saving the Normilized value

x_grid_size = 128
y_grid_size = 128

xx = np.linspace(0, 1, x_grid_size)
yy = np.linspace(0, 1, y_grid_size)
x_grid, y_grid = np.meshgrid(xx, yy)

x_train = np.tile(x_grid, (psi_train.shape[0], 1, 1))
y_train = np.tile(y_grid, (psi_train.shape[0], 1, 1))

x_test = np.tile(x_grid, (psi_test.shape[0], 1, 1))
y_test = np.tile(y_grid, (psi_test.shape[0], 1, 1))

Max_Train = []
Max_Test = []

Max_Train.append(np.max(np.abs(psi_train), axis=(1, 2), keepdims=True))
Max_Train.append(np.amax(label_train, axis=1))

Max_Test.append(np.max(np.abs(psi_test), axis=(1, 2), keepdims=True))
Max_Test.append(np.amax(label_test, axis=1))

psi_train = (psi_train/Max_Train[0])
label_train = label_train/Max_Train[1][:, None]

psi_test = (psi_test/Max_Test[0])
label_test = label_test/Max_Test[1][:, None]

# Neural Network 

input = layers.Input(shape=label_train[0].shape,  name="labels")
X1 = layers.Dense(8, activation="relu")(input[:, 0:1])
X2 = layers.Dense(8, activation="relu")(input[:, 1:2])
X3 = layers.Dense(8, activation="relu")(input[:, 2:3])
X4 = layers.Dense(8, activation="relu")(input[:, 3:4])
X = layers.Concatenate(axis=1)([X1, X2, X3, X4])

X = layers.Dense(128, activation="relu")(X)
X = layers.Dense(128, activation="relu")(X)
X = layers.Dense(128, activation="relu")(X)
X = layers.Dense(128, activation="relu")(X)
X = layers.Dense(128, activation="relu")(X)
X = layers.Dense(16*16*16)(X)

X = layers.ReLU()(X)

X = layers.Reshape((16, 16, 16))(X)

X = layers.Conv2D(16, (3, 3),
     padding="same")(X)
X = layers.ReLU()(X)

X = layers.UpSampling2D((2, 2))(X)
X = layers.Conv2D(128, (3, 3),
     padding="same")(X)

X = layers.ReLU()(X)

X = layers.UpSampling2D((2, 2))(X)
X = layers.Conv2D(128, (3, 3),
    padding="same")(X)
X = layers.ReLU()(X)
X = layers.Conv2D(128, (3, 3),
    padding="same")(X)
X = layers.ReLU()(X)
X = layers.Conv2D(128, (3, 3),
    padding="same")(X)
X = layers.ReLU()(X)


X = layers.UpSampling2D((2, 2))(X)
X = layers.Conv2D(64, (3, 3),
    padding="same")(X)
X = layers.ReLU()(X)
X = layers.Conv2D(32, (3, 3),
    padding="same")(X)
X = layers.ReLU()(X)

X = layers.Conv2D(1, (3, 3), activation="linear",
    padding="same")(X)

output = X

# Physics informed
# Define gradient function
psi = X

u, v = tf.image.image_gradients(psi)
u = u / dy
v = -v / dx

_, u_x = tf.image.image_gradients(u)
v_y, _ = tf.image.image_gradients(v)

u_x = u_x / dx
v_y = v_y / dy
con = tf.abs(tf.reduce_sum(u_x + v_y))

DLDNN = Model(inputs=input, outputs=output, name="DLDNN")

# DLDNN.add_loss(con)
# DLDNN.add_metric(con, name='continuity')
opt = keras.optimizers.Adam(lr)
DLDNN.compile(optimizer=opt, loss='mse')

if summary:
    DLDNN.summary() 

opt = keras.optimizers.Adam(learning_rate=lr, beta_1=0.5)
DLDNN.compile(optimizer=opt, loss='mse')


if T2:
    class myCallback(keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            if (epoch+1) % EPOCH == 0:
                psi_DLD = DLDNN.predict(dataset[3])
                psi_DLD = psi_DLD[:, :, :, 0]
                display(dataset[0], psi_DLD)
                
    callback = myCallback()  
    history = DLDNN.fit(
        x=label_train,
        y=psi_train,
        epochs=epoch,
        batch_size=batch_size,
        shuffle=False,
        validation_data=(label_test, psi_test),
        callbacks=[callback])

    plt.figure()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.yscale('log')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    DLDNN.save('model_DLDNN.h5')
else:
    DLDNN.load_weights('model_DLDNN.h5')


psi_DLD = DLDNN.predict(dataset[3])
psi_DLD = psi_DLD[:, :, :, 0]
display(dataset[0], psi_DLD)