import os
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

def display( original_img, decoded_img, DLD_img, num_data=3, streamline=True):
    """
    Displays ten random images from each one of the supplied arrays.
    """
    indices = np.random.randint(len(original_img), size=num_data)
    images1 = original_img[indices, :]
    images2 = decoded_img[indices, :]
    images3 = DLD_img[indices, :]

    grid_size_oi = original_img[0].shape
    grid_size_di = decoded_img[0].shape
    grid_size_dld = DLD_img[0].shape

    plt.figure(figsize=(10, 6))
    for i, (image1, image2, image3) in enumerate(zip(images1, images2, images3)):

        ax = plt.subplot(3, num_data, i + 1)
        plt.imshow(image1.reshape(grid_size_oi))
        plt.jet()
        if i == 0:
            plt.title("Original field")
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        ax = plt.subplot(3, num_data, i + 1 + num_data)
        plt.imshow(image2.reshape(grid_size_di))
        plt.jet()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        if i == 0:
            plt.title("Decoded field")

        ax = plt.subplot(3, num_data, i + 1 + 2*num_data)
        plt.imshow(image3.reshape(grid_size_dld))
        plt.jet()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        if i == 0:
            plt.title("DLDNN outputs")

    plt.show()
    return indices

test_frac = 0.2
summary = False
epoch = 100
EPOCH = 10
batch_size = 32
T1 = False
freeze = True
T2 = True

x_grid_size = 128
y_grid_size = 128

xx = np.linspace(0, 1, x_grid_size)
yy = np.linspace(0, 1, y_grid_size)
x_grid, y_grid = np.meshgrid(xx, yy)

dx = xx[1] - xx[0]
dy = yy[1] - yy[0]
# loading data
dataset = utl.load_data('dataset')


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

psi_train = psi_train/Max_Train[0]
label_train = label_train/Max_Train[1][:, None]

psi_test = psi_test/Max_Test[0]
label_test = label_test/Max_Test[1][:, None]


encoder_input_psi = layers.Input(
    shape=(psi_train[0].shape[0], psi_train[0].shape[1], 1), name="original_img_psi")
X = layers.Conv2D(64, (3, 3), activation="relu",
                  padding="same")(encoder_input_psi)

X = layers.MaxPooling2D((2, 2), padding="same")(X)


# X = layers.Conv2D(64, (3, 3), activation="relu",
#                   padding="same")(X)
# X = layers.MaxPooling2D((2, 2), padding="same")(X)

X = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(X)
encoder_output_psi = layers.MaxPooling2D((2, 2), padding="same")(X)
encoder_psi = Model(
    encoder_input_psi, encoder_output_psi, name="encoder_psi")

if summary:
    encoder_psi.summary()
# Decoder
decoder_input_psi = layers.Input(
    shape=(32, 32, 32), name="encoded_img_psi")
X = layers.Conv2DTranspose(
    32, (3, 3), strides=2, activation="relu",
    padding="same")(decoder_input_psi)

# X = layers.Conv2DTranspose(
#     64, (3, 3), strides=2, activation="relu",
#     padding="same")(X)

X = layers.Conv2DTranspose(
    64, (3, 3), strides=2, activation="relu",
    padding="same")(X)
decoder_output_psi = layers.Conv2D(
    1, (3, 3), activation="linear", padding="same")(X)
decoder_psi = Model(
    decoder_input_psi, decoder_output_psi, name="decoder_psi")
if summary:
    decoder_psi.summary()

# Autoencoder psi
autoencoder_input_psi = layers.Input(
    shape=(psi_train[0].shape[0], psi_train[0].shape[1], 1), name="img_psi")
encoded_img_psi = encoder_psi(autoencoder_input_psi)
decoded_img_psi = decoder_psi(encoded_img_psi)
autoencoder_psi = Model(
    autoencoder_input_psi, decoded_img_psi, name="autoencoder_psi")

if summary:
    autoencoder_psi.summary()

FCNN_input = layers.Input(shape=label_train[0].shape,  name="labels")
# x = layers.Input(shape=x_train[0].shape, name="x")
# y = layers.Input(shape=y_train[0].shape, name="y")

# xy = Concatenate()([x, y])
# xy = Flatten()(xy)
# xy_dense = layers.Dense(1, activation="tanh")(xy)
# FCNN_input = Concatenate()([xy_dense, FCNN_input])

# psi branch
X = layers.Dense(512, activation="tanh")(FCNN_input)
X = layers.Dropout(0.2)(X)
X = layers.Dense(1024, activation="tanh")(X)
X = layers.Dropout(0.2)(X)
X = layers.Dense(4096, activation="tanh")(X)
X = layers.Dropout(0.2)(X)
X = layers.Dense(2*16384, activation="relu")(X)
X = layers.Dropout(0.2)(X)
FCNN_output_psi = layers.Reshape((32, 32, 32))(X)

FCNN = Model(inputs=FCNN_input,
             outputs=FCNN_output_psi,
             name="FCNN")

if summary:
    FCNN.summary()

encoded_img_psi = FCNN(FCNN_input)
decoded_img_psi = decoder_psi(encoded_img_psi)

# Physics informed
# Define gradient function



psi = decoded_img_psi

u, v = tf.image.image_gradients(psi)
u = u / dy
v = -v / dx

_, u_x = tf.image.image_gradients(u)
v_y, _ = tf.image.image_gradients(v)

u_x = u_x / dx
v_y = v_y / dy
con = tf.abs(tf.reduce_sum(u_x + v_y))

DLDNN = Model(inputs=FCNN_input, outputs=decoded_img_psi, name="DLDNN")

plot_model(DLDNN, to_file='Conv_Base_plot.png', show_shapes=True, show_layer_names=True)
plot_model(FCNN, to_file='FCNN_plot.png', show_shapes=True, show_layer_names=True)

opt = keras.optimizers.Adam(0.01)
DLDNN.add_loss(con)
DLDNN.add_metric(con, name='continuity')

autoencoder_psi.compile(optimizer=opt, loss='mse')
DLDNN.compile(optimizer=opt, loss='mse')


if T1:
    class myCallback(keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            if (epoch+1) % EPOCH == 0:
                psi_AutE = autoencoder_psi.predict(dataset[0])[:, :, :, 0]
                psi_DLD = DLDNN.predict(dataset[3])
                psi_DLD = psi_DLD[:, :, :, 0]
                display(dataset[0], psi_AutE, psi_DLD)
                
    callback = myCallback()  
         
    history = autoencoder_psi.fit(
        x=psi_train,
        y=psi_train,
        epochs=epoch,
        batch_size=batch_size,
        shuffle=False,
        validation_data=(psi_test, psi_test)
    )

    plt.figure()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.yscale('log')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    autoencoder_psi.save('model_autoencoder_psi.h5')
else:
    autoencoder_psi.load_weights('model_autoencoder_psi.h5')

if freeze:
    decoder_psi.trainable = False
    autoencoder_psi.compile(optimizer=opt, loss='mse')
    DLDNN.compile(optimizer=opt, loss='mse')


if T2:
    class myCallback(keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            if (epoch+1) % EPOCH == 0:
                psi_AutE = autoencoder_psi.predict(dataset[0])[:, :, :, 0]
                psi_DLD = DLDNN.predict(dataset[3])
                psi_DLD = psi_DLD[:, :, :, 0]
                display(dataset[0], psi_AutE, psi_DLD)
                
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


psi_AutE = autoencoder_psi.predict(dataset[0])[:, :, :, 0]
psi_DLD = DLDNN.predict(dataset[3])
psi_DLD = psi_DLD[:, :, :, 0]
display(dataset[0], psi_AutE, psi_DLD)