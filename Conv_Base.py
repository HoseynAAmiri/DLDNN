import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from DLD_env import DLD_env, Pillar
from DLD_Utils import DLD_Utils as utl
from keras.models import load_model
from tensorflow.keras.models import Model
from tensorflow.keras import layers
import tensorflow.keras.backend as K
from keras.utils.vis_utils import plot_model
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Lambda, Add



test_frac = 0.2
summary = True
epoch = 200
batch_size = 2

# loading data 
dataset = utl.load_data('tiny_dataset')


# Spiliting data to train test sections
train_ix = np.random.choice(len(dataset[0]), size=int(
    (1-test_frac)*len(dataset[0])), replace=False)
test_ix = np.setdiff1d(np.arange(len(dataset[0])), train_ix)

psi_train, label_train = np.nan_to_num(dataset[0][train_ix]), np.nan_to_num(dataset[3][train_ix])

psi_test, label_test = np.nan_to_num(dataset[0][test_ix]), np.nan_to_num(dataset[3][test_ix])

# Normilizing data and saving the Normilized value

Max_Train = []
Max_Test = []

Max_Train.append(np.max(psi_train, axis=(1, 2), keepdims=True))
Max_Train.append(np.amax(label_train, axis=1))

Max_Test.append(np.max(psi_test, axis=(1, 2), keepdims=True))
Max_Test.append(np.amax(label_test, axis=1))

psi_train = psi_train/Max_Train[0]
label_train = label_train/Max_Train[1][:, None]

psi_test = psi_test/Max_Test[0]
label_test = label_test/Max_Test[1][:, None]




encoder_input_psi = layers.Input(
        shape=(psi_train[0].shape[0], psi_train[0].shape[1], 1), name="original_img_psi")
X = layers.Conv2D(16, (3, 3), activation="relu",
                          padding="same")(encoder_input_psi)
X = layers.MaxPooling2D((2, 2), padding="same")(X)

X = layers.Conv2D(16, (3, 3), activation="relu",
                          padding="same")(X)
X = layers.MaxPooling2D((2, 2), padding="same")(X)

X = layers.Conv2D(16, (3, 3), activation="relu", padding="same")(X)
encoder_output_psi = layers.MaxPooling2D((2, 2), padding="same")(X)
encoder_psi = Model(
    encoder_input_psi, encoder_output_psi, name="encoder_psi")

if summary:
    encoder_psi.summary()
# Decoder
decoder_input_psi = layers.Input(
    shape=(16, 16, 16), name="encoded_img_psi")
X = layers.Conv2DTranspose(
    16, (3, 3), strides=2, activation="relu",
    padding="same")(decoder_input_psi)

X = layers.Conv2DTranspose(
    16, (3, 3), strides=2, activation="relu",
    padding="same")(X)

X = layers.Conv2DTranspose(
    16, (3, 3), strides=2, activation="relu",
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
x = layers.Input(shape=psi_train[0].shape,  name="x")
y = layers.Input(shape=psi_train[0].shape,  name="y")

# psi branch
X = layers.Dense(64, activation="relu")(FCNN_input)
X = layers.Dropout(0.2)(X)
X = layers.Dense(512, activation="relu")(X)
X = layers.Dropout(0.2)(X)
X = layers.Dense(2048, activation="relu")(X)
X = layers.Dropout(0.2)(X)
X = layers.Dense(4096, activation="linear")(X)
X = layers.Dropout(0.2)(X)
FCNN_output_psi = layers.Reshape((16, 16, 16))(X)

FCNN = Model(inputs=FCNN_input,
                    outputs=FCNN_output_psi,
                    name="FCNN")

if summary:
    FCNN.summary()

encoded_img_psi = FCNN(FCNN_input, x, y)
decoded_img_psi = decoder_psi(encoded_img_psi)

### Physics informed
# Define gradient function
def gradient(y, x, name, order=1):

    g = Lambda(lambda z: tf.gradients(
        z[0], z[1], unconnected_gradients='zero')[0], name=name)
    for _ in range(order):
        y = g([y, x])

    return y

psi = decoded_img_psi

x_grid_size = 128
y_grid_size = 128

xx = np.linspace(0, 1, x_grid_size)
yy = np.linspace(0, 1, y_grid_size)
x_grid, y_grid = np.meshgrid(xx, yy)

x = tf.Variable(x_grid, dtype=tf.float32)
y = tf.Variable(y_grid, dtype=tf.float32)

u = gradient(psi, y, name='u')
v = gradient(-psi, x, name='v')

u_x = gradient(u, x, name='u_x')
v_y = gradient(v, y, name='v_y')

con = Add()([u_x, v_y])

DLDNN = Model(inputs=FCNN_input, outputs=[decoded_img_psi, con], name="DLDNN")

opt = keras.optimizers.Adam()

autoencoder_psi.compile(optimizer=opt, loss= 'mse')
DLDNN.compile(optimizer=opt, loss= 'mse')

# history = autoencoder_psi.fit(
#     x=psi_train,
#     y=psi_train,
#     epochs=epoch,
#     batch_size=batch_size,
#     shuffle=True,
#     validation_data=(psi_test, psi_test)
# )

# plt.figure()
# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.title('model loss')
# plt.ylabel('loss')
# plt.yscale('log')
# plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='upper left')
# plt.show()
# autoencoder_psi.save('model_autoencoder_psi.h5')

autoencoder_psi.load_weights('model_autoencoder_psi.h5')
decoder_psi.trainable = False
autoencoder_psi.compile(optimizer=opt, loss= 'mse')
DLDNN.compile(optimizer=opt, loss= 'mse')

history = DLDNN.fit(
    x=label_train,
    y=[psi_train, np.zeros_like(psi_train)],
    epochs=epoch,
    batch_size=batch_size,
    shuffle=True,
    validation_data=(label_test, psi_test))

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

DLDNN.load_weights('model_DLDNN.h5')




