
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
from sklearn.model_selection import train_test_split
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Initialize the utility class
utl = utl()


class ConvNet():
    def __init__(self):
        pass

    def preprocess(self, input_array, test_frac=0.2):

        # Spiliting data to train test sections
        train_ix = np.random.choice(len(input_array[0]), size=int(
            (1-test_frac)*len(input_array[0])), replace=False)
        test_ix = np.setdiff1d(np.arange(len(input_array[0])), train_ix)

        psi_train, px_train, py_train, label_train = np.nan_to_num(input_array[0][train_ix]), np.nan_to_num(
            input_array[1][train_ix]), np.nan_to_num(input_array[2][train_ix]), np.nan_to_num(input_array[3][train_ix])
        
        psi_test, px_test, py_test, label_test = np.nan_to_num(input_array[0][test_ix]), np.nan_to_num(
            input_array[1][test_ix]), np.nan_to_num(input_array[2][test_ix]), np.nan_to_num(input_array[3][test_ix])

        # Normilizing data and saving the Normilized value

        Max_Train = []
        Max_Test = []

        Max_Train.append(np.max(psi_train, axis=(1, 2), keepdims=True))
        Max_Train.append(np.max(px_train, axis=(1, 2), keepdims=True))
        Max_Train.append(np.max(py_train, axis=(1, 2), keepdims=True))
        Max_Train.append(np.amax(label_train, axis=1))

        Max_Test.append(np.max(psi_test, axis=(1, 2), keepdims=True))
        Max_Test.append(np.max(px_test, axis=(1, 2), keepdims=True))
        Max_Test.append(np.max(py_test, axis=(1, 2), keepdims=True))
        Max_Test.append(np.amax(label_test, axis=1))

        output_psi_train = psi_train/Max_Train[0]
        output_px_train = px_train/Max_Train[1]
        output_py_train = py_train/Max_Train[2]
        output_label_train = label_train/Max_Train[3][:, None]
        output_train = (output_psi_train, output_px_train, output_py_train, output_label_train)

        output_psi_test = psi_test/Max_Test[0]
        output_px_test = px_test/Max_Test[1]
        output_py_test = py_test/Max_Test[2]
        output_label_test = label_test/Max_Test[3][:, None]
        output_test = (output_psi_test, output_px_test, output_py_test, output_label_test)

        return output_train, output_test

    def display(self, original_img, decoded_img, DLD_img, num_data=5, streamline=True):
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

        plt.figure(figsize=(20, 6))
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
                plt.title("Decoded Filed")

            ax = plt.subplot(3, num_data, i + 1 + 2*num_data)
            plt.imshow(image3.reshape(grid_size_dld))
            plt.jet()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            if i == 0:
                plt.title("DLDNN outputs")

        plt.show()
        return indices

    def create_model(self, input_shape_field, input_shape_label,
                     auteloss="mse", dldnnloss="mse", summary=False, PINN=True):

        self.auteloss = auteloss
        self.dldnnloss = dldnnloss
        ##########################################################
        #                  psi autoencoder                       #
        ##########################################################

        encoder_input_psi = layers.Input(
            shape=(input_shape_field[0], input_shape_field[1], 1), name="original_img_psi")
        X = layers.Conv2D(16, (3, 3), activation="relu",
                          padding="same")(encoder_input_psi)
        X = layers.MaxPooling2D((2, 2), padding="same")(X)

        X = layers.Conv2D(16, (3, 3), activation="relu",
                          padding="same")(X)
        X = layers.MaxPooling2D((2, 2), padding="same")(X)

        X = layers.Conv2D(16, (3, 3), activation="relu", padding="same")(X)
        encoder_output_psi = layers.MaxPooling2D((2, 2), padding="same")(X)
        self.encoder_psi = Model(
            encoder_input_psi, encoder_output_psi, name="encoder_psi")

        if summary:
            self.encoder_psi.summary()
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

        self.decoder_psi = Model(
            decoder_input_psi, decoder_output_psi, name="decoder_psi")
        if summary:
            self.decoder_psi.summary()

        # Autoencoder psi
        autoencoder_input_psi = layers.Input(
            shape=(input_shape_field[0], input_shape_field[1], 1), name="img_psi")
        encoded_img_psi = self.encoder_psi(autoencoder_input_psi)
        decoded_img_psi = self.decoder_psi(encoded_img_psi)
        self.autoencoder_psi = Model(
            autoencoder_input_psi, decoded_img_psi, name="autoencoder_psi")

        if summary:
            self.autoencoder_psi.summary()

        ##########################################################
        #                   p_x autoencoder                      #
        ##########################################################

        encoder_input_px = layers.Input(
            shape=(input_shape_field[0], input_shape_field[1], 1), name="original_img_px")
        X = layers.Conv2D(16, (3, 3), activation="relu",
                          padding="same")(encoder_input_px)
        X = layers.MaxPooling2D((2, 2), padding="same")(X)

        X = layers.Conv2D(16, (3, 3), activation="relu",
                          padding="same")(X)
        X = layers.MaxPooling2D((2, 2), padding="same")(X)

        X = layers.Conv2D(16, (3, 3), activation="relu", padding="same")(X)
        encoder_output_px = layers.MaxPooling2D((2, 2), padding="same")(X)
        self.encoder_px = Model(
            encoder_input_px, encoder_output_px, name="encoder_px")

        if summary:
            self.encoder_px.summary()
        # Decoder
        decoder_input_px = layers.Input(
            shape=(16, 16, 16), name="encoded_img_px")
        X = layers.Conv2DTranspose(
            16, (3, 3), strides=2, activation="relu",
            padding="same")(decoder_input_px)

        X = layers.Conv2DTranspose(
            16, (3, 3), strides=2, activation="relu",
            padding="same")(X)

        X = layers.Conv2DTranspose(
            16, (3, 3), strides=2, activation="relu",
            padding="same")(X)
        decoder_output_px = layers.Conv2D(
            1, (3, 3), activation="linear", padding="same")(X)

        self.decoder_px = Model(
            decoder_input_px, decoder_output_px, name="decoder_Px")
        if summary:
            self.decoder_px.summary()

        # Autoencoder px
        autoencoder_input_px = layers.Input(
            shape=(input_shape_field[0], input_shape_field[1], 1), name="img_px")
        encoded_img_px = self.encoder_px(autoencoder_input_px)
        decoded_img_px = self.decoder_px(encoded_img_px)
        self.autoencoder_px = Model(
            autoencoder_input_px, decoded_img_px, name="autoencoder_px")

        if summary:
            self.autoencoder_px.summary()

        ##########################################################
        #                    py autoencoder                       #
        ##########################################################

        encoder_input_py = layers.Input(
            shape=(input_shape_field[0], input_shape_field[1], 1), name="original_img_py")
        X = layers.Conv2D(16, (3, 3), activation="relu",
                          padding="same")(encoder_input_py)
        X = layers.MaxPooling2D((2, 2), padding="same")(X)

        X = layers.Conv2D(16, (3, 3), activation="relu",
                          padding="same")(X)
        X = layers.MaxPooling2D((2, 2), padding="same")(X)

        X = layers.Conv2D(16, (3, 3), activation="relu", padding="same")(X)
        encoder_output_py = layers.MaxPooling2D((2, 2), padding="same")(X)
        self.encoder_py = Model(
            encoder_input_py, encoder_output_py, name="encoder_py")

        if summary:
            self.encoder_py.summary()
        # Decoder py
        decoder_input_py = layers.Input(
            shape=(16, 16, 16), name="encoded_img_py")
        X = layers.Conv2DTranspose(
            16, (3, 3), strides=2, activation="relu",
            padding="same")(decoder_input_py)

        X = layers.Conv2DTranspose(
            16, (3, 3), strides=2, activation="relu",
            padding="same")(X)

        X = layers.Conv2DTranspose(
            16, (3, 3), strides=2, activation="relu",
            padding="same")(X)
        decoder_output_py = layers.Conv2D(
            1, (3, 3), activation="linear", padding="same")(X)

        self.decoder_py = Model(
            decoder_input_py, decoder_output_py, name="decoder_Py")
        if summary:
            self.decoder_py.summary()

        # Autoencoder py
        autoencoder_input_py = layers.Input(
            shape=(input_shape_field[0], input_shape_field[1], 1), name="img_py")
        encoded_img_py = self.encoder_py(autoencoder_input_py)
        decoded_img_py = self.decoder_py(encoded_img_py)
        self.autoencoder_py = Model(
            autoencoder_input_py, decoded_img_py, name="autoencoder_py")

        if summary:
            self.autoencoder_py.summary()

        ##########################################################
        #        Main-fully connected neural network             #
        ##########################################################

        FCNN_input = layers.Input(shape=input_shape_label,  name="labels")

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

        # px branch
        X = layers.Dense(64, activation="relu")(FCNN_input)
        X = layers.Dropout(0.2)(X)
        X = layers.Dense(512, activation="relu")(X)
        X = layers.Dropout(0.2)(X)
        X = layers.Dense(2048, activation="relu")(X)
        X = layers.Dropout(0.2)(X)
        X = layers.Dense(4096, activation="linear")(X)
        X = layers.Dropout(0.2)(X)
        FCNN_output_px = layers.Reshape((16, 16, 16))(X)

        # py branch
        X = layers.Dense(64, activation="relu")(FCNN_input)
        X = layers.Dropout(0.2)(X)
        X = layers.Dense(512, activation="relu")(X)
        X = layers.Dropout(0.2)(X)
        X = layers.Dense(2048, activation="relu")(X)
        X = layers.Dropout(0.2)(X)
        X = layers.Dense(4096, activation="linear")(X)
        X = layers.Dropout(0.2)(X)
        FCNN_output_py = layers.Reshape((16, 16, 16))(X)

        self.FCNN = Model(inputs=FCNN_input,
                          outputs=[FCNN_output_psi,
                                   FCNN_output_px, FCNN_output_py],
                          name="FCNN")

        if summary:
            self.FCNN.summary()

        [encoded_img_psi, encoded_img_px,
            encoded_img_py] = self.FCNN(FCNN_input)
        decoded_img_psi = self.decoder_psi(encoded_img_psi)
        decoded_img_px = self.decoder_px(encoded_img_px)
        decoded_img_py = self.decoder_py(encoded_img_py)

        self.DLDNN = Model(inputs=FCNN_input,
                           outputs=[decoded_img_psi, decoded_img_px, decoded_img_py], name="DLDNN")

        # Apply physics informed Loss to the DLDNN model
        if PINN:
            dy, dx = 1/(np.array(decoded_img_psi[0, :, :, 0].shape))
            
            u, v= tf.image.image_gradients(decoded_img_psi)
            u = u
            v = -v

            u_y, u_x = tf.image.image_gradients(u)
            v_y, v_x = tf.image.image_gradients(v)
    
            _, u_xx = tf.image.image_gradients(u_x)
            u_yy, _ = tf.image.image_gradients(u_y)
    
            _, v_xx  = tf.image.image_gradients(v_x)
            v_yy, _ = tf.image.image_gradients(v_y)
            
            Re = FCNN_input[:, 3]

            f_u = Re * (u * u_x + v * u_y) + Re * decoded_img_px - (u_xx + u_yy) 
            f_v = Re * (u * v_x + v * v_y) + Re * decoded_img_py - (v_xx + v_yy) 
    
            PINN_loss = tf.math.reduce_sum(tf.math.abs(f_u)) + tf.math.reduce_sum(
                tf.math.abs(f_v)) + tf.math.reduce_sum(tf.math.abs(u_x + v_y))
            self.DLDNN.add_loss(PINN_loss)
        
        if summary:
            plot_model(self.DLDNN, to_file='DLDNN_PINN_plot.png',
                       show_shapes=True, show_layer_names=True)
            self.DLDNN.summary()
        # set optimizer
        self.opt = keras.optimizers.Adam()
        # compile
        self.compile_models()

    def compile_models(self):
        self.autoencoder_psi.compile(optimizer=self.opt, loss=self.auteloss)
        self.autoencoder_px.compile(optimizer=self.opt, loss=self.auteloss)
        self.autoencoder_py.compile(optimizer=self.opt, loss=self.auteloss)
        self.DLDNN.compile(optimizer=self.opt, loss=[
                           self.dldnnloss, self.dldnnloss, self.dldnnloss])

    def train_AutoE_psi(self, train_data, test_data, epoch, batch_size=128):

        history = self.autoencoder_psi.fit(
            x=train_data,
            y=train_data,
            epochs=epoch,
            batch_size=batch_size,
            shuffle=True,
            validation_data=(test_data, test_data)
        )

        plt.figure()
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')

        plt.show()

        return history.history

    def train_AutoE_px(self, train_data, test_data, epoch, batch_size):

        history = self.autoencoder_px.fit(
            x=train_data,
            y=train_data,
            epochs=epoch,
            batch_size=batch_size,
            shuffle=True,
            validation_data=(test_data, test_data)
        )

        plt.figure()
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()

        return history.history

    def train_AutoE_py(self, train_data, test_data, epoch, batch_size):

        history = self.autoencoder_py.fit(
            x=train_data,
            y=train_data,
            epochs=epoch,
            batch_size=batch_size,
            shuffle=True,
            validation_data=(test_data, test_data)
        )

        plt.figure()
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()

        return history.history

    def train_DLDNN(self, x_train, y_train, x_test, y_test, epoch, batch_size):

        history = self.DLDNN.fit(
            x=x_train,
            y=y_train,
            epochs=epoch,
            batch_size=batch_size,
            shuffle=True,
            validation_data=(x_test, y_test)
        )

        plt.figure()
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')

        plt.show()

        return history.history

    def prediction(self, model, test_data, num_data=5):
        predictions = model.predict(test_data)[:, :, :, 0]
        self.display(test_data, predictions, num_data=num_data)

######################################################################
######################################################################
##                Training Neurals Networks                         ##
######################################################################
######################################################################


def network_train(epoch_AutoE=10, batch_size_AutoE=32, epoch_DLDNN=10,
                  batch_size_DLDNN=32, AutoE_psi_train=False, AutoE_px_train=False,
                  AutoE_py_train=False, DLDNN_train=False
                  ):

    # loading dataset from pickle file
    dataset = utl.load_data('dataset')

    # Initializing our Neural Network class
    NN = ConvNet()

    # spiliting and Normilizing data
    Data_train, Data_test = NN.preprocess(dataset)

    # Determinig the grid size and label size from the data shape
    grid_size = Data_train[0][0].shape
    label_size = Data_train[3][0].shape

    # Create the Neural networks
    NN.create_model(grid_size, label_size, summary=False, auteloss="mse",
                    dldnnloss="mse")

    # Train the Autoencoders
    if AutoE_psi_train:
        history = NN.train_AutoE_psi(
            Data_train[0], Data_test[0], epoch_AutoE, batch_size_AutoE)
        NN.autoencoder_psi.save('model_autoencoder_psi.h5')
        np.save('AutoE_psi_history.npy', history)

    if AutoE_px_train:
        history = NN.train_AutoE_px(
            Data_train[1], Data_test[1], epoch_AutoE, batch_size_AutoE)
        NN.autoencoder_px.save('model_autoencoder_px.h5')
        np.save('AutoE_px_history.npy', history)

    if AutoE_py_train:
        history = NN.train_AutoE_py(
            Data_train[2], Data_test[2], epoch_AutoE, batch_size_AutoE)
        NN.autoencoder_py.save('model_autoencoder_py.h5')
        np.save('AutoE_py_history.npy', history)

    # load the autoencoder weight for transfer learning
    NN.autoencoder_psi.load_weights('model_autoencoder_psi.h5')
    NN.autoencoder_px.load_weights('model_autoencoder_px.h5')
    NN.autoencoder_py.load_weights('model_autoencoder_py.h5')
    # freeze the decoder's weights
    #NN.decoder_psi.trainable = False
    #NN.decoder_p.trainable = False
    # NN.compile_models()

    # Training the DLDNN network
    if DLDNN_train:
        history = NN.train_DLDNN(Data_train[3], [Data_train[0], Data_train[1], Data_train[2]],
                                 Data_test[3], [Data_test[0], Data_test[1], Data_test[2]], epoch_DLDNN, batch_size_DLDNN)
        NN.DLDNN.save('model_DLDNN1.h5')
        np.save('DLDNN_history.npy', history)
    # load the DLDNN model
    NN.DLDNN.load_weights('model_DLDNN1.h5')
    # Make predictions by Autoencoder and DLDNN
    psi_AutE = NN.autoencoder_psi.predict(Data_test[0])[:, :, :, 0]
    px_AutE = NN.autoencoder_px.predict(Data_test[1])[:, :, :, 0]
    py_AutE = NN.autoencoder_py.predict(Data_test[2])[:, :, :, 0]
    [psi_DLD, px_DLD, py_DLD] = NN.DLDNN.predict(Data_test[3])

    psi_DLD = psi_DLD[:, :, :, 0]
    px_DLD = px_DLD[:, :, :, 0]
    py_DLD = py_DLD[:, :, :, 0]

    # display original fields and predicted autoencoder and DLDNN result
    NN.display(Data_test[0], psi_AutE, psi_DLD)
    NN.display(Data_test[1], px_AutE, px_DLD)
    NN.display(Data_test[1], py_AutE, py_DLD)
    

def Network_evaluation(D, N, G_X, G_R, Re, grid_size, dp, start_point):
    NN = ConvNet()
    label_size = 4
    NN.create_model(grid_size, label_size, auteloss="mse",
                    dldnnloss="mse", summary=False)
    NN.DLDNN.load_weights('model_DLDNN.h5')

    net_input = [D, N, G_X, Re]
    net_input_norm = net_input/np.max(net_input)

    pillar = Pillar(D, N, G_X, G_R)
    dld = DLD_env(pillar, Re, resolution=grid_size)

    psi = NN.DLDNN.predict(net_input_norm[None, :])[0, :, :, 0]
    print(psi.shape)
    v, u = utl.gradient(psi, -dld.dx, dld.dy)
    plt.figure()
    plt.imshow(np.flip(psi, axis=0))
    plt.show()
    uv = (u, v)
    #dld.simulate_particle(dp/(D+G_X), uv, pillar.pillars, start_point, periods=6, plot=True)

    


epoch_AutoE = 30
batch_size_AutoE = 32
epoch_DLDNN = 20
batch_size_DLDNN = 32
a, b = network_train(epoch_AutoE, batch_size_AutoE, epoch_DLDNN, batch_size_DLDNN,
              AutoE_psi_train=False, AutoE_px_train=False, AutoE_py_train=False, DLDNN_train=True)

D = 20
N = 5
G_X = 40
G_R = 1
Re = 1
grid_size = (128, 128)
start_point = (0, 0.4)
dp = 10
#Network_evaluation(D, N, G_X, G_R, Re, grid_size, dp, start_point)
