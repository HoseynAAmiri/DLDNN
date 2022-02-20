
from email.utils import decode_params
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
import scipy.io

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

        u_train, v_train, p_train, t_train = np.nan_to_num(input_array[0][train_ix]), np.nan_to_num(
            input_array[1][train_ix]), np.nan_to_num(input_array[2][train_ix]), np.nan_to_num(input_array[3][train_ix])
        
        u_test, v_test, p_test, t_test = np.nan_to_num(input_array[0][test_ix]), np.nan_to_num(
            input_array[1][test_ix]), np.nan_to_num(input_array[2][test_ix]), np.nan_to_num(input_array[3][test_ix])

        # Normilizing data and saving the Normilized value

        Max_Train = []
        Max_Test = []

        Max_Train.append(np.max(u_train, axis=(1, 2), keepdims=True))
        Max_Train.append(np.max(v_train, axis=(1, 2), keepdims=True))
        Max_Train.append(np.max(p_train, axis=(1, 2), keepdims=True))
        Max_Train.append(np.amax(t_train))

        Max_Test.append(np.max(u_test, axis=(1, 2), keepdims=True))
        Max_Test.append(np.max(v_test, axis=(1, 2), keepdims=True))
        Max_Test.append(np.max(p_test, axis=(1, 2), keepdims=True))
        Max_Test.append(np.amax(t_test))

        output_v_train = u_train/Max_Train[0]
        output_u_train = v_train/Max_Train[1]
        output_p_train = p_train/Max_Train[2]
        output_t_train = t_train/Max_Train[3]
        output_train = (output_u_train, output_v_train, output_p_train, output_t_train)

        output_u_test = u_test/Max_Test[0]
        output_v_test = v_test/Max_Test[1]
        output_p_test = p_test/Max_Test[2]
        output_t_test = t_test/Max_Test[3]
        output_test = (output_u_test, output_v_test, output_p_test, output_t_test)

        return output_train, output_test

    def display(self, original, decoded, VSH, num_data=5, streamline=True):
        """
        Displays ten random images from each one of the supplied arrays.
        """
        indices = np.random.randint(len(original), size=num_data)
        images1 = original[indices, :]
        images2 = decoded[indices, :]
        images3 = VSH[indices, :]

        grid_size_oi = original[0].shape
        grid_size_di = decoded[0].shape
        grid_size_dld = VSH[0].shape

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
                plt.title("VSH outputs")

        plt.show()
        return indices

    def create_model(self, input_shape_field, input_shape_t,
                     auteloss="mse", VSHloss="mse", summary=True, PINN=True):

        self.auteloss = auteloss
        self.VSHloss = VSHloss
        encoded_shape = (7, 13, 16)
        ##########################################################
        #                  u autoencoder                       #
        ##########################################################

        encoder_input_u = layers.Input(
            shape=(input_shape_field[0], input_shape_field[1], 1), name="original_u")
        X = layers.Conv2D(16, (3, 3), activation="relu",
                          padding="same")(encoder_input_u)
        X = layers.MaxPooling2D((2, 2), padding="same")(X)

        X = layers.Conv2D(16, (3, 3), activation="relu",
                          padding="same")(X)
        X = layers.MaxPooling2D((2, 2), padding="same")(X)

        X = layers.Conv2D(16, (3, 3), activation="relu", padding="same")(X)
        encoder_output_u = layers.MaxPooling2D((2, 2), padding="same")(X)
        self.encoder_u = Model(
            encoder_input_u, encoder_output_u, name="encoder_u")

        if summary:
            self.encoder_u.summary()
        # Decoder
        decoder_input_u = layers.Input(
            shape=encoded_shape, name="encoded_u")
        X = layers.Conv2DTranspose(
            16, (3, 3), strides=2, activation="relu",
            padding="same")(decoder_input_u)

        X = layers.Conv2DTranspose(
            16, (3, 3), strides=2, activation="relu",
            padding="same")(X)

        X = layers.Conv2DTranspose(
            16, (3, 3), strides=2, activation="relu",
            padding="same")(X)
        decoder_output_u = layers.Conv2D(
            1, (3, 3), activation="linear", padding="same")(X)

        self.decoder_u = Model(
            decoder_input_u, decoder_output_u, name="decoder_u")
        if summary:
            self.decoder_u.summary()



        ##########################################################
        #                   v autoencoder                      #
        ##########################################################

        encoder_input_v = layers.Input(
            shape=(input_shape_field[0], input_shape_field[1], 1), name="original_v")
        X = layers.Conv2D(16, (3, 3), activation="relu",
                          padding="same")(encoder_input_v)
        X = layers.MaxPooling2D((2, 2), padding="same")(X)

        X = layers.Conv2D(16, (3, 3), activation="relu",
                          padding="same")(X)
        X = layers.MaxPooling2D((2, 2), padding="same")(X)

        X = layers.Conv2D(16, (3, 3), activation="relu", padding="same")(X)
        encoder_output_v = layers.MaxPooling2D((2, 2), padding="same")(X)
        self.encoder_v = Model(
            encoder_input_v, encoder_output_v, name="encoder_v")

        if summary:
            self.encoder_v.summary()
        # Decoder
        decoder_input_v = layers.Input(
            shape=encoded_shape, name="encoded_img_px")
        X = layers.Conv2DTranspose(
            16, (3, 3), strides=2, activation="relu",
            padding="same")(decoder_input_v)

        X = layers.Conv2DTranspose(
            16, (3, 3), strides=2, activation="relu",
            padding="same")(X)

        X = layers.Conv2DTranspose(
            16, (3, 3), strides=2, activation="relu",
            padding="same")(X)
        decoder_output_v = layers.Conv2D(
            1, (3, 3), activation="linear", padding="same")(X)

        self.decoder_v = Model(
            decoder_input_v, decoder_output_v, name="decoder_v")
        if summary:
            self.decoder_v.summary()

        ##########################################################
        #                    p autoencoder                      #
        ##########################################################

        encoder_input_p = layers.Input(
            shape=(input_shape_field[0], input_shape_field[1], 1), name="original_p")
        X = layers.Conv2D(16, (3, 3), activation="relu",
                          padding="same")(encoder_input_p)
        X = layers.MaxPooling2D((2, 2), padding="same")(X)

        X = layers.Conv2D(16, (3, 3), activation="relu",
                          padding="same")(X)
        X = layers.MaxPooling2D((2, 2), padding="same")(X)

        X = layers.Conv2D(16, (3, 3), activation="relu", padding="same")(X)
        encoder_output_p = layers.MaxPooling2D((2, 2), padding="same")(X)
        self.encoder_p = Model(
            encoder_input_p, encoder_output_p, name="encoder_p")

        if summary:
            self.encoder_p.summary()
        # Decoder py
        decoder_input_p = layers.Input(
            shape=encoded_shape, name="encoded_p")
        X = layers.Conv2DTranspose(
            16, (3, 3), strides=2, activation="relu",
            padding="same")(decoder_input_p)

        X = layers.Conv2DTranspose(
            16, (3, 3), strides=2, activation="relu",
            padding="same")(X)

        X = layers.Conv2DTranspose(
            16, (3, 3), strides=2, activation="relu",
            padding="same")(X)
        decoder_output_p = layers.Conv2D(
            1, (3, 3), activation="linear", padding="same")(X)

        self.decoder_p = Model(
            decoder_input_p, decoder_output_p, name="decoder_P")
        
        if summary:
            self.decoder_p.summary()
       
        # Autoencoder psi
        input_u = layers.Input(
            shape=(input_shape_field[0], input_shape_field[1], 1), name="original_u")
        encoded_u = self.encoder_u(input_u)
        decoded_u = self.decoder_v(encoded_u)

        input_v = layers.Input(
            shape=(input_shape_field[0], input_shape_field[1], 1), name="original_v")
        encoded_v = self.encoder_v(input_v)
        decoded_v = self.decoder_v(encoded_v)

        input_p = layers.Input(
            shape=(input_shape_field[0], input_shape_field[1], 1), name="original_p")
        encoded_p = self.encoder_p(input_p)
        decoded_p = self.decoder_p(encoded_p)
        
        self.autoencoder = Model(
            [input_u, input_v, input_p], [decoded_u, decoded_v, decoded_p], name="autoencoder")
        if summary:
            self.autoencoder.summary()

        ##########################################################
        #        Main-fully connected neural network             #
        ##########################################################

        FCNN_input = layers.Input(shape=input_shape_t,  name="t")

        # u branch
        X = layers.Dense(64, activation="relu")(FCNN_input)
        X = layers.Dropout(0.2)(X)
        X = layers.Dense(512, activation="relu")(X)
        X = layers.Dropout(0.2)(X)
        X = layers.Dense(2048, activation="relu")(X)
        X = layers.Dropout(0.2)(X)
        X = layers.Dense(4096, activation="linear")(X)
        X = layers.Dropout(0.2)(X)
        FCNN_output_u = layers.Reshape(encoded_shape)(X)

        # v branch
        X = layers.Dense(64, activation="relu")(FCNN_input)
        X = layers.Dropout(0.2)(X)
        X = layers.Dense(512, activation="relu")(X)
        X = layers.Dropout(0.2)(X)
        X = layers.Dense(2048, activation="relu")(X)
        X = layers.Dropout(0.2)(X)
        X = layers.Dense(4096, activation="linear")(X)
        X = layers.Dropout(0.2)(X)
        FCNN_output_v = layers.Reshape(encoded_shape)(X)

        # p branch
        X = layers.Dense(64, activation="relu")(FCNN_input)
        X = layers.Dropout(0.2)(X)
        X = layers.Dense(512, activation="relu")(X)
        X = layers.Dropout(0.2)(X)
        X = layers.Dense(2048, activation="relu")(X)
        X = layers.Dropout(0.2)(X)
        X = layers.Dense(4096, activation="linear")(X)
        X = layers.Dropout(0.2)(X)
        FCNN_output_p = layers.Reshape(encoded_shape)(X)

        self.FCNN = Model(inputs=FCNN_input,
                          outputs=[FCNN_output_u,
                                   FCNN_output_v, FCNN_output_p],
                          name="FCNN")

        if summary:
            self.FCNN.summary()

        [encoded_u, encoded_v,
            encoded_p] = self.FCNN(FCNN_input)
        decoded_u = self.decoder_u(encoded_u)
        decoded_v = self.decoder_v(encoded_v)
        decoded_p = self.decoder_p(encoded_p)

        self.VSH = Model(inputs=FCNN_input,
                           outputs=[decoded_u, decoded_v, decoded_p], name="VSH")

        # Apply physics informed Loss to the DLDNN model
        if PINN:
            #dy, dx = 1/(np.array(decoded_img_psi[0, :, :, 0].shape))
            
            u = decoded_u
            v = decoded_v
            p = decoded_p
            Re = 100

            u_y, u_x = tf.image.image_gradients(u)
            v_y, v_x = tf.image.image_gradients(v)
    
            _, u_xx = tf.image.image_gradients(u_x)
            u_yy, _ = tf.image.image_gradients(u_y)
    
            _, v_xx  = tf.image.image_gradients(v_x)
            v_yy, _ = tf.image.image_gradients(v_y)
            
            p_y, p_x = tf.image.image_gradients(p)

            f_u = Re * (u * u_x + v * u_y) + Re * p_x - (u_xx + u_yy) 
            f_v = Re * (u * v_x + v * v_y) + Re * P_y - (v_xx + v_yy) 
    
            PINN_loss = tf.math.reduce_sum(tf.math.abs(f_u)) + tf.math.reduce_sum(
                tf.math.abs(f_v)) + tf.math.reduce_sum(tf.math.abs(u_x + v_y))
            self.VSH.add_loss(PINN_loss)
        
        if summary:
            plot_model(self.VSH, to_file='VSH_PINN_plot.png',
                       show_shapes=True, show_layer_names=True)
            self.VSH.summary()
        # set optimizer
        self.opt = keras.optimizers.Adam()
        # compile
        self.compile_models()

    def compile_models(self):
        self.autoencoder.compile(optimizer=self.opt, loss=self.auteloss)
        self.VSH.compile(optimizer=self.opt, loss=[
                           self.VSHloss, self.VSHloss, self.VSHloss])

    def train_AutoE(self, train_data, test_data, epoch, batch_size=128):

        history = self.autoencoder.fit(
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

    def train_VSH(self, x_train, y_train, x_test, y_test, epoch, batch_size):

        history = self.VSH.fit(
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


def network_train(epoch_AutoE=10, batch_size_AutoE=32, epoch_VSH=10,
                  batch_size_VSH=32, AutoE_train=False, VSH_train=False):
    # loading dataset from pickle file
    data = scipy.io.loadmat('cylinder_nektar_wake.mat')
    # Make Maziar Rasisi's data shape compatable to ConVnetKeras
    U_star = data['U_star'] # N x 2 x T
    P_star = data['p_star'] # N x T
    t_star = data['t'] # T x 1
    X_star = data['X_star'] # N x 2

    N = X_star.shape[0]
    T = t_star.shape[0]

    # Rearrange Data 
    XX = np.tile(X_star[:,0:1], (1,T)) # N x T
    YY = np.tile(X_star[:,1:2], (1,T)) # N x T
    TT = np.tile(t_star, (1,N)).T # N x T

    UU = U_star[:,0,:] # N x T
    VV = U_star[:,1,:] # N x T
    PP = P_star # N x T

    x = np.zeros((200,50,100))
    y = np.zeros((200,50,100))
    u = np.zeros((200,50,100))
    v = np.zeros((200,50,100))
    p = np.zeros((200,50,100))

    for i in range(200):
        
        x[i, :, :] = np.reshape(XX[:,i],(50,100))
        y[i, :, :] = np.reshape(YY[:,i],(50,100))
        u[i, :, :] = np.reshape(UU[:,i],(50,100))
        v[i, :, :] = np.reshape(VV[:,i],(50,100))
        p[i, :, :] = np.reshape(PP[:,i],(50,100))
        
    t = TT[0,:].T

    dataset = (u, v, p, t)

    # Initializing our Neural Network class
    NN = ConvNet()

    # spiliting and Normilizing data
    Data_train, Data_test = NN.preprocess(dataset)

    # Determinig the grid size and label size from the data shape
    grid_size = Data_train[0][0].shape
    t_size = Data_train[3][0].shape
    print("dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd")
    print(t_size)
    # Create the Neural networks
    NN.create_model(grid_size, t_size, summary=True, auteloss="mse",
                    VSHloss="mse")

    # Train the Autoencoders
    if AutoE_train:
        history = NN.train_AutoE_u(
            Data_train[0], Data_test[0], epoch_AutoE, batch_size_AutoE)
        NN.autoencoder.save('model_autoencoder.h5')
        np.save('AutoE_history.npy', history)
    else:
        # load the autoencoder weight for transfer learning
        NN.autoencoder_u.load_weights('model_autoencoder.h5')

    # freeze the decoder's weights
    #NN.decoder_psi.trainable = False
    #NN.decoder_p.trainable = False
    # NN.compile_models()

    # Training the DLDNN network
    if VSH_train:
        history = NN.train_DLDNN(Data_train[3], [Data_train[0], Data_train[1], Data_train[2]],
                                 Data_test[3], [Data_test[0], Data_test[1], Data_test[2]], epoch_VSH, batch_size_VSH)
        NN.DLDNN.save('model_VSH.h5')
        np.save('VSH_history.npy', history)
    else:
        # load the DLDNN model
        NN.DLDNN.load_weights('model_DLDNN1.h5')
    
    # Make predictions by Autoencoder and DLDNN
    [u_AutE, v_AutE, p_AutE] = NN.autoencoder_v.predict(Data_test[0])[:, :, :, 0]
    [u_VSH, v_VSH, p_VSH] = NN.VSH.predict(Data_test[3])

    u_VSH = u_VSH[:, :, :, 0]
    v_VSH = v_VSH[:, :, :, 0]
    p_VSH = p_VSH[:, :, :, 0]

    # display original fields and predicted autoencoder and DLDNN result
    NN.display(Data_test[0], u_AutE, u_VSH)
    NN.display(Data_test[1], v_AutE, v_VSH)
    NN.display(Data_test[1], p_AutE, p_VSH)
    

epoch_AutoE = 30
batch_size_AutoE = 32
epoch_VSH = 20
batch_size_VSH = 32
Re = 100
grid_size = (50, 100)

network_train(epoch_AutoE, batch_size_AutoE, epoch_VSH, batch_size_VSH,
              AutoE_train=False, VSH_train=True)

