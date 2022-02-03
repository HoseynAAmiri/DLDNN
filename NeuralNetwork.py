@ -1,232 +0,0 @@
from tensorflow.keras.models import Model
from tensorflow.keras import layers
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class NeuralNetwork():
    def __init__(self):
        pass

    def preprocess(self, input_array, test_frac = 0.2):
        
        # Spiliting data to train test sections
        train_ix = np.random.choice(len(input_array[0]), size=int(
            (1-test_frac)*len(input_array[0])), replace=False)
        test_ix = np.setdiff1d(np.arange(len(input_array[0])), train_ix)

        psi_train, p_train, label_train = np.nan_to_num(input_array[0][train_ix]), np.nan_to_num(input_array[1][train_ix]), np.nan_to_num(input_array[2][train_ix])
        psi_test, p_test, label_test = np.nan_to_num(input_array[0][test_ix]), np.nan_to_num(input_array[1][test_ix]), np.nan_to_num(input_array[2][test_ix])

        # Normilizing data and saving the Normilized value 
        
        Max_Train = []
        Max_Test = []
        
        Max_Train.append(np.max(psi_train, axis=(1,2), keepdims=True))
        Max_Train.append(np.max(p_train, axis=(1,2), keepdims=True))
        Max_Train.append(np.amax(label_train, axis=1))
        
        Max_Test.append(np.max(psi_test, axis=(1,2), keepdims=True))
        Max_Test.append(np.max(p_test, axis=(1,2), keepdims=True))
        Max_Test.append(np.amax(label_test, axis=1))
        
        output_psi_train = psi_train/Max_Train[0]
        output_p_train = p_train/Max_Train[1]
        output_label_train = label_train/Max_Train[2][:,None]
        output_train = (output_psi_train, output_p_train, output_label_train)
        
        output_psi_test = psi_test/Max_Test[0]
        output_p_test = p_test/Max_Test[1]
        output_label_test = label_test/Max_Test[2][:,None]
        output_test = (output_psi_test, output_p_test, output_label_test)

        return output_train, Max_Train, output_test, Max_Test

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
                     auteloss="mse", dldnnloss="mse", summary=False):

        self.auteloss  = auteloss        
        self.dldnnloss = dldnnloss

        encoder_input = layers.Input(
            shape=(input_shape_field[0], input_shape_field[1], 1), name="original_img")
        # Encoder
        X = layers.Conv2D(16, (3, 3), activation="relu",
                          padding="same")(encoder_input)
        X = layers.MaxPooling2D((2, 2), padding="same")(X)
        
        X = layers.Conv2D(16, (3, 3), activation="relu",
                          padding="same")(X)
        X = layers.MaxPooling2D((2, 2), padding="same")(X)
        

        X = layers.Conv2D(16, (3, 3), activation="relu", padding="same")(X)
        encoder_output = layers.MaxPooling2D((2, 2), padding="same")(X)
        self.encoder = Model(
            encoder_input, encoder_output, name="encoder")

        if summary:
            self.encoder.summary()
        # Decoder
        decoder_input = layers.Input(shape=(16, 16, 16), name="encoded_img")
        X = layers.Conv2DTranspose(
            16, (3, 3), strides=2, activation="relu",
            padding="same")(decoder_input)
        
        X = layers.Conv2DTranspose(
            16, (3, 3),strides=2, activation="relu",
            padding="same")(X)
        
        X = layers.Conv2DTranspose(
            16, (3, 3),strides=2, activation="relu",
            padding="same")(X)                  
        decoder_output = layers.Conv2D(
            1, (3, 3), activation="linear", padding="same")(X)

        self.decoder = Model(
            decoder_input, decoder_output, name="decoder")
        if summary:
            self.decoder.summary()

        # Autoencoder
        autoencoder_input = layers.Input(
            shape=(input_shape_field[0], input_shape_field[1], 1), name="img")
        encoded_img = self.encoder(autoencoder_input)
        decoded_img = self.decoder(encoded_img)
        self.autoencoder = Model(
            autoencoder_input, decoded_img, name="autoencoder")

        if summary:
            self.autoencoder.summary()

        # Fully Conncted Neural network
        FCNN_input = layers.Input(shape = input_shape_label,  name="labels")
       
        X = layers.Dense(64, activation="relu")(FCNN_input)
        X = layers.Dropout(0.2)(X)
        X = layers.Dense(512, activation="relu")(X)
        X = layers.Dropout(0.2)(X)
        X = layers.Dense(1024, activation="relu")(X)
        X = layers.Dropout(0.2)(X)
        X = layers.Dense(4096, activation="relu")(X)
        X = layers.Dropout(0.2)(X)
        FCNN_output = layers.Reshape((16, 16, 16))(X)

        self.FCNN = Model(FCNN_input, FCNN_output, name="FCNN")
        if summary:
            self.FCNN.summary()

        encoded_img = self.FCNN(FCNN_input)
        decoded_img = self.decoder(encoded_img)
        self.DLDNN = Model(FCNN_input, decoded_img, name="DLDNN")

        if summary:
            self.DLDNN.summary()

        # compile
        self.compile_models()
        

    def compile_models(self):
        self.autoencoder.compile(optimizer='adam', loss= self.auteloss)
        self.DLDNN.compile(optimizer='adam', loss= self.dldnnloss)

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

    def train_DLDNN(self, x_train, y_train, x_test, y_test, epoch, batch_size=128):

        history = self.DLDNN.fit(
            x=x_train,
            y=y_train,
            epochs=epoch,
            batch_size=batch_size,
            shuffle=True,
            validation_data=(x_test, y_test)
        )
        
        plt.figure()
        plt.subplot(1, 2, 2)
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


