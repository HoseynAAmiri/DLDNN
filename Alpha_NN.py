import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import layers
from tensorflow.keras.models import Model

class Autoencoder():
    def __init__(self):
        pass
    def preprocess(self, input_array):
        
        max_holder = []
        for i in range(len(input_array)):
            max_holder.append(np.max(input_array[0], axis=(1,2), keepdims=True))
       
        output_array_u = input_array[0]/max_holder[0]
        output_array_v = input_array[1]/max_holder[1]
        output_array_lable = input_array[2]/max_holder[2]
        
        output_array = (output_array_u, output_array_v, output_array_lable)
        
        return output_array, max_holder
        
        
    def display(self, array1, array2, num_data=5, streamline=True):
        """
        Displays ten random images from each one of the supplied arrays.
        """
        indices = np.random.randint(len(array1), size=num_data)        
        images1 = array1[indices, :]
        images2 = array2[indices, :]       
        grid_size_in = array1[0].shape
        grid_size_out = array2[0].shape
        
        
        plt.figure(figsize=(20, 4))
        for i, (image1, image2) in enumerate(zip(images1, images2)):                
            ax = plt.subplot(2, num_data, i + 1)
            plt.imshow(image1.reshape(grid_size_in))
            plt.jet()
            if i ==0:    
                plt.title("Inputs")
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            
            
    
            ax = plt.subplot(2, num_data, i + 1 + num_data)
            plt.imshow(image2.reshape(grid_size_out))
            plt.jet()
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            if i ==0:    
                plt.title("Outputs")
    
        plt.show()
        return indices
    
    def create_model(self, input_shape, loss="binary_crossentropy", summary=False):
        
        input = layers.Input(shape=(input_shape[0], input_shape[1], 1))
        # Encoder
        X = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(input)
        X = layers.MaxPooling2D((2, 2), padding="same")(X)
        X = layers.Conv2D(32, (3, 3), activation="relu", padding="same")(X)
        encoded = layers.MaxPooling2D((2, 2), padding="same")(X)
        
        # Decoder
        X = layers.Conv2DTranspose(32, (3, 3), strides=2, activation="relu", padding="same")(encoded)
        X = layers.Conv2DTranspose(32, (3, 3), strides=2, activation="relu", padding="same")(X)
        decoded = layers.Conv2D(1, (3, 3), activation="linear", padding="same")(X)
        
        # Autoencoder
        self.autoencoder = Model(input, decoded)
        self.encoder = Model(input, encoded)
        self.decoder = Model(encoded, decoded)
        self.autoencoder.compile(optimizer="adam", loss=loss)
        
        if summary:
            self.autoencoder.summary()
        
        
    def train(self, train_data, test_data, epoch, batch_size=128):
        
        self.autoencoder.fit(
        x=train_data,
        y=train_data,
        epochs=epoch,
        batch_size=batch_size,
        shuffle=True,
        validation_data=(test_data, test_data),
        )
    
    def test(self, test_data, num_data=5):
        predictions = autoencoder.predict(test_data)
        display(test_data, predictions, num_data=num_data)
        
        
               
