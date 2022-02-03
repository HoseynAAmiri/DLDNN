from keras.models import load_model
import matplotlib.pyplot as plt
from DLD_Utils import DLD_Utils as utl
import numpy as np
from NeuralNetwork import NeuralNetwork

# Choosing the training part
AutoE_train = False
DLDNN_train = True

# Initialize the utility class
utl = utl()

# loading dataset from pickle file
dataset = utl.load_data('dataset128')

# Initializing our Neural Network class
NN = NeuralNetwork()

# spiliting and Normilizing data
Data_train, MAX_train, Data_test, MAX_test,  = NN.preprocess(dataset)
#np.save('MAX_train', MAX_train)
#np.save('MAX_test', MAX_test)

# Determinig the grid size and label size from the data shape
grid_size = Data_train[0][0].shape
label_size = Data_train[2][0].shape

# Create the Neural networks
NN.create_model(grid_size, label_size, auteloss="mse",
                dldnnloss="mse", summary=False)

# Train the Autoencoder
if AutoE_train:
    history = NN.train_AutoE(Data_train[0], Data_test[0], 50, batch_size=32)
    NN.autoencoder.save('model_autoencoder.h5')
    np.save('AutoE_history.npy',history)

#load the autoencoder weight for transfer learning 
NN.autoencoder.load_weights('model_autoencoder_T150.h5')
# freeze the decoder's weights
NN.decoder.trainable = False
NN.compile_models()

# Training the DLDNN network
if DLDNN_train:
    history = NN.train_DLDNN(Data_train[2], Data_train[0],
                   Data_test[2], Data_test[0], 100, batch_size=32)
    NN.DLDNN.save('model_DLDNN.h5')
    np.save('DLDNN_history.npy',history)
# load the DLDNN model
NN.DLDNN.load_weights('model_DLDNN.h5')
# Make predictions by Autoencoder and DLDNN 
psi_AutE = NN.autoencoder.predict(Data_test[0])[:, :, :, 0]
psi_DLD = NN.DLDNN.predict(Data_test[2])[:, :, :, 0]

# display original fields and predicted autoencoder and DLDNN result
NN.display(Data_test[0], psi_AutE, psi_DLD)

