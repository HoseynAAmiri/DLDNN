# import sys
# sys.path.append('/content/drive/MyDrive/DLDNN/')

from keras.models import load_model
import matplotlib.pyplot as plt
from DLD_Utils import DLD_Utils as utl
import numpy as np
from DLD_Utils import DLD_Utils as utl
from NeuralNetwork import NeuralNetwork

utl = utl()

# loading dataset from pickle file
dataset = utl.load_data('dataset')

# Initializing our Neural Network class
NN = NeuralNetwork()

# spiliting and Normilizing data
Data_train, MAX_train, Data_test, MAX_test,  = NN.preprocess(dataset)
#np.save('MAX_train', MAX_train)
#np.save('MAX_test', MAX_test)

grid_size = Data_train[0][0].shape
label_size = Data_train[2][0].shape
NN.create_model(grid_size, label_size, auteloss="mse",
                dldnnloss="mse", summary=True, plot=False)

NN.train_AutoE(Data_train[0], Data_test[0], 25
               , batch_size=32)
NN.autoencoder.save('model_autoencoder.h5')
#NN.train_DLDNN(Data_train[2], Data_train[0], Data_test[2], Data_test[0], 50, batch_size=32)
#NN.DLDNN.save('model_DLDNN.h5')

AutE = load_model('model_autoencoder.h5', compile = False)
DLDNN = load_model('model_DLDNN.h5', compile = False)

psi_AutE = AutE.predict(Data_test[0])[:, :, :, 0]
psi_DLD = DLDNN.predict(Data_test[2])[:, :, :, 0]

NN.display(Data_test[0], psi_AutE, psi_DLD)


# field prediction
# due the inconsistancy of between our code and CNN we had to change array's shape
#test = M.predict(u_test)[:,:,:,0]
#aute.display(u_test, test)


#encoded_imgs = aute.encoder.predict(u_test)
#aute.display(u_test, encoded_imgs[:,:,:,1])
#decoded_imgs = aute.decoder.predict(encoded_imgs)
#aute.display(u_test, decoded_imgs)
