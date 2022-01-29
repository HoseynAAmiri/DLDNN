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

"""
grid_size = Data_Norm[0][0].shape
label_size = Data_Norm[2][0].shape
NN.create_model(grid_size, label_size, loss="mse", summary=True, plot=False)
#
#NN.train_AutoE(u_train, u_test, 10, batch_size=32)
#NN.autoencoder.save('model_autoencoder.h5')
#NN.train_DLDNN(label_train, u_train, label_test, u_test, 20, batch_size=32)
#NN.DLDNN.save('model_DLDNN.h5')

AutE = load_model('model_autoencod.h5')
DLDNN = load_model('model_DLDNN.h5')

u_AutE = AutE.predict(v_test)[:,:,:,0]
u_DLD = DLDNN.predict(label_test)[:,:,:,0]

NN.display(v_test, u_AutE, u_DLD)


# field prediction
# due the inconsistancy of between our code and CNN we had to change array's shape
#test = M.predict(u_test)[:,:,:,0]
#aute.display(u_test, test)




#encoded_imgs = aute.encoder.predict(u_test)
#aute.display(u_test, encoded_imgs[:,:,:,1])
#decoded_imgs = aute.decoder.predict(encoded_imgs)
#aute.display(u_test, decoded_imgs)
"""

