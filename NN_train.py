# import sys
# sys.path.append('/content/drive/MyDrive/DLDNN/')

from keras.models import load_model
import matplotlib.pyplot as plt
from DLD_Utils import DLD_Util as utl
import numpy as np
from DLD_Util import DLD_Util as utl
from Alpha_NN import NeuralNetwork

utl = utl()
merged_data = utl.load_data('merged_data')

NN = NeuralNetwork()
Data_Norm, MAX_data = NN.preprocess(merged_data)
#np.save('MAX_data', MAX_data)
grid_size = Data_Norm[0][0].shape
label_size = Data_Norm[2][0].shape

NN.create_model(grid_size, label_size, loss="mse", summary=True, plot=False)

test_frac = 0.2
train_ix = np.random.choice(len(Data_Norm[0]), size=int(
    (1-test_frac)*len(Data_Norm[0])), replace=False)
test_ix = np.setdiff1d(np.arange(len(Data_Norm[0])), train_ix)
u_train, v_train, label_train = Data_Norm[0][train_ix], Data_Norm[1][train_ix], Data_Norm[2][train_ix]
u_test, v_test, label_test = Data_Norm[0][test_ix], Data_Norm[1][test_ix], Data_Norm[2][test_ix]

#
NN.train_AutoE(u_train, u_test, 10, batch_size=32)
NN.autoencoder.save('model_autoencoder.h5')
NN.train_DLDNN(label_train, u_train, label_test, u_test, 20, batch_size=32)
NN.DLDNN.save('model_DLDNN.h5')

AutE = load_model('model_autoencod.h5')
DLDNN = load_model('model_DLDNN.h5')

u_AutE = AutE.predict(u_test)[:,:,:,0]
u_DLD = DLDNN.predict(label_test)[:,:,:,0]

NN.display(u_test, u_AutE, u_DLD)


# field prediction
# due the inconsistancy of between our code and CNN we had to change array's shape
#test = M.predict(u_test)[:,:,:,0]
#aute.display(u_test, test)




#encoded_imgs = aute.encoder.predict(u_test)
#aute.display(u_test, encoded_imgs[:,:,:,1])
#decoded_imgs = aute.decoder.predict(encoded_imgs)
#aute.display(u_test, decoded_imgs)
