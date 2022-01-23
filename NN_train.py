# import sys
# sys.path.append('/content/drive/MyDrive/DLDNN/')

from DLD_Utils import DLD_Util as utl
import numpy as np
from Alpha_NN import Autoencoder

utl = utl()
merged_data = utl.load_data('merged_data')

aute = Autoencoder()
Data_Norm, MAX_data = aute.preprocess(merged_data)
np.save('MAX_data', MAX_data)
grid_size = Data_Norm[0][0].shape
aute.create_model(grid_size, loss="mse", summary=False)


test_frac = 0.2
train_ix = np.random.choice(len(Data_Norm[0]), size=int(
    (1-test_frac)*len(Data_Norm[0])), replace=False)
test_ix = np.setdiff1d(np.arange(len(Data_Norm[0])), train_ix)
u_train, v_train, label_train = Data_Norm[0][train_ix], Data_Norm[1][train_ix], Data_Norm[2][train_ix]
u_test, v_test, label_test = Data_Norm[0][test_ix], Data_Norm[1][test_ix], Data_Norm[2][test_ix]

# aute.train(u_train, u_test, 10, batch_size=32)
# aute.autoencoder.save('model_autoencod.h5')

from keras.models import load_model
M = load_model('model_autoencod.h5')

# field prediction
# due the inconsistancy of between our code and CNN we had to change array's shape
test = M.predict(u_test)[:,:,:,0]
aute.display(u_test, test)


#encoded_imgs = aute.encoder.predict(u_test)
#aute.display(u_test, encoded_imgs[:,:,:,1])
#decoded_imgs = aute.decoder.predict(encoded_imgs)
#aute.display(u_test, decoded_imgs)


