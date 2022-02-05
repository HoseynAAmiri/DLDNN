from keras.models import load_model
import matplotlib.pyplot as plt
from DLD_Utils import DLD_Utils as utl
import numpy as np
from NeuralNetwork import NeuralNetwork
from DLD_env import DLD_env, Pillar
# Initialize the utility class
utl = utl()

def network_train(AutoE_train=True, DLDNN_train=True):    
    
    # loading dataset from pickle file
    dataset = utl.load_data('dataset')
    
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
                    dldnnloss="mse", summary=True)
    
    # Train the Autoencoder
    if AutoE_train:
        history = NN.train_AutoE(Data_train[0], Data_test[0], 20, batch_size=32)
        NN.autoencoder.save('model_autoencoder.h5')
        np.save('AutoE_history.npy',history)
    
    #load the autoencoder weight for transfer learning 
    NN.autoencoder.load_weights('model_autoencoder.h5')
    # freeze the decoder's weights
    #NN.decoder.trainable = False
    #NN.compile_models()
    
    NN.DLDNN.load_weights('model_DLDNN.h5')
    # Training the DLDNN network
    if DLDNN_train:
        history = NN.train_DLDNN(Data_train[2], Data_train[0],
                       Data_test[2], Data_test[0], 20, batch_size=32)
        NN.DLDNN.save('model_DLDNN.h5')
        np.save('DLDNN_history.npy',history)
    # load the DLDNN model
    NN.DLDNN.load_weights('model_DLDNN.h5')
    # Make predictions by Autoencoder and DLDNN 
    psi_AutE = NN.autoencoder.predict(Data_test[0])[:, :, :, 0]
    psi_DLD = NN.DLDNN.predict(Data_test[2])[:, :, :, 0]
    
    # display original fields and predicted autoencoder and DLDNN result
    NN.display(Data_test[0], psi_AutE, psi_DLD)

def Network_evaluation(D, N, G_X, G_R, Re, grid_size, dp, start_point):
    NN = NeuralNetwork()
    label_size = 4
    NN.create_model(grid_size, label_size, auteloss="mse",
                    dldnnloss="mse", summary=False)
    NN.DLDNN.load_weights('model_DLDNN.h5')
    
    net_input = [D, N, G_X, Re]
    net_input_norm = net_input/np.max(net_input)
    
    pillar = Pillar(D, N, G_X, G_R)
    dld = DLD_env(pillar, Re, resolution=grid_size)
    
    psi = NN.DLDNN.predict(net_input_norm[None,:])[0,:,:,0]
    print(psi.shape)
    v, u = utl.gradient(psi, -dld.dx, dld.dy)
    plt.figure()
    plt.imshow(np.flip(psi,axis=0))
    plt.show()
    uv = (u, v)
    #dld.simulate_particle(dp/(D+G_X), uv, pillar.pillars, start_point, periods=6, plot=True)

D = 20
N = 5
G_X = 40
G_R = 1
Re = 1
grid_size =(128, 128)
start_point = (0, 0.4)
dp = 10
#network_train()
Network_evaluation(D, N, G_X, G_R, Re, grid_size, dp, start_point)
    