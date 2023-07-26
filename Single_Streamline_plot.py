# Design automation procedure consist of getting the diameters of
# particles for separation as input and propose dld design

import numpy as np
from keras.models import load_model


from DLD_Utils import DLD_Utils as utl
from Conv_Base import DLD_Net
from DLD_env import DLD_env, Pillar  


########################################################################
# Here we can plot streamline plot of a single device ##################
########################################################################



D1 = 8 # Bigger particle diameter [um]
D2 = 5  # Smaller particle diameter [um]
f = 0.25
N = 9.0
Re = 1.28
G = 15.79 
norm_vec = [0.75, 10, 25]

# Do you want to see the streamline of the two particles
streamline = True
num_per = 10 # Number of periods


#Loading Neural Network
DNN = load_model('DNN_model_hlayers8_nodes_128.h5')




# Find the optimum label for given input 
opt_label = np.zeros((len(norm_vec))) # Optimum results' labels

opt_label[0] = f
opt_label[1] = N
opt_label[2] = Re


print(opt_label[0:3])

if streamline:
    NN = DLD_Net()
    label_shape = opt_label[0:3].shape
    NN.create_model(label_shape, summary=False)
    NN.DLDNN.load_weights(NN.checkpoint_filepath)
    f =  opt_label[0]
    
    periods = num_per


    pillar = Pillar(opt_label[0], opt_label[1])
    NN.dld = DLD_env(pillar, opt_label[2], resolution=NN.grid_size)

    input = opt_label[0:3] / norm_vec
    u, v = NN.DLDNN.predict(input[None, :])
    u = u[0, :, :, 0]
    v = v[0, :, :, 0]
    uv = (u, v)
    
    dp = D1 / G
    start_point = (0, f/2+dp*(1-f)/2)
    s1, m1 = NN.dld.simulate_particle(dp*(1-f), uv, start_point, periods, plot=True)

    dp = D2 / G
    start_point = (0, f/2+dp*(1-f)/2)
    s1, m1 = NN.dld.simulate_particle(dp*(1-f), uv, start_point, periods, plot=True)
