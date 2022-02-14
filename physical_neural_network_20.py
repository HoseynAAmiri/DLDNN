import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np

# from tensorflow.python.framework.ops import disable_eager_execution
# disable_eager_execution()
import numpy as np 
from tensorflow.keras.layers import Lambda, Add, Dense, Input
from tensorflow.keras import Model
import tensorflow.keras.backend as K
import tensorflow as tf 
from keras.utils.vis_utils import plot_model

import tensorflow.keras as keras
from time import strftime

from DLD_Utils import DLD_Utils as utl
utl=utl()

class PINN:
    def __init__(self, input_shape, output_shape, hidden_layers, summary=False):
        self.create_model(input_shape, output_shape, hidden_layers, summary=summary)
    
    def create_model(self, input_shape, output_shape, hidden_layers, summary=False):
        
        # Define gradient function
        def grad(y, x, nameit):    
            return Lambda(lambda z: tf.gradients(z[0], z[1], unconnected_gradients='zero')[0], name=nameit)([y,x]) 
        
        # define network 
        def network_func(input, hidden_layers): 
            for i, layer in enumerate(hidden_layers):
                if i==0:
                    X = Dense(layer, activation="relu")(input)
                    #X = layers.Dropout(0.2)(X)
                else:
                    X = Dense(layer, activation="relu")(X)
                    #X = layers.Dropout(0.2)(X)
            output = Dense(output_shape, activation='relu')(X)
            return output
        
        input = Input(shape=(input_shape,), name = "Network_Input")

        output = network_func(input, hidden_layers)

        x = Lambda(lambda z: z[:,0], name='x')(input)
        y = Lambda(lambda z: z[:,1], name='y')(input)
        Re = Lambda(lambda z: z[:,5], name='Re')(input)
        
        psi = Lambda(lambda z: z[:,0], name='psi')(output)
        p = Lambda(lambda z: z[:,1], name='p')(output)

        u = grad(psi, y,'u')
        v = grad(-psi, x,'v')
        
        u_x = grad(u, x,'u_x')
        u_y = grad(u, y,'u_y')
        u_xx = grad(u_x, x,'u_xx')
        u_yy = grad(u_y, y,'u_yy')

        v_x = grad(v, x,'v_x')
        v_y = grad(v, y,'v_y')
        v_xx = grad(v_x, x,'v_xx')
        v_yy = grad(v_y, y,'v_yy')

        p_x = grad(p, x, 'p_x')
        p_y = grad(p, y, 'p_y')
        
        
        def NS_func(z, nameit):
            #f_u = (u*u_x + v*u_y) + p_x - (u_xx + u_yy) / Re
            #f_v = (u*v_x + v*v_y) + p_y - (v_xx + v_yy) / Re   
            return Lambda(lambda z:((z[0]*z[1]+z[2]*z[3])+z[4]-(z[5]+z[6]))/z[7], output_shape=[1], name=nameit)(z) 
        
        f_u = NS_func([u, u_x, v, u_y, p_x, u_xx, u_yy, Re], 'f_u')
        f_v = NS_func([u, v_x, v, v_y, p_y, v_xx, v_yy, Re], 'f_v')
        continuity = Add()([u_x, v_y])

        self.neural_net = Model(inputs=input, outputs=[psi, p, f_u, f_v, continuity], name="neural_net")
        
        
        if summary:
            self.neural_net.summary()
            plot_model(self.neural_net, to_file='PINN_plot.png', show_shapes=True, show_layer_names=True)
        # set optimizer
        self.opt = keras.optimizers.Adam()           
        self.neural_net.compile(optimizer=self.opt, loss='mse')
    
    
    def train(self, x_train, y_train, x_test, y_test, epoch, batch_size):
         # Stroing training logs
        history = self.neural_net.fit(
            x=x_train,
            y=y_train,
            epochs=epoch,
            batch_size=batch_size,
            shuffle=True,
            validation_data=(x_test, y_test))
    
        return history.history
        
    # def callback(self, loss_psi, loss_p, loss_f_u, loss_f_v):
    #     print('loss_psi: %.3e, loss_p: %.3e, loss_f_u: %.3e, loss_f_v: %.3e' %
    #           (loss_psi, loss_p, loss_f_u, loss_f_v))
            

if __name__ == "__main__":

    N_train = 100_000
    epoch = 10
    batch_size = 32
    hidden_layers = [20, 20, 20, 20, 20, 20, 20, 20, 20, 20]

    # Load Data
    dataset = utl.load_data('dataset')

    psi = dataset[0]  # L x N x N
    pre = dataset[1]  # L x N x N
    l = dataset[2]  # L x 4

    N = psi[0].shape[0]
    L = l.shape[0]

    # Rearrange Data
    xx = np.linspace(0, 1, N)
    yy = np.linspace(0, 1, N)
    x_grid, y_grid = np.meshgrid(xx, yy)  # N x N

    XX = np.tile(np.array([x_grid.flatten()]), (1, L))  # N2 x L
    YY = np.tile(np.array([y_grid.flatten()]), (1, L))  # N2 x L

    DD = np.tile(np.array([l[:, 0]]), (N * N, 1))  # N2 x L
    NN = np.tile(np.array([l[:, 1]]), (N * N, 1))  # N2 x L
    GG = np.tile(np.array([l[:, 2]]), (N * N, 1))  # N2 x L
    RR = np.tile(np.array([l[:, 3]]), (N * N, 1))  # N2 x L

    SS = np.array(psi).reshape(L, N * N).T  # N2 x L
    PP = np.array(pre).reshape(L, N * N).T  # N2 x L

    x = XX.flatten()[:, None]  # N2L x 1
    y = YY.flatten()[:, None]  # N2L x 1

    d = DD.flatten()[:, None]  # N2L x 1
    n = NN.flatten()[:, None]  # N2L x 1
    g = PP.flatten()[:, None]  # N2L x 1
    r = RR.flatten()[:, None]  # N2L x 1

    s = SS.flatten()[:, None]  # N2L x 1
    p = PP.flatten()[:, None]  # N2L x 1

    ######################################################################
    ######################## Noiseles Data ###############################
    ######################################################################
    # Training Data
    train_idx = np.random.choice(N * N * L, N_train, replace=False)
    
    x_train = x[train_idx, :]
    y_train = y[train_idx, :]

    d_train = d[train_idx, :]
    n_train = n[train_idx, :]
    g_train = g[train_idx, :]
    r_train = r[train_idx, :]

    s_train = s[train_idx, :]
    p_train = p[train_idx, :]
    # constraint
    c_train = np.zeros_like(s_train)

    X_nn_train = np.concatenate([x_train, y_train, d_train, n_train, g_train, r_train], 1)
    y_nn_train = np.concatenate([s_train, p_train, c_train, c_train, c_train], 1)
    # Test Data
    test_idx = np.setdiff1d(np.arange(N * N * L), train_idx)
    
    x_test = x[test_idx, :]
    y_test = y[test_idx, :]
    
    d_test = d[test_idx, :]
    n_test = n[test_idx, :]
    g_test = g[test_idx, :]
    r_test = r[test_idx, :]

    s_test = s[test_idx, :]
    p_test = p[test_idx, :]
    
    c_test = np.zeros_like(s_test)

    X_nn_test = np.concatenate([x_test, y_test, d_test, n_test, g_test, r_test], 1)
    y_nn_test = np.concatenate([s_test, p_test, c_test, c_test, c_test], 1)
    
    # Training
    print(X_nn_train.shape[1])
    model = PINN(X_nn_train.shape[1], y_nn_train.shape[1], hidden_layers, summary=True)
    
    print(model.neural_net.predict([[0, 0, 0, 0, 0, 0]]))

    # model.train(X_nn_train, y_nn_train, X_nn_test, y_nn_test, epoch, batch_size)
    # saving
    # model.save(f'models/PINN_{strftime("%Y-%m-%d_%H-%M-%S")}.model')

    