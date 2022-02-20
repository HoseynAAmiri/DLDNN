from ast import Sub
import os
from tkinter import Y
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np

# from tensorflow.python.framework.ops import disable_eager_execution
# disable_eager_execution()
import numpy as np 
from tensorflow.keras.layers import Lambda, Add, Multiply, Concatenate, Subtract, Dense, Input
from tensorflow.keras import Model
import tensorflow.keras.backend as K
import tensorflow as tf 
from keras.utils.vis_utils import plot_model
import tensorflow.keras as keras
from time import strftime
import matplotlib.pyplot as plt
from DLD_Utils import DLD_Utils as utl
utl=utl()

class PINN:
    def __init__(self, input_shape, output_shape, hidden_layers, summary=False):
        self.create_model(input_shape, output_shape, hidden_layers, summary=summary)
    
    def create_model(self, input_shape, output_shape, hidden_layers, summary=False):
        
        # Define gradient function
        def gradient(y, x, order=1, name='gradient'):

            g = Lambda(lambda z: tf.gradients(z[0], z[1], unconnected_gradients='zero')[0], name=name)
            for _ in range(order):
                y = g([y, x])

            return y

        # define network
        #initializer = tf.keras.initializers.GlorotUniform()# , kernel_initializer=initializer
        def network_func(input, hidden_layers): 

            for i, layer in enumerate(hidden_layers):
                if i==0:
                    X = Dense(layer, activation="tanh")(input)
                else:
                    X = Dense(layer, activation="tanh")(X)

            output = Dense(output_shape, activation='linear')(X)

            return output
        
        input = [Input(shape=1) for _ in range(input_shape)]
        conc = Concatenate()(input)

        psi = network_func(conc, hidden_layers)
        p_x = network_func(conc, hidden_layers)
        p_y = network_func(conc, hidden_layers)

        x = input[0]
        y = input[1]
        Re = input[5]

        u = gradient(psi, y, name='u')
        v = gradient(-psi, x, name='v')
        
        u_x = gradient(u, x, name='u_x')
        u_y = gradient(u, y, name='u_y')
        u_xx = gradient(u_x, x, name='u_xx')
        u_yy = gradient(u_y, y, name='u_yy')

        v_x = gradient(v, x, name='v_x')
        v_y = gradient(v, y, name='v_y')
        v_xx = gradient(v_x, x, name='v_xx')
        v_yy = gradient(v_y, y, name='v_yy')

        def NS_func(z, name):
            # f_u = (u*u_x + v*u_y) * Re + p_x - (u_xx + u_yy)
            # f_v = (u*v_x + v*v_y) * Re + p_y - (v_xx + v_yy)  
            return Lambda(lambda z: ((z[0]*z[1]+z[2]*z[3])*z[4]+z[5]-(z[6]+z[7])), name=name)(z) 
        
        f_u = NS_func([u, u_x, v, u_y, Re, p_x, u_xx, u_yy], 'f_u')
        f_v = NS_func([u, v_x, v, v_y, Re, p_y, v_xx, v_yy], 'f_v')

        continuity = Add(name='continuity')([u_x, v_y])

        self.neural_net = Model(inputs=input, outputs=[psi, p_x, p_y, f_u, f_v, continuity], name="neural_net")
        
        if summary:
            # self.neural_net.summary()
            plot_model(self.neural_net, to_file='PINN3_plot.png', show_shapes=True, show_layer_names=True)
        
        # set optimizer
        self.opt = keras.optimizers.Adam()
        # mse = keras.losses.MeanSquaredError()           
        # losses = {"psi":mse, "p":mse}
        #, , "f_u":mse, "f_v":mse, "continuity":mse
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
        
        plt.figure()
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.yscale('log')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()
    
        return history.history
           

if __name__ == "__main__":

    N_train = 100_000
    epoch =  1000
    batch_size = 256
    hidden_layers = 10*[25]
    
    # Load Data
    dataset = utl.load_data('dataset')

    psi = dataset[0]  # L x N x N
    p_x = dataset[1]  # L x N x N
    p_y = dataset[2]  # L x N x N
    l = dataset[3]  # L x 4

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
    PPx = np.array(p_x).reshape(L, N * N).T  # N2 x L
    PPy = np.array(p_y).reshape(L, N * N).T  # N2 x L

    x = XX.flatten()[:, None]  # N2L x 1
    y = YY.flatten()[:, None]  # N2L x 1

    d = DD.flatten()[:, None]  # N2L x 1
    n = NN.flatten()[:, None]  # N2L x 1
    g = GG.flatten()[:, None]  # N2L x 1
    r = RR.flatten()[:, None]  # N2L x 1

    s = SS.flatten()[:, None]  # N2L x 1
    px = PPx.flatten()[:, None]  # N2L x 1
    py = PPy.flatten()[:, None]  # N2L x 1

    dn = d/np.max(np.abs(d))
    nn = n/np.max(np.abs(n))
    gn = g/np.max(np.abs(g))
    rn = r/np.max(np.abs(r))
    
    sn = s/np.max(np.abs(s))
    pnx = px/np.max(np.abs(px))
    pny = py/np.max(np.abs(py))

    ######################################################################
    ######################## Noiseles Data ###############################
    ######################################################################
    # Training Data
    train_idx = np.random.choice(N * N * L, N_train, replace=False)
    
    x_train = x[train_idx, :]
    y_train = y[train_idx, :]

    d_train = dn[train_idx, :]
    n_train = nn[train_idx, :]
    g_train = gn[train_idx, :]
    r_train = rn[train_idx, :]

    s_train = sn[train_idx, :]
    px_train = pnx[train_idx, :]
    py_train = pny[train_idx, :]
    
    # constraints
    c_train = np.zeros_like(s_train)

    X_nn_train = [x_train, y_train, d_train, n_train, g_train, r_train]
    #y_nn_train = np.concatenate([s_train, px_train, py_train, c_train, c_train, c_train], 1)
    y_nn_train = [s_train, px_train, py_train, c_train, c_train, c_train]
    
    # Test Data
    test_idx = np.setdiff1d(np.arange(N * N * L), train_idx)[0:1000]
    
    x_test = x[test_idx, :]
    y_test = y[test_idx, :]
    
    d_test = dn[test_idx, :]
    n_test = nn[test_idx, :]
    g_test = gn[test_idx, :]
    r_test = rn[test_idx, :]

    s_test = sn[test_idx, :]
    px_test = pnx[test_idx, :]
    py_test = pny[test_idx, :]

    c_test = np.zeros_like(s_test)

    X_nn_test = [x_test, y_test, d_test, n_test, g_test, r_test]
    # y_nn_test = np.concatenate([s_test, px_test, py_test, c_test, c_test, c_test], 1)
    y_nn_test = [s_test, px_test, py_test, c_test, c_test, c_test]

    # Training
    model = PINN(6, 1, hidden_layers, summary=True)
    
    model.train(X_nn_train, y_nn_train, X_nn_test, y_nn_test, epoch, batch_size)
    
    # saving
    model.neural_net.save(f'models/PINN_{strftime("%Y-%m-%d_%H-%M-%S")}.model')
    
    # A = model.neural_net.predict(X_norm_train[0:20])
    