import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras import layers
from tensorflow import keras
from time import strftime
import numpy as np
from DLD_Utils import DLD_Utils as utl
utl=utl()

from tensorflow.python.framework.ops import disable_eager_execution
disable_eager_execution()


class PINN:
    def __init__(self, input_shape, output_shape, hidden_layers, summary=False):
        self.create_model(input_shape, output_shape, hidden_layers, summary=summary)
    @tf.function
    def create_model(self, input_shape, output_shape, hidden_layers, summary=False):
           
        input = layers.Input(shape=input_shape, name = "Network_Input")
        output_true = layers.Input(shape=output_shape, name="true_output")

        for i, layer in enumerate(hidden_layers):
            if i==0:
                X = layers.Dense(layer, activation="relu")(input)
                X = layers.Dropout(0.2)(X)
            else:
                X = layers.Dense(layer, activation="relu")(X)
                X = layers.Dropout(0.2)(X)

        output = layers.Dense(output_shape, activation='linear')(X)
        self.neural_net = Model(inputs=[input,output_true], outputs=output, name="neural_net")

        if summary:
            self.neural_net.summary()
        # set optimizer
        self.opt = keras.optimizers.Adam()
        

        def nested_loss(output_true, output, input):
            
            f_u, f_v = self.net_NS(input, output)
            psi_true = output_true[:, 0]
            psi_pred = output[:, 0]
            p_true = output_true[:, 1]
            p_pred = output[:, 1]

            self.loss_psi = K.sum(K.square(psi_true - psi_pred))
            self.loss_p = K.sum(K.square(p_true - p_pred))
            self.loss_f_u = K.sum(K.square(f_u))
            self.loss_f_v = K.sum(K.square(f_v))
            self.loss = self.loss_psi + self.loss_p + self.loss_f_u + self.loss_f_v

            return self.loss

            
        self.neural_net.add_loss(nested_loss(output_true, output, input))       
        self.neural_net.compile(optimizer=self.opt, loss=None)
    
    def net_NS(self, NN_input, NN_output):

        x = NN_input[:, 0:1]
        y = NN_input[:, 1:2]
        Re = NN_input[:, 5:6]

        psi = NN_output[:, 0:1]
        p = NN_output[:, 1:2] 
        # with tf.GradientTape() as tape3:
        #     with tf.GradientTape() as tape2:
        #         with tf.GradientTape() as tape1:

        #             x = NN_input[:, 0:1]
        #             y = NN_input[:, 1:2]
        #             Re = NN_input[:, 5:6]

        #             psi = NN_output[:, 0:1]
        #             p = NN_output[:, 1:2]

        #         u = tape1.gradient(psi, y)
        #         v = -tape1.gradient(psi, x)
        #         p_x = tape1.gradient(p, x)
        #         p_y = tape1.gradient(p, y)

        #     u_x = tape2.gradient(u, x)
        #     u_y = tape2.gradient(u, y)
        #     v_x = tape2.gradient(v, x)
        #     v_y = tape2.gradient(v, y)

        # u_xx = tape3.gradient(u_x, x)
        # u_yy = tape3.gradient(u_y, y)
        # v_xx = tape3.gradient(v_x, x)
        # v_yy = tape3.gradient(v_y, y)

        u = tf.gradients(psi, y)[0]
        v = -tf.gradients(psi, x)[0]

        u_x = tf.gradients(u, x)[0]
        u_y = tf.gradients(u, y)[0]
        u_xx = tf.gradients(u_x, x)[0]
        u_yy = tf.gradients(u_y, y)[0]

        v_x = tf.gradients(v, x)[0]
        v_y = tf.gradients(v, y)[0]
        v_xx = tf.gradients(v_x, x)[0]
        v_yy = tf.gradients(v_y, y)[0]

        p_x = tf.gradients(p, x)[0]
        p_y = tf.gradients(p, y)[0]

        f_u = (u*u_x + v*u_y) + p_x - (u_xx + u_yy) / Re
        f_v = (u*v_x + v*v_y) + p_y - (v_xx + v_yy) / Re

        return f_u, f_v
    
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
        
    def callback(self, loss_psi, loss_p, loss_f_u, loss_f_v):
        print('loss_psi: %.3e, loss_p: %.3e, loss_f_u: %.3e, loss_f_v: %.3e' %
              (loss_psi, loss_p, loss_f_u, loss_f_v))
            

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

    X_nn_train = np.concatenate([x_train, y_train, d_train, n_train, g_train, r_train], 1)
    y_nn_train = np.concatenate([s_train, p_train], 1)
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
    
    X_nn_test = np.concatenate([x_test, y_test, d_test, n_test, g_test, r_test], 1)
    y_nn_test = np.concatenate([s_test, p_test], 1)
    
    # Training
    model = PINN(X_nn_train.shape[1], y_nn_train.shape[1], hidden_layers, summary=False)
    
    model.train(X_nn_train, y_nn_train, X_nn_test, y_nn_test, epoch, batch_size)
    # saving
    model.save(f'models/PINN_{strftime("%Y-%m-%d_%H-%M-%S")}.model')

    