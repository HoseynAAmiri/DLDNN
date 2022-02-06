import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import time
from datetime import datetime
import numpy as np
from DLD_Utils import DLD_Utils as utl
utl=utl()


class PINN:
    def __init__(self, x, y, D, N, G, Re, psi, p, layers):

        tf.compat.v1.disable_eager_execution()

        X = np.concatenate([x, y, D, N, G, Re], 1)

        self.lb = X.min(0)
        self.ub = X.max(0)

        self.x = X[:, 0:1]
        self.y = X[:, 1:2]

        self.D = X[:, 2:3]
        self.N = X[:, 3:4]
        self.G = X[:, 4:5]
        self.Re = X[:, 5:6]

        self.psi = psi
        self.p = p

        self.layers = layers

        # Initialize NN
        self.weights, self.biases = self.initialize_NN(self.layers)

        # tf placeholders and graph
        self.sess = tf.compat.v1.Session()

        self.x_tf = tf.compat.v1.placeholder(
            tf.float32, shape=[None, self.x.shape[1]])
        self.y_tf = tf.compat.v1.placeholder(
            tf.float32, shape=[None, self.y.shape[1]])
        
        self.D_tf = tf.compat.v1.placeholder(
            tf.float32, shape=[None, self.D.shape[1]])
        self.N_tf = tf.compat.v1.placeholder(
            tf.float32, shape=[None, self.N.shape[1]])
        self.G_tf = tf.compat.v1.placeholder(
            tf.float32, shape=[None, self.G.shape[1]])
        self.Re_tf = tf.compat.v1.placeholder(
            tf.float32, shape=[None, self.Re.shape[1]])

        self.psi_tf = tf.compat.v1.placeholder(
            tf.float32, shape=[None, self.psi.shape[1]])
        self.p_tf = tf.compat.v1.placeholder(
            tf.float32, shape=[None, self.p.shape[1]])

        self.psi_pred, self.p_pred, self.f_u_pred, self.f_v_pred = self.net_NS(
            self.x_tf, self.y_tf, self.D_tf, self.N_tf, self.G_tf, self.Re_tf)

        self.loss_psi = tf.reduce_sum(tf.square(self.psi_tf - self.psi_pred))
        self.loss_p = tf.reduce_sum(tf.square(self.p_tf - self.p_pred))
        self.loss_f_u = tf.reduce_sum(tf.square(self.f_u_pred))
        self.loss_f_v = tf.reduce_sum(tf.square(self.f_v_pred))
        self.loss = self.loss_psi + self.loss_p + self.loss_f_u + self.loss_f_v

        self.optimizer_Adam = tf.compat.v1.train.AdamOptimizer()
        self.train_op_Adam = self.optimizer_Adam.minimize(self.loss)

        init = tf.compat.v1.global_variables_initializer()
        self.sess.run(init)

    def initialize_NN(self, layers):        
        weights = []
        biases = []
        num_layers = len(layers) 
        for l in range(0,num_layers-1):
            W = self.xavier_init(size=[layers[l], layers[l+1]])
            b = tf.Variable(tf.zeros([1,layers[l+1]], dtype=tf.float32), dtype=tf.float32)
            weights.append(W)
            biases.append(b)        
        return weights, biases
        
    def xavier_init(self, size):
        in_dim = size[0]
        out_dim = size[1]        
        xavier_stddev = np.sqrt(2/(in_dim + out_dim))
        return tf.Variable(tf.random.truncated_normal([in_dim, out_dim], stddev=xavier_stddev), dtype=tf.float32)
    
    @tf.function
    def neural_net(self, X, weights, biases):
        num_layers = len(weights) + 1
        
        H = 2.0*(X - self.lb)/(self.ub - self.lb) - 1.0
        for l in range(0,num_layers-2):
            W = weights[l]
            b = biases[l]
            H = tf.add(tf.matmul(H, W), b)
        W = weights[-1]
        b = biases[-1]
        Y = tf.add(tf.matmul(H, W), b)
        return Y

    def net_NS(self, x, y, D, N, G, Re):
        # Set up logging.
        # stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        # logdir = 'logs/func/%s' % stamp
        # writer = tf.summary.create_file_writer(logdir)

        X = tf.concat([x, y, D, N, G, Re], 1)

        # Bracket the function call with
        # tf.summary.trace_on() and tf.summary.trace_export().
        # tf.summary.trace_on(graph=True, profiler=True)
        # Call only one tf.function when tracing.

            
        psi_and_p = self.neural_net(X, self.weights, self.biases)
        # with writer.as_default():
        #     tf.summary.trace_export(
        #         name="Neural_net_trace",
        #         step=0,
        #         profiler_outdir=logdir)

        psi = psi_and_p[:, 0:1]
        p = psi_and_p[:, 1:2]

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

        return psi, p, f_u, f_v

    def callback(self, loss_psi, loss_p, loss_f_u, loss_f_v):
        print('loss_psi: %.3e, loss_p: %.3e, loss_f_u: %.3e, loss_f_v: %.3e' %
              (loss_psi, loss_p, loss_f_u, loss_f_v))
    
    def load_model(self, weights, biases):

        self.weights = weights
        self.biases = biases

    def train(self, nIter):

        tf_dict = {self.x_tf: self.x, self.y_tf: self.y, self.D_tf: self.D,
                   self.N_tf: self.N, self.G_tf: self.G, self.Re_tf: self.Re, self.psi_tf: self.psi, self.p_tf: self.p}

        start_time = time.time()
        for it in range(nIter):
            self.sess.run(self.train_op_Adam, tf_dict)

            # Print
            if it % 10 == 0:
                elapsed = time.time() - start_time
                
                loss_psi = self.sess.run(self.loss_psi, tf_dict)
                loss_p = self.sess.run(self.loss_p, tf_dict)
                loss_f_u = self.sess.run(self.loss_f_u, tf_dict)
                loss_f_v = self.sess.run(self.loss_f_v, tf_dict)
                loss = loss_psi + loss_p + loss_f_u + loss_f_v

                print('It: %d, Loss: %.3e, Time: %.2f' %
                      (it, loss, elapsed))
                self.callback(loss_psi, loss_p, loss_f_u, loss_f_v)

                start_time = time.time()
    
    def save_model(self, name='model'):
        saver = tf.compat.v1.train.Saver()
        saver.save(self.sess, name+".ckpt")

    def load_model(self, name='model'):
        saver = tf.compat.v1.train.Saver()
        saver.restore(self.sess, name+".ckpt")

    def predict(self, x, y, D, N, G, Re):
        tf_dict = {self.x_tf: x, self.y_tf: y, self.D_tf: D,
                   self.N_tf: N, self.G_tf: G, self.Re_tf: Re}

        psi = self.sess.run(self.psi_pred, tf_dict)
        p = self.sess.run(self.p_pred, tf_dict)

        return psi, p
    
        

if __name__ == "__main__":

    N_train = 1_000
    nIter = 10

    layers = [6, 20, 20, 20, 20, 20, 20, 20, 20, 20, 20, 2]

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
    idx = np.random.choice(N * N * L, N_train, replace=False)
    x_train = x[idx, :]
    y_train = y[idx, :]

    d_train = d[idx, :]
    n_train = n[idx, :]
    g_train = g[idx, :]
    r_train = r[idx, :]

    s_train = s[idx, :]
    p_train = p[idx, :]

    # Training
    model = PINN(x_train, y_train, d_train, n_train,
                 g_train, r_train, s_train, p_train, layers)
    model.train(nIter)
    # saving
    model.save_model(name='model')

    # Test Data
    snap = np.array([100])

    d_test = d[:,0,snap]
    n_test = n[:,0,snap]
    g_test = g[:,0,snap]
    r_test = r[:,0,snap]

    s_test = s[:,snap]
    p_test = p[:,snap]

    # Prediction
    s_pred, p_pred = model.predict(x, y, d_test, n_test, g_test, r_test)
    
    # Error
    error_s = np.linalg.norm(s_test-s_pred,2)/np.linalg.norm(s_test,2)
    error_p = np.linalg.norm(p_test-p_pred,2)/np.linalg.norm(p_test,2)

    print('Error psi: %e' % (error_s))       
    print('Error p: %e' % (error_p))    
    
    # Plot Results
#    plot_solution(x, psi_pred, 1)
#    plot_solution(x, p_pred, 2)
#    plot_solution(x, s_test - s_pred, 3)
#    plot_solution(x, p_test - p_pred, 4)
