import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import tensorflow.compat.v1 as tf1


class PINN:
    def __init__(self, X, Y, D, N, G, RE):
        
        self.X = X
        self.Y = Y
        self.D = D
        self.N = N
        self.G = G
        self.RE = RE

        self.dens = 1000
        self.visc = 0.001

        # tf placeholders and graph
        self.sess = tf1.Session(config=tf1.ConfigProto(allow_soft_placement=True,
                                                     log_device_placement=True))   

        self.x_tf = tf1.placeholder(tf.float32, shape=[None, self.x.shape[1]])
        self.y_tf = tf1.placeholder(tf.float32, shape=[None, self.y.shape[1]])
        self.t_tf = tf1.placeholder(tf.float32, shape=[None, self.t.shape[1]])
        
        self.u_tf = tf1.placeholder(tf.float32, shape=[None, self.u.shape[1]])
        self.v_tf = tf1.placeholder(tf.float32, shape=[None, self.v.shape[1]])
        
        self.u_pred, self.v_pred, self.p_pred, self.f_u_pred, self.f_v_pred = self.net_NS(self.x_tf, self.y_tf, self.t_tf)
        
        self.loss = tf.math.reduce_sum(tf.square(self.u_tf - self.u_pred)) + \
                    tf.math.reduce_sum(tf.square(self.v_tf - self.v_pred)) + \
                    tf.math.reduce_sum(tf.square(self.f_u_pred)) + \
                    tf.math.reduce_sum(tf.square(self.f_v_pred))
        
        self.optimizer = tfp.optimizer

        self.optimizer_Adam = tf1.train.AdamOptimizer()
        self.train_op_Adam = self.optimizer_Adam.minimize(self.loss)                    
        
        init = tf1.global_variables_initializer()
        self.sess.run(init)

    def predict(self, X, Y, D, N, G, RE):

        psi_and_p = self.neural_net(tf.concat([x,y,t], 1), self.weights, self.biases)
        psi = psi_and_p[:,0:1]
        p = psi_and_p[:,1:2]
        
        u = tf.gradients(psi, Y)[0]
        v = -tf.gradients(psi, X)[0]  
        
        u_x = tf.gradients(u, X)[0]
        u_y = tf.gradients(u, Y)[0]
        u_xx = tf.gradients(u_x, X)[0]
        u_yy = tf.gradients(u_y, Y)[0]
        
        v_x = tf.gradients(v, X)[0]
        v_y = tf.gradients(v, Y)[0]
        v_xx = tf.gradients(v_x, X)[0]
        v_yy = tf.gradients(v_y, Y)[0]
        
        p_x = tf.gradients(p, X)[0]
        p_y = tf.gradients(p, Y)[0]

        f_u = (u*u_x + v*u_y) + p_x - (u_xx + u_yy) / RE
        f_v = (u*v_x + v*v_y) + p_y - (v_xx + v_yy) / RE
        
        return u, v, p, f_u, f_v
