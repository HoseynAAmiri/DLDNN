import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import tf.compat.v1 as tf1
import numpy as np
import time
from DLD_Utils import DLD_Utils as utl
utl=utl()

class PINN:
    def __init__(self, x, y, D, N, G, Re, psi, p, layers):
        
        self.X = X
        self.Y = Y
        self.D = D
        self.N = N
        self.G = G
        self.Re = Re
        self.layers = layers
        
        # Initialize NN
        self.weights, self.biases = self.initialize_NN(self.layers)  
        

        # tf placeholders and graph
        self.sess = tf1.Session(config=tf1.ConfigProto(allow_soft_placement=True,
                                                     log_device_placement=True))
        self.x_tf = tf1.placeholder(tf.float32, shape=[None, self.x.shape[1]])
        self.y_tf = tf1.placeholder(tf.float32, shape=[None, self.y.shape[1]])
        self.D_tf = tf1.placeholder(tf.float32, shape=[None, self.D.shape[1]])
        self.N_tf = tf1.placeholder(tf.float32, shape=[None, self.N.shape[1]])
        self.G_tf = tf1.placeholder(tf.float32, shape=[None, self.G.shape[1]])
        self.Re_tf = tf1.placeholder(tf.float32, shape=[None, self.Re.shape[1]])

        self.psi_tf = tf1.placeholder(tf.float32, shape=[None, self.psi.shape[1]])
        self.p_tf = tf1.placeholder(tf.float32, shape=[None, self.p.shape[1]])
        
        self.psi_pred, self.p_pred, self.f_u_pred, self.f_v_pred = self.net_NS(self.x_tf, self.y_tf, self.D, self.N, self.G, self.Re, self.layers)

        self.loss_psi = tf.math.reduce_sum(tf.math.square(self.psi_tf - self.psi_pred))
        self.loss_p =  tf.math.reduce_sum(tf.math.square(self.p_tf - self.p_pred))
        self.loss_f_u =  tf.math.reduce_sum(tf.math.square(self.f_u_pred))
        self.loss_f_v =  tf.math.reduce_sum(tf.math.square(self.f_v_pred))
        self.loss = self.loss_psi + self.loss_p + self.loss_f_u + self.loss_f_v
        
        self.optimizer_Adam = tf1.train.AdamOptimizer()
        self.train_op_Adam = self.optimizer_Adam.minimize(self.loss)                    
         
        init = tf1.global_variables_initializer()
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
        return tf.Variable(tf1.truncated_normal([in_dim, out_dim], stddev=xavier_stddev), dtype=tf.float32)
    
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

       psi_and_p = self.neural_net(tf.concat([x,y], 1), self.weights, self.biases)
       psi = psi_and_p[:,0:1]
       p = psi_and_p[:,1:2]
       
       u = tf1.gradients(psi, y)[0]
       v = -tf1.gradients(psi, x)[0]  
       
       u_x = tf1.gradients(u, x)[0]
       u_y = tf1.gradients(u, y)[0]
       u_xx = tf1.gradients(u_x, x)[0]
       u_yy = tf1.gradients(u_y, y)[0]
       
       v_x = tf1.gradients(v, x)[0]
       v_y = tf1.gradients(v, y)[0]
       v_xx = tf1.gradients(v_x, x)[0]
       v_yy = tf1.gradients(v_y, y)[0]
       
       p_x = tf1.gradients(p, x)[0]
       p_y = tf1.gradients(p, y)[0]

       f_u = (u*u_x + v*u_y) + p_x - (u_xx + u_yy) / Re
       f_v = (u*v_x + v*v_y) + p_y - (v_xx + v_yy) / Re
       
       return psi, p, f_u, f_v
   
    def callback(self, loss_psi, loss_p, loss_f_u, loss_f_v):
        print('loss_psi: %.5f, loss_p: %.5f, loss_f_u: %.5f, loss_f_v: %.5f' % (loss_psi, loss_p, loss_f_u, loss_f_v))
        
        
    def train(self, nIter): 

        tf_dict = {self.x_tf: self.x, self.y_tf: self.y, self.D_tf: self.D,
                   self.N_tf: self.N, self.G_tf: self.G, self.Re_tf: self.Re, self.psi_tf: self.psi, self.p_tf: self.p}
        
        start_time = time.time()
        for it in range(nIter):
            self.sess.run(self.train_op_Adam, tf_dict)
            
            # Print
            if it % 10 == 0:
                elapsed = time.time() - start_time
                loss = self.sess.run(self.loss, tf_dict)
                # self.callback()
            
                print('It: %d, Loss: %.3e, Time: %.2f' % 
                      (it, loss, elapsed))
                start_time = time.time()
            
        # self.optimizer.minimize(self.sess,
        #                         feed_dict = tf_dict,
        #                         fetches = [self.loss],
        #                         loss_callback = self.callback)
        
        
    def predict(self, x, y, D, N, G, Re):
        
        tf_dict = {self.x_tf: self.x, self.y_tf: self.y, self.D_tf: self.D,
                   self.N_tf: self.N, self.G_tf: self.G, self.Re_tf: self.Re}
        
        psi = self.sess.run(self.psi_pred, tf_dict)
        p = self.sess.run(self.p_pred, tf_dict)
        
        return psi, p
    

if __name__ == "__main__": 
      
    N_train = 100,000
    
    layers = [3, 20, 20, 20, 20, 20, 20, 20, 20, 2]
    
    # Load Data
    dataset = utl.load_data('dataset')
           
    psi = dataset[0] # l x N x N
    p = dataset[1] # l x N x N
    l = dataset[2] # l x 4
    
    N = psi[0].shape[0]
    L = l.shape[0]
    
    # Rearrange Data 
    xx = np.linspace(0, 1, N)
    yy = np.linspace(0, 1, N)
    x_grid, y_grid = np.meshgrid(xx, yy)
    
    XX = np.tile(x_grid.flatten(), (1,L)) # N2 x l
    YY = np.tile(y_grid.flatten(), (1,L)) # N2 x l
    DD = 
    
    UU = U_star[:,0,:] # N x T
    VV = U_star[:,1,:] # N x T
    PP = P_star # N x T
    
    x = XX.flatten()[:,None] # NT x 1
    y = YY.flatten()[:,None] # NT x 1
    t = TT.flatten()[:,None] # NT x 1
    
    u = UU.flatten()[:,None] # NT x 1
    v = VV.flatten()[:,None] # NT x 1
    p = PP.flatten()[:,None] # NT x 1
    
    '''
    ######################################################################
    ######################## Noiseles Data ###############################
    ######################################################################
    # Training Data    
    idx = np.random.choice(N*T, N_train, replace=False)
    x_train = x[idx,:]
    y_train = y[idx,:]
    t_train = t[idx,:]
    u_train = u[idx,:]
    v_train = v[idx,:]

    # Training
    model = PhysicsInformedNN(x_train, y_train, t_train, u_train, v_train, layers)
    model.train(200000)
    
    # Test Data
    snap = np.array([100])
    x_star = X_star[:,0:1]
    y_star = X_star[:,1:2]
    t_star = TT[:,snap]
    
    u_star = U_star[:,0,snap]
    v_star = U_star[:,1,snap]
    p_star = P_star[:,snap]
    
    # Prediction
    u_pred, v_pred, p_pred = model.predict(x_star, y_star, t_star)
    lambda_1_value = model.sess.run(model.lambda_1)
    lambda_2_value = model.sess.run(model.lambda_2)
    
    # Error
    error_u = np.linalg.norm(u_star-u_pred,2)/np.linalg.norm(u_star,2)
    error_v = np.linalg.norm(v_star-v_pred,2)/np.linalg.norm(v_star,2)
    error_p = np.linalg.norm(p_star-p_pred,2)/np.linalg.norm(p_star,2)

    error_lambda_1 = np.abs(lambda_1_value - 1.0)*100
    error_lambda_2 = np.abs(lambda_2_value - 0.01)/0.01 * 100
    
    print('Error u: %e' % (error_u))    
    print('Error v: %e' % (error_v))    
    print('Error p: %e' % (error_p))    
    print('Error l1: %.5f%%' % (error_lambda_1))                             
    print('Error l2: %.5f%%' % (error_lambda_2))                  
    
    # Plot Results
#    plot_solution(X_star, u_pred, 1)
#    plot_solution(X_star, v_pred, 2)
#    plot_solution(X_star, p_pred, 3)    
#    plot_solution(X_star, p_star, 4)
#    plot_solution(X_star, p_star - p_pred, 5)
    
    # Predict for plotting
    lb = X_star.min(0)
    ub = X_star.max(0)
    nn = 200
    x = np.linspace(lb[0], ub[0], nn)
    y = np.linspace(lb[1], ub[1], nn)
    X, Y = np.meshgrid(x,y)
    
    UU_star = griddata(X_star, u_pred.flatten(), (X, Y), method='cubic')
    VV_star = griddata(X_star, v_pred.flatten(), (X, Y), method='cubic')
    PP_star = griddata(X_star, p_pred.flatten(), (X, Y), method='cubic')
    P_exact = griddata(X_star, p_star.flatten(), (X, Y), method='cubic')
    ''' 
