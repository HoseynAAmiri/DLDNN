from nntplib import NNTP
import os
from pickletools import uint1
from telnetlib import X3PAD
from tkinter.tix import X_REGION
from pyrsistent import freeze
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import Lambda, Add, Concatenate, Flatten
from tensorflow.keras.models import Model
import numpy as np
import matplotlib.pyplot as plt
from keras.utils import plot_model
from keras.models import load_model
from DLD_Utils import DLD_Utils as utl
from DLD_env import DLD_env, Pillar
import time
from tqdm import tqdm
np.random.seed(1234)
tf.random.set_seed(1234)

class DLD_Net:
    def __init__(self):
        self.checkpoint_filepath = './tmp9/checkpoint'
        x_grid_size = 128
        y_grid_size = 128
        self.grid_size = (x_grid_size, y_grid_size)
        xx = np.linspace(0, 1, x_grid_size)
        yy = np.linspace(0, 1, y_grid_size)
        x_grid, y_grid = np.meshgrid(xx, yy)

        self.dx = xx[1] - xx[0]
        self.dy = yy[1] - yy[0]
        
    def analyse_data(self, field, field_norm, n):
        idx = np.random.choice(len(field_norm[0]), size=n, replace=False)
        plt.figure()
        for i in range(len(idx)):
            plt.subplot(2, len(idx), i+1)
            plt.imshow(np.flip(field[idx[i]], axis=0))
            plt.colorbar()
            plt.jet()
            plt.subplot(2, len(idx), i+1+len(idx))
            plt.imshow(np.flip(field_norm[idx[i]], axis=0))
            plt.colorbar()
            plt.jet()
        plt.show()


    def create_model(self, label_shape, summary):
        # Neural Network 
        label_expansion_layer = 16
        def GenNet(input):
            # Expand input 
            X1 = layers.Dense(label_expansion_layer, activation="relu")(input[:, 0:1])
            X2 = layers.Dense(label_expansion_layer, activation="relu")(input[:, 1:2])
            X3 = layers.Dense(label_expansion_layer, activation="relu")(input[:, 2:3])
            X = layers.Concatenate(axis=1)([X1, X2, X3])
            # FCNN layers
            X = layers.Dense(256, activation="relu")(X)
            X = layers.Dense(256, activation="relu")(X)
            X = layers.Dense(16*16*64)(X)
            X = layers.ReLU()(X)
            # Reshape to mactch convolutional layer
            X = layers.Reshape((16, 16, 64))(X)
            X = layers.Conv2D(64, (3, 3),
                padding="same")(X)
            X = layers.ReLU()(X)

            # 1
            X = layers.UpSampling2D((2, 2))(X)
            
            X = layers.Conv2D(256, (3, 3),
                padding="same")(X)
            X = layers.ReLU()(X)

            # X = layers.Conv2D(128, (3, 3),
            #     padding="same")(X)
            # X = layers.ReLU()(X)
            # 2
            X = layers.UpSampling2D((2, 2))(X)
            
            X = layers.Conv2D(256, (3, 3),
                padding="same")(X)
            X = layers.ReLU()(X)

            # X = layers.Conv2D(128, (3, 3),
            #     padding="same")(X)
            # X = layers.ReLU()(X)

            # X = layers.Conv2D(128, (3, 3),
            #     padding="same")(X)
            # X = layers.ReLU()(X)

            # 4
            X = layers.UpSampling2D((2, 2))(X)
            
            # X = layers.Conv2D(128, (3, 3),
            #     padding="same")(X)
            # X = layers.ReLU()(X)

            X = layers.Conv2D(64, (3, 3),
                padding="same")(X)
            X = layers.ReLU()(X)

            X = layers.Conv2D(1, (3, 3), activation="tanh",
                padding="same")(X)

            return  X

        input = layers.Input(shape=label_shape,  name="labels")
        u = GenNet(input)
        v = GenNet(input)
        self.DLDNN = Model(inputs=input, outputs=[u, v], name="DLDNN")

        if summary:
            self.DLDNN.summary() 


    
    def train(self, u_train, v_train, label_train, u_test, v_test, label_test,
     epoch, N_EPOCH, batch_size, lr ):
        
        opt = keras.optimizers.Adam(learning_rate=lr, beta_1=0.5)
        self.DLDNN.compile(optimizer=opt, loss='mse')
        DLD_self = self
        class myCallback(keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                if (epoch+1) % N_EPOCH == 0:
                    u_DLD, v_DLD = DLD_self.DLDNN.predict(label_test)
                    u_DLD = u_DLD[:, :, :, 0]
                    v_DLD = v_DLD[:, :, :, 0]
                    DLD_self.display(u_test, u_DLD)
                    DLD_self.display(v_test, v_DLD)
                    # DLD_self.DLDNN.save('models/model_DLDNN_R10_50_{}.h5'.format(epoch))
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="./logs")
        cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=self.checkpoint_filepath, monitor='val_loss', mode='min', save_best_only=True, save_weights_only=True, verbose=1)
        callback_list = [myCallback(), tensorboard_callback, cp_callback]
        history = self.DLDNN.fit(
            x=label_train,
            y=[u_train, v_train],
            epochs=epoch,
            batch_size=batch_size,
            shuffle=False,
            validation_data=(label_test, [u_test, v_test]),
            callbacks=callback_list)
        
        
        plt.figure()
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.yscale('log')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()

        np.save('history', history.history)
        
    def display(self, Ground_truth, predicted, num_data=3, streamline=True):
    
        # Displays 'num_data' random images from each one of the supplied arrays.
        # Data should be normalized between -1 to 1 
        indices = np.random.randint(len(Ground_truth), size=num_data)
        images1 = Ground_truth[indices, :]
        images2 = predicted[indices, :]
        images3 = Ground_truth[indices, :] - predicted[indices, :]

        grid_size_oi = Ground_truth[0].shape
        grid_size_dld = predicted[0].shape

        plt.figure(figsize=(10, 4))
        for i, (image1, image2, image3) in enumerate(zip(images1, images2, images3)):

            ax = plt.subplot(3, num_data, i + 1)
            plt.imshow(image1.reshape(grid_size_oi))
            plt.jet()
            plt.colorbar()
            # plt.clim(-1, 1) 
            if i == 0:
                plt.title("GT")
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

            ax = plt.subplot(3, num_data, i + 1 + 1*num_data)
            plt.imshow(image2.reshape(grid_size_dld))
            plt.jet()
            plt.colorbar()
            # plt.clim(-1, 1) 
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            if i == 0:
                plt.title("Predicted")
            
            ax = plt.subplot(3, num_data, i + 1 + 2*num_data)
            plt.imshow(image3.reshape(grid_size_dld))
            plt.jet()
            plt.colorbar()
            # plt.clim(-1, 1) 
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            if i == 0:
                plt.title("Error")

        plt.show(block=False)
        plt.pause(10)
        plt.close()
        return indices
    
    def critical_dia(self, f, uv, dld, periods, tolerance):
        D = f
        G_X= 1-f
        
        x1 = 0.1
        x2 = 0.95

        _, modex1 = dld.simulate_particle(x1*G_X, uv, (0, (D/2+x1*G_X/2)), periods, plot=False)
        _, modex2 = dld.simulate_particle(x2*G_X, uv, (0, (D/2+x2*G_X/2)), periods, plot=False)

        if modex1 == -1 and modex2 == -1:
            return 1

        elif modex1 == -1 and modex2 == 1:
           
            while True:

                if (x2 - x1) < tolerance:
                    break

                x = (x1 + x2) * 0.5
                _, modex = dld.simulate_particle(x*G_X, uv, (0, (D/2+x*G_X/2)), periods, plot=False)
                
                if modex == 1:
                    x2 = x
                    modex2 = modex
                else:
                    x1 = x
                    modex1 = modex
                # if modex == 0:
                #     return x
        else:
            x = 0 

        return x

    def network_evaluation(self, plot_frac, dataset_norm, MAX):
         # specify the fraction of data you want your graph to be drawn        
        
        numElems = int(np.floor(plot_frac * len(dataset_norm[0])))
        idx = np.round(np.linspace(0, len(dataset_norm[0]) - 1, numElems)).astype(int)
        
        u_gt = dataset_norm[0][idx]
        v_gt = dataset_norm[1][idx]
        labels_norm = dataset_norm[2][idx] 
        labels = labels_norm * MAX[1]
        u_pred, v_pred = self.DLDNN.predict(labels_norm)
        u_pred = u_pred[:, :, :, 0]
        v_pred = v_pred[:, :, :, 0]
        
        d_gt = []
        d_pred = []
        
        pbar = tqdm(total=len(labels_norm), position=0, leave=True)
        for i in range(len(labels_norm)):
            f, N, Re = labels[i]
            
            pillar = Pillar(f, N)
            dld = DLD_env(pillar, Re, resolution=self.grid_size)
            
            uv_gt = (u_gt[i], v_gt[i])
            d_gt.append(self.critical_dia(f, uv_gt, dld, 1, 0.01))
            
            uv_pred = (u_pred[i], v_pred[i])

            d_pred.append(self.critical_dia(f, uv_pred, dld, 1, 0.01))
            pbar.update(1)
            time.sleep(0.1)
    
        plt.figure(figsize=(16, 9))
        plt.subplot(2, 2, 1)
        plt.scatter(d_pred, d_gt, c=np.array(d_gt)-np.array(d_pred))
        plt.colorbar()
        plt.plot([0, 1], [0, 1])
        
        plt.subplot(2, 2, 2)
        plt.scatter(labels[:, 0], d_gt)
        plt.scatter(labels[:, 0], d_pred)
        plt.xlabel('f')
        plt.ylabel('Critical Diameter')
        plt.legend(['GT', 'Prediction'], loc='upper right')

        plt.subplot(2, 2, 3)
        plt.scatter(labels[:, 1], d_gt)
        plt.scatter(labels[:, 1], d_pred)
        plt.xlabel('N')
        plt.ylabel('Critical Diameter')
        plt.legend(['GT', 'Prediction'], loc='upper right')

        plt.subplot(2, 2, 4)
        plt.scatter(labels[:, 2], d_gt)
        plt.scatter(labels[:, 2], d_pred)
        plt.xlabel('Re')
        plt.ylabel('Critical Diameter')
        plt.legend(['GT', 'Prediction'], loc='upper right')
        
        plt.savefig('eval_data.png')
        plt.show()
        return np.array([idx, labels[:,0], labels[:,1], labels[:,2], d_gt, d_pred]).T

        
    def strmline_comparison(self, dataset_norm, MAX, label_number, dp, periods, start_point):
    
        u_gt = dataset_norm[0][label_number]
        v_gt = dataset_norm[1][label_number]      
        uv_gt = (u_gt, v_gt)
    
        input = dataset_norm[2][label_number]
        f, N, Re = input * MAX[1]

        pillar = Pillar(f, N)
        self.dld = DLD_env(pillar, Re, resolution=self.grid_size)

        u, v = self.DLDNN.predict(input[None, :])
        u = u[0, :, :, 0]
        v = v[0, :, :, 0]
        uv = (u, v)

    
        plt.figure()
        plt.subplot(3,2,1)
        plt.imshow(np.flip(u_gt, axis=0))
        plt.colorbar()
        plt.jet()
        
        plt.subplot(3,2,3)
        plt.imshow(np.flip(u, axis=0)) 
        plt.colorbar()
        plt.jet()
    
        plt.subplot(3,2,5)
        plt.imshow(np.flip(u_gt-u, axis=0)) 
        plt.colorbar()
        plt.jet()
    
        plt.subplot(3,2,2)
        plt.imshow(np.flip(v_gt, axis=0)) 
        plt.colorbar()
        plt.jet()
    
        plt.subplot(3,2,4)
        plt.imshow(np.flip(v, axis=0)) 
        plt.colorbar()
        plt.jet()
    
        plt.subplot(3,2,6)
        plt.imshow(np.flip(v_gt-v, axis=0)) 
        plt.colorbar()
        plt.jet()
    
        plt.show()
    
        s1, m1 = self.dld.simulate_particle(dp*(1-f), uv, start_point, periods, plot=True)
        s2, m2 = self.dld.simulate_particle(dp*(1-f), uv_gt, start_point, periods, plot=True)

        print(len(s1[-1]), m1)
        print(len(s2[-1]), m2)


