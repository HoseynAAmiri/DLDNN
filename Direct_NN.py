# System libraries
import os
import numpy as np 
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow import keras
import tensorflow as tf

# Personal libraries
from DLD_Utils import DLD_Utils as utl

np.random.seed(1234)
tf.random.set_seed(1234)

# Network Parameters
test_frac = 0.3
epoch = 1000
batch_size = 32
lr = 0.0001
summary = False

# Import the data 
dataset_name = "direct_dataset10400"
dataset = utl.load_data(dataset_name)

# Normalizing the data
# D criticals are normalized 
dataset_norm = []
dataset_norm.append(dataset[0] / np.max(dataset[0], axis=0))
dataset_norm.append(dataset[1])

# Creating train and test datasets
train_ix = np.random.choice(len(dataset_norm[0]), size=int(
    (1-test_frac)*len(dataset_norm[0])), replace=False)

test_ix = np.setdiff1d(np.arange(len(dataset_norm[0])), train_ix)

X_train, y_train = dataset_norm[0][train_ix], dataset_norm[1][train_ix]
X_test, y_test = dataset_norm[0][test_ix], dataset_norm[1][test_ix]

hidden_layers = [4, 5, 6, 8 ,10]
nodes = [8 , 10, 16, 32, 64, 128]

os.mkdir('history')
for hidden_layer in hidden_layers:
    for node in nodes:

        def netgen(input, hidden_layer, node):
            for i in range(hidden_layer):
                if i==0:
                    X = layers.Dense(node, activation="relu")(input)
                else:
                    X = layers.Dense(node, activation="relu")(X)
                    # X = layers.Dropout(0.2)(X)
            X = layers.Dense(1, activation="sigmoid")(X)
            return X

        input = layers.Input(shape=X_train[0].shape,  name="labels")
        output = netgen(input, hidden_layer, node)

        DNN = Model(inputs=input, outputs= output, name="DNN")
        DNN.summary()
        opt = keras.optimizers.Adam(learning_rate=lr)
        DNN.compile(optimizer=opt, loss='mse')

        history = DNN.fit(
            x=X_train,
            y=y_train,
            epochs=epoch,
            batch_size=batch_size,
            shuffle=False,
            validation_data=(X_test, y_test))

        DNN.save('models/DNN_model_hlayers{}_nodes_{}.h5'.format(hidden_layer, node))
        
        np.save('history/DNN_history_hlayers{}_nodes_{}'.format(hidden_layer, node), history.history)
        
