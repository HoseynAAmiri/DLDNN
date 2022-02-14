
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np 
from tensorflow.keras.layers import Lambda, Dense, Input
from tensorflow.keras import Model
import tensorflow.keras.backend as K
import tensorflow as tf 
from keras.utils.vis_utils import plot_model

def grad(y, x, nameit):
    return Lambda(lambda z: tf.gradients(z[0], z[1], unconnected_gradients='zero')[0], name = nameit)([y,x])

def network(i):
    m = Dense(100, activation='sigmoid')(i)
    j = Dense(1, name="networkout", activation='relu')(m)
    return j

x1 = Input(shape=(1,))

a = network(x1)
b = grad(a, x1, "dudx1")
c = grad(b, x1, "dudx11")

model = Model(inputs = [x1], outputs=[c])
model.summary()
plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

model.compile(optimizer='rmsprop',
              loss='mean_squared_error',
              metrics=['accuracy'])

x1_data = np.random.random((1000, 1))
labels = np.zeros((1000,1))
model.fit(x1_data,labels, epochs=5)
