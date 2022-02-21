import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras import Model
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Lambda, Add, Multiply, Subtract, Dense, Input
from tensorflow.keras.utils import plot_model

def gradient(y, x, order=1, name='gradient'):

    g = Lambda(lambda z: tf.gradients(z[0], z[1], unconnected_gradients='zero')[0], name=name)
    ds = y
    for _ in range(order):
        ds = g([ds, x])

    return ds

# initializer = tf.keras.initializers.Ones()
# kernel_initializer=initializer,
def network(i):

    nn = Dense(5, activation='tanh', name='h1')(i)
    nn = Dense(5, activation='tanh', name='h2')(nn)
    nn = Dense(1, activation='linear', name='out')(nn)

    return nn

x1 = Input(shape=1)
x2 = Input(shape=1)

nn = network(x1)

g1 = gradient(nn, x1, order=1, name="g1")
g2 = gradient(nn, x1, order=2, name="g2")
g3 = gradient(nn, x1, order=3, name="g3")


model = Model(inputs=[x1], outputs=[nn, g1, g2, g3])
plot_model(model, to_file='PINN_Base_plot.png', show_shapes=True, show_layer_names=True)

# model.summary()

x = np.linspace(-100, 100, 10001) / 100
y = x ** 2
# dydx = 2 * x
dydx = 2 * x
dy2dx2 = np.ones_like(x) * 2
dy3dx3 = np.zeros_like(x)

# x = np.linspace(1, 10, 10).reshape(10, 1)
# print(model.predict(x))

model.compile(optimizer = tf.keras.optimizers.Adam(), loss='mse')

# hist = model.fit(x, [y, dydx, dy2dx2, dy3dx3], batch_size=256, epochs=400)
# model.save('model_quad_i.h5')
model.load_weights('model_quad_i.h5')
y_pred = model.predict(2*x)

import matplotlib.pyplot as plt

plt.figure()
plt.subplot(2, 2, 1)
plt.plot([0, 1], [0, 1])
plt.scatter(y, y_pred[0])

plt.subplot(2, 2, 2)
plt.scatter(x, y_pred[1])

plt.subplot(2, 2, 3)
plt.scatter(x, y_pred[2])

plt.subplot(2, 2, 4)
plt.scatter(x, y_pred[3])
plt.show()

print(model.predict([-10, -8, -4, -2, 1, 2, 4, 8, 10]))