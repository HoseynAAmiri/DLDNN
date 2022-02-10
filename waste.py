import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow.python.framework.ops import disable_eager_execution
disable_eager_execution()
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras import layers
from tensorflow import keras
from DLD_Utils import DLD_Utils as utl
utl=utl()

hidden_layers = [20, 20, 20, 20, 20]
input_shape = 6
output_shape = 2
           
input = keras.Input(shape=input_shape, name = "Network_Input")
X = layers.Dense(hidden_layers[0], activation="relu")(input)

for layer in hidden_layers:
    X = layers.Dense(layer, activation="relu")(X)

output = layers.Dense(output_shape, activation='linear')(X)
neural_net = Model(input, output, name="neural_net")

# neural_net.summary()
print(K.gradients(output[0], output[0])[0])