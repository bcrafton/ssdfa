
import numpy as np
import tensorflow as tf
import keras

(x_train, _), (_, _) = tf.keras.datasets.mnist.load_data()

x_train = np.reshape(x_train, (60000, 784))
x_train = x_train / 8.
# x_train = np.ceil(x_train).astype(int)
x_train = np.floor(x_train).astype(int)

batch = x_train[0:32]
batch = np.reshape(batch, -1)
batch = keras.utils.to_categorical(batch, 32)
print (np.shape(batch))
batch = np.reshape(batch, (32, 784, 32))


