
import tensorflow as tf
import keras
import numpy as np

(x_train, _), (_, _) = tf.keras.datasets.mnist.load_data()
x_train = np.reshape(x_train, (60000, 784))

trains = []
for ii in range(32):
    cmp_arr = np.random.uniform(low=0., high=1., size=(1, 32, 1))
    next = (cmp_arr < (ii / 32.)) * 1.0
    next = next.T
    trains.append(next)
    
x_train = np.ceil(x_train / 8.).astype(int)
batch = x_train[0:32]
batch = np.reshape(batch, -1)
print (np.shape(batch))
batch = trains[batch]
batch = np.reshape(batch, (32, 784, 64))

