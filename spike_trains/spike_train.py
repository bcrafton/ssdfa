
import tensorflow as tf
import keras
import numpy as np

##############################

def to_spike_train(mat, times):
    shape = np.shape(mat)
    assert(len(shape) == 2)
    N, O = shape
    mat = np.reshape(mat, (N, 1, O))
    
    out_shape = N, times, O
    train = np.random.uniform(low=0.0, high=1.0, size=out_shape)
    train = train < mat
    
    return train

##############################

(x_train, _), (_, _) = tf.keras.datasets.mnist.load_data()

x_train = x_train.reshape(60000, 784)
x_train = x_train.astype('float32')
x_train /= 255.

##############################

start = 0
end = 64
times = 64
xs = x_train[start:end]
xs = to_spike_train(xs, times)

##############################

print (np.shape(xs))
