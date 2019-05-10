
import tensorflow as tf
import numpy as np
import keras

################################################

filter_size = 5
n = 4
data_size = 64

################################################

x = np.array(range(n * data_size))
x = np.reshape(x, (n, data_size))

################################################

xs = []
for ii in range(filter_size):
    start = ii 
    end = ii + data_size - filter_size + 1
    next = x[:, start:end]
    next = np.reshape(next, (n, -1, 1))
    xs.append(next)
    
xs = np.concatenate(xs, axis=2)
print (xs)
print (np.shape(xs))
