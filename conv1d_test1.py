
import tensorflow as tf
import numpy as np
import keras

################################################

filter_size = 5
data_size = 64

################################################

x = np.array(range(data_size))

################################################

xs = []
for ii in range(filter_size):
    start = ii 
    end = ii + data_size - filter_size + 1
    next = x[start:end]
    next = np.reshape(next, (-1, 1))
    xs.append(next)
    
xs = np.concatenate(xs, axis=1)
print (xs)
