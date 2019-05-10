
import tensorflow as tf
import numpy as np
import keras

################################################

def conv1d_extract_patches(batch_size, data_size, filter_size, x):
    xs = []
    
    for ii in range(filter_size):
        start = ii 
        end = ii + data_size - filter_size + 1
        next = x[:, start:end]
        next = tf.reshape(next, (batch_size, -1, 1))
        xs.append(next)
        
    xs = tf.concat(xs, axis=2)
    return xs

################################################

filter_size = 5
batch_size = 4
data_size = 64

################################################

_x = np.array(range(batch_size * data_size))
_x = np.reshape(_x, (batch_size, data_size))

_f = np.array([0., 0., 1., 0., 0.])
_f = np.reshape(_f, (-1, 1))
_f = np.repeat(_f, repeats=batch_size, axis=1)
_f = np.reshape(_f, (4, 1, 5))

################################################

x = tf.Variable(_x, dtype=tf.float32)
f = tf.Variable(_f, dtype=tf.float32)
y = conv1d_extract_patches(batch_size=batch_size, data_size=data_size, filter_size=filter_size, x=x)
z = y * f

################################################

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
tf.local_variables_initializer().run()

################################################

[_z] = sess.run([z], feed_dict={})
print (np.shape(_z))

################################################

