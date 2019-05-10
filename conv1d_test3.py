
import tensorflow as tf
import numpy as np
import keras

################################################

def conv1d_transform(batch_size, data_size, filter_size, x):
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

################################################

x = tf.Variable(_x, dtype=tf.float32)
o = conv1d_transform(batch_size=batch_size, data_size=data_size, filter_size=filter_size, x=x)

################################################

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
tf.local_variables_initializer().run()

################################################

[_o] = sess.run([o], feed_dict={})
print (_o)

################################################

