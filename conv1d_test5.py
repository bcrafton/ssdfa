
import tensorflow as tf
import numpy as np
import keras

################################################

def conv1d_extract_patches(input_shape, filter_shape, x):
    batch, time, input_size = input_shape
    fw, tmp = filter_shape
    assert(input_size == tmp)
        
    pad = fw // 2
    x = tf.pad(x, [[0, 0], [pad, pad], [0, 0]])
    time = time + 2 * pad
    
    xs = []
    for ii in range(fw):
        start = ii 
        end = ii + time - fw + 1
        next = x[:, start:end, :]
        next = tf.reshape(next, (batch_size, -1, 1, input_size))
        xs.append(next)
        
    xs = tf.concat(xs, axis=2)
    return xs

###################################################################

def conv1d(input_shape, filter_shape, x, f):
    batch, time, input_size = input_shape
    fw, tmp = filter_shape
    assert(input_size == tmp)
    
    # 50, 64, 5, 64
    patches = conv1d_extract_patches(input_shape=input_shape, filter_shape=filter_shape, x=x)
    _f = tf.reshape(f, (1, 1, fw, input_size))
    out = patches * _f
    out = tf.reduce_sum(out, axis=2)
    return out

################################################

filter_size = 5
batch_size = 16
time_size = 64
input_size = 64

################################################

_x = np.random.uniform(low=0., high=1., size=(batch_size, time_size, input_size))
_f = np.random.uniform(low=0., high=1., size=(filter_size, input_size))

################################################

x = tf.Variable(_x, dtype=tf.float32)
f = tf.Variable(_f, dtype=tf.float32)
z = conv1d(input_shape=(batch_size, time_size, input_size), filter_shape=(filter_size, input_size), x=x, f=f)

################################################

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
tf.local_variables_initializer().run()

################################################

[_z] = sess.run([z], feed_dict={})
print (np.shape(_z))

################################################

