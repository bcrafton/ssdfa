
# https://www.programcreek.com/python/example/90557/tensorflow.SparseTensor

import tensorflow as tf
import numpy as np
import math

rate = 0.1
shape = (1000, 1000)
num = int(shape[0] * shape[1] * rate)
sqrt_fan_in = math.sqrt(shape[0])

################################################

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
tf.local_variables_initializer().run()

weights = tf.zeros(shape=shape)   
idx = tf.where(weights <= 0)
idx = tf.random_shuffle(idx)
idx = tf.cast(idx, tf.int32)
idx = tf.slice(idx, [0, 0], [num, 2])
val = tf.random_uniform(minval=-1.0/sqrt_fan_in, maxval=1.0/sqrt_fan_in, shape=(num,))
weights = tf.scatter_nd(indices=idx, updates=val, shape=shape)
val = tf.ones(shape=(num,))
mask = tf.scatter_nd(indices=idx, updates=val, shape=shape)

'''
x = tf.SparseTensor(indices=idxs, values=vals, dense_shape=shape)
y = tf.SparseTensor(indices=idxs, values=vals, dense_shape=shape)
# https://stackoverflow.com/questions/34030140/is-sparse-tensor-multiplication-implemented-in-tensorflow

a = tf.sparse_tensor_to_dense(y, validate_indices=False)
z = tf.sparse_tensor_dense_matmul(x, a)
'''

################################################

[_weights, _mask] = sess.run([weights, mask], feed_dict={})
print (np.shape(_weights), _weights, np.shape(_mask), _mask)

