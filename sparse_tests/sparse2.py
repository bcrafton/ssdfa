
# https://www.programcreek.com/python/example/90557/tensorflow.SparseTensor

import tensorflow as tf
import numpy as np
import math
import scipy

rate = 0.25
shape = (5, 5)
num = int(shape[0] * shape[1] * rate)
sqrt_fan_in = math.sqrt(shape[0])
mask = np.random.choice([0, 1], size=shape, p=[1.-rate, rate])
mat1 = np.random.uniform(low=-1., high=1., size=shape) * mask
mat2 = np.random.uniform(low=-1., high=1., size=shape)
res = np.dot(mat1, mat2)

################################################

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
tf.local_variables_initializer().run()

idx = tf.where(tf.abs(mat1) > 0)
idx = tf.cast(idx, tf.int64)
val = tf.gather_nd(mat1, idx)

x = tf.SparseTensor(indices=idx, values=val, dense_shape=shape)
z = tf.sparse_tensor_dense_matmul(x, mat2)

################################################

[_idx, _val, _z] = sess.run([idx, val, z], feed_dict={})
print (_idx, _val)
print (_z)
print (res)
