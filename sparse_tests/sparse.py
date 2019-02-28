
# https://www.programcreek.com/python/example/90557/tensorflow.SparseTensor

import tensorflow as tf
import numpy as np

rate = 0.1

_shape = np.array([1000, 1000])
num = int(rate * _shape[0] * _shape[1])
_idx_choices = []
for ii in range(_shape[0]):
    for jj in range(_shape[1]):
        _idx_choices.append([ii, jj])

_idx_choices = np.array(_idx_choices)

# T = [L[i] for i in Idx]
_idxs = np.random.choice(len(_idx_choices), size=num, replace=False)
_idxs = _idx_choices[_idxs]
_vals = np.random.rand(num)

print (np.shape(_idxs), np.shape(_vals), _shape)

################################################

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
tf.local_variables_initializer().run()

idxs = tf.placeholder(tf.int64, [num, 2])
vals = tf.placeholder(tf.float32, [num])
shape = tf.placeholder(tf.int64, [2])

################################################

x = tf.SparseTensor(indices=idxs, values=vals, dense_shape=shape)
y = tf.SparseTensor(indices=idxs, values=vals, dense_shape=shape)
# https://stackoverflow.com/questions/34030140/is-sparse-tensor-multiplication-implemented-in-tensorflow

a = tf.sparse_tensor_to_dense(y, validate_indices=False)
z = tf.sparse_tensor_dense_matmul(x, a)

################################################

[_z] = sess.run([z], feed_dict={idxs: _idxs, vals: _vals, shape: _shape})
print (_z)

