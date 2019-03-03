
# https://www.programcreek.com/python/example/90557/tensorflow.SparseTensor

import tensorflow as tf
import numpy as np
import time
import itertools
np.set_printoptions(threshold=np.inf)

rate = 0.05
N = 1000
itrs = 1000

num = int(rate * N * N)

combs = np.array(list(itertools.product(range(N), range(N))))
choices = range(len(combs))
_idxs = np.random.choice(a=choices, size=num, replace=False).tolist()
_idxs = combs[_idxs]
_idxs = _idxs.tolist()
_idxs = sorted(_idxs)
# print (_idxs)
# print (np.shape(_idxs))
_vals = np.float32(np.random.rand(num))
_y = np.random.uniform(low=-1., high=1., size=(N, N))
_z = np.random.uniform(low=-1., high=1., size=(N, N))

################################################

# supposed to use reorder.
x = tf.SparseTensor(indices=_idxs, values=_vals, dense_shape=(N, N))
y = tf.Variable(_y, dtype=tf.float32)
z = tf.Variable(_z, dtype=tf.float32)

ret = tf.sparse_tensor_dense_matmul(x, y)
# ret = tf.sparse_tensor_dense_matmul(x, y, adjoint_a=False, adjoint_b=False)
# ret = tf.matmul(z, y)

################################################

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
tf.local_variables_initializer().run()

start = time.time()
for i in range(itrs):
    print (i)
    [_ret] = sess.run([ret], feed_dict={})
print (time.time() - start)

################################################

