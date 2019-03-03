
# https://www.programcreek.com/python/example/90557/tensorflow.SparseTensor

import tensorflow as tf
import numpy as np
import time
import itertools
np.set_printoptions(threshold=np.inf)

################################################

N = 1000
itrs = 100
_y = np.random.uniform(low=-1., high=1., size=(N, N))

y = tf.Variable(_y, dtype=tf.float32)
z = tf.matmul(y, y)

################################################

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
tf.local_variables_initializer().run()

start = time.time()
for i in range(itrs):
    print (i)
    [_z] = sess.run([z], feed_dict={})
print (time.time() - start)



