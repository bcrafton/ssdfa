import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import keras
np.set_printoptions(threshold=np.inf)

#########################################################

mat = np.array(range(16))
mat = tf.Variable(mat, dtype=tf.float32)

mat = tf.reshape(mat, (16, 1))
mat = tf.tile(mat, [1, 4])
mat = tf.reshape(mat, (4, 8, 2))
mat = tf.transpose(mat, (0, 2, 1))
mat = tf.reshape(mat, (8, 8))

#########################################################

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
tf.local_variables_initializer().run()

#####################################################

[_mat] = sess.run([mat], feed_dict={})
print (_mat)

#####################################################






