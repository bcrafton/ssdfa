import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import keras
np.set_printoptions(threshold=np.inf)

#########################################################

mat = range(16 * 5)
mat = np.reshape(mat, (16, 5))
'''
mat = range(100)
mat = np.array(mat)
'''
mat = tf.Variable(mat, dtype=tf.float32)
mat = tf.reverse(mat, axis=[1])

#########################################################

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
tf.local_variables_initializer().run()

#####################################################

[_mat] = sess.run([mat], feed_dict={})
print (_mat)

#####################################################






