
import tensorflow as tf
import numpy as np
import keras

################################################

_x = np.ones(shape=(64, 64, 64))
_f = np.ones(shape=(1, 64, 1))

################################################

x = tf.Variable(_x, dtype=tf.float32)
f = tf.Variable(_f, dtype=tf.float32)
o = tf.nn.conv1d(value=x, filters=f, stride=1, padding='SAME')
    
################################################
    
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
tf.local_variables_initializer().run()
    
################################################

[_o] = sess.run([o], feed_dict={})

################################################

print (np.shape(_x))
print (np.shape(_f))
print (np.shape(_o))
