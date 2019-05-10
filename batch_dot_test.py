
import tensorflow as tf
import numpy as np
import keras

################################################

def to_spike_train(mat, times):
    shape = np.shape(mat)
    assert(len(shape) == 2)
    N, O = shape
    mat = np.reshape(mat, (N, 1, O))
    
    out_shape = N, times, O
    train = np.random.uniform(low=0.0, high=1.0, size=out_shape)
    train = train < mat
    
    return train

################################################

(x_train, _), (_, _) = tf.keras.datasets.mnist.load_data()

x_train = x_train.reshape(60000, 784)
x_train = x_train.astype('float32')
x_train /= 255.

################################################

start = 0
end = 64
times = 64
_x = x_train[start:end]
_x = to_spike_train(_x, times)

_w = np.ones(shape=(784, 64))

################################################

x = tf.Variable(_x, dtype=tf.float32)
w = tf.Variable(_w, dtype=tf.float32)
# o = tf.keras.backend.batch_dot(x, w)
# o = tf.tensordot(x, w, axes=)
o = tf.keras.backend.dot(x, w)

################################################
    
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
tf.local_variables_initializer().run()
    
################################################

[_o] = sess.run([o], feed_dict={})

################################################

print (np.shape(_x))
print (np.shape(_w))
print (np.shape(_o))






