
import numpy as np
import tensorflow as tf
from lib.Layer import Layer

class Activation(object):
    def forward(self, x):
        pass

    def gradient(self, x):
        pass
        
class Sigmoid(Activation):
    def __init__(self):
        pass

    def forward(self, x):
        return tf.sigmoid(x)

    def sigmoid_gradient(self, x):
        sig = tf.sigmoid(x)
        return tf.multiply(sig, tf.subtract(1.0, sig))
        
    def gradient(self, x):
        return tf.multiply(x, tf.subtract(1.0, x))
        
class Relu(Activation, Layer):

    def __init__(self):
        pass

    #############################################

    def forward(self, x):
        return tf.nn.relu(x)

    def gradient(self, x):
        # pretty sure this gradient works for A and Z
        return tf.cast(x > 0.0, dtype=tf.float32)

    #############################################

    def get_weights(self):
        return []

    def num_params(self):
        return 0

    #############################################

    def backward(self, AI, AO, DO):    
        return tf.cast(AO > 0.0, dtype=tf.float32) * DO

    def gv(self, AI, AO, DO):    
        return []
        
    def train(self, AI, AO, DO): 
        return []

    #############################################
    
    def dfa_backward(self, AI, AO, E, DO):
        return tf.cast(AO > 0.0, dtype=tf.float32) * DO

    def dfa_gv(self, AI, AO, E, DO):
        return []
        
    def dfa(self, AI, AO, E, DO): 
        return []

    #############################################
    
    def lel_backward(self, AI, AO, E, DO, Y):
        return tf.cast(AO > 0.0, dtype=tf.float32) * DO

    def lel_gv(self, AI, AO, E, DO, Y):
        return []
        
    def lel(self, AI, AO, E, DO, Y): 
        return []

    #############################################

# https://theclevermachine.wordpress.com/tag/tanh-function/ 
class Tanh(Activation):

    def __init__(self):
        pass

    def forward(self, x):
        return tf.tanh(x)

    def gradient(self, x):
        # this is gradient wtf A, not Z
        return 1 - tf.pow(x, 2)
        
# https://medium.com/@aerinykim/how-to-implement-the-softmax-derivative-independently-from-any-loss-function-ae6d44363a9d
# /home/brian/tensorflow/tensorflow/python/ops/nn_grad ... grep "_SoftmaxGrad"

class Softmax(Activation):
    def __init__(self):
        pass

    def forward(self, x):
        return tf.softmax(x)

    # this is gradient for A
    def gradient(self, x):
        # this is impossible and not bio plausible
        assert(False)
        
        flat = tf.reshape(x, [-1])
        diagflat = tf.diag(flat)
        dot = tf.matmul(flat, tf.transpose(flat))
        return diagflag - dot
        
class LeakyRelu(Activation):
    def __init__(self, leak=0.2):
        self.leak=leak

    def forward(self, x):
        return tf.nn.leaky_relu(x, alpha=self.leak)

    def gradient(self, x):
        # pretty sure this gradient works for A and Z
        return tf.add(tf.cast(x > 0.0, dtype=tf.float32), tf.cast(x < 0.0, dtype=tf.float32) * self.leak)
        
class SqrtRelu(Activation):
    def __init__(self):
        pass

    def forward(self, x):
        return tf.sqrt(tf.nn.relu(x))

    def gradient(self, x):
        # pretty sure this gradient works for A and Z
        return tf.cast(x > 0.0, dtype=tf.float32)
        
class Linear(Activation):

    def __init__(self):
        pass

    def forward(self, x):
        return x 

    def gradient(self, x):
        return tf.ones(shape=tf.shape(x))
       
        
        
        
        
