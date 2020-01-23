
import numpy as np
import tensorflow as tf
# import tensorflow_probability as tfp
from lib.Layer import Layer

###################################################################

def quantize_activations(a):
  # scale = (15 - 0) / (tfp.stats.percentile(a, 95) - tfp.stats.percentile(a, 5))
  scale = (15 - 0) / (tf.reduce_max(a) - tf.reduce_min(a))
  # scale = (15 - 0) / (2 * tf.math.reduce_std(a))

  a = scale * a
  a = tf.floor(a)
  a = tf.clip_by_value(a, 0, 15)
  return a, scale
  
def quantize_activations2(a, scale):
  a = scale * a
  a = tf.floor(a)
  a = tf.clip_by_value(a, 0, 15)
  return a, scale
  
###################################################################

class Relu(Layer):

    def __init__(self, scale):
        self.scale = scale

    #########

    def get_weights(self):
        return []
        
    def num_params(self):
        return 0

    ###################################################################

    def forward(self, x):
        A = tf.nn.relu(x)
        A, scale = quantize_activations(A)
        A = A / scale
        return A, (scale,)

    def forward1(self, x):
        A = tf.nn.relu(x)
        A, scale = quantize_activations(A)
        # A = A / scale
        return A, (scale,)

    def forward2(self, x):
        A = tf.nn.relu(x)
        A, scale = quantize_activations2(A, self.scale)
        # A = A / scale
        return A, (scale,)

    ###################################################################

    def bp(self, AI, AO, DO, cache):    
        DI = tf.cast(AO > 0.0, dtype=tf.float32) * DO
        return DI, []

    def dfa(self, AI, AO, E, DO, cache):
        return self.bp(AI, AO, DO, cache)
        
    def lel(self, AI, AO, DO, Y, cache): 
        return self.bp(AI, AO, DO, cache)

###################################################################

        
        
