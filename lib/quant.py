
import tensorflow as tf
# import tensorflow_probability as tfp
import numpy as np

###################

xlow = 0
xhigh = 15

wlow = -8
whigh = 7

alow = 0
ahigh = 15

###################

def quantize_x(x):
  scale = (np.max(x) - np.min(x)) / (xhigh - xlow)
  x = x / scale
  x = np.floor(x)
  x = np.clip(x, xlow, xhigh)
  return x, scale

###################

def quantize_conv(w):
  scale = (tf.reduce_max(w) - tf.reduce_min(w)) / (whigh - wlow)
  w = w / scale
  w = tf.floor(w)
  w = tf.clip_by_value(w, wlow, whigh)
  return w, scale

def quantize_conv_bias(w):
  scale = (tf.reduce_max(w) - tf.reduce_min(w)) / (whigh - wlow)
  w = w / scale
  w = tf.floor(w)
  w = tf.clip_by_value(w, wlow, whigh)
  return w, scale
  
def quantize_conv_activations(a):
  scale = (tf.reduce_max(a) - tf.reduce_min(a)) / (ahigh - alow)
  a = a / scale
  a = tf.floor(a)
  a = tf.clip_by_value(a, alow, ahigh)
  return a, scale
  
def quantize_conv_activations2(a, scale):
  a = a / scale
  a = tf.floor(a)
  a = tf.clip_by_value(a, alow, ahigh)
  return a, scale
  
###################

def quantize_dense(w):
  scale = (tf.reduce_max(w) - tf.reduce_min(w)) / (whigh - wlow)
  w = w / scale
  w = tf.floor(w)
  w = tf.clip_by_value(w, wlow, whigh)
  return w, scale

def quantize_dense_bias(w):
  scale = (tf.reduce_max(w) - tf.reduce_min(w)) / (whigh - wlow)
  w = w / scale
  w = tf.floor(w)
  w = tf.clip_by_value(w, wlow, whigh)
  return w, scale

def quantize_dense_activations(a):
  scale = (tf.reduce_max(a) - tf.reduce_min(a)) / (ahigh - alow)
  a = a / scale
  a = tf.floor(a)
  a = tf.clip_by_value(a, alow, ahigh)
  return a, scale
  
def quantize_dense_activations2(a, scale):
  a = a / scale
  a = tf.floor(a)
  a = tf.clip_by_value(a, alow, ahigh)
  return a, scale
  
###################
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
