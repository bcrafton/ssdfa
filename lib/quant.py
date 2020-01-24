
import tensorflow as tf
# import tensorflow_probability as tfp
import numpy as np

###################

def quantize_conv(w):
  scale = (tf.reduce_max(w) - tf.reduce_min(w)) / (7. - (-8.))
  w = w / scale
  w = tf.floor(w)
  w = tf.clip_by_value(w, -8, 7)
  return w, scale

def quantize_conv_bias(w):
  scale = (tf.reduce_max(w) - tf.reduce_min(w)) / (1023. - (-1024.))
  w = w / scale
  w = tf.floor(w)
  w = tf.clip_by_value(w, -1024, 1023)
  return w, scale
  
def quantize_conv_activations(a):
  scale = (tf.reduce_max(a) - tf.reduce_min(a)) / (15 - 0)
  a = a / scale
  a = tf.floor(a)
  a = tf.clip_by_value(a, 0, 15)
  return a, scale
  
def quantize_conv_activations2(a, scale):
  a = a / scale
  a = tf.floor(a)
  a = tf.clip_by_value(a, 0, 15)
  return a, scale
  
###################

def quantize_dense(w):
  scale = (tf.reduce_max(w) - tf.reduce_min(w)) / (7. - (-8.))
  w = w / scale
  w = tf.floor(w)
  w = tf.clip_by_value(w, -8, 7)
  return w, scale

def quantize_dense_bias(w):
  scale = (tf.reduce_max(w) - tf.reduce_min(w)) / (1023. - (-1024.))
  w = w / scale
  w = tf.floor(w)
  w = tf.clip_by_value(w, -1024, 1023)
  return w, scale

def quantize_dense_activations(a):
  scale = (tf.reduce_max(a) - tf.reduce_min(a)) / (7. - (-8.))
  a = a / scale
  a = tf.floor(a)
  a = tf.clip_by_value(a, -8, 7)
  return a, scale
  
def quantize_dense_activations2(a, scale):
  a = a / scale
  a = tf.floor(a)
  a = tf.clip_by_value(a, -8, 7)
  return a, scale
  
###################
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
