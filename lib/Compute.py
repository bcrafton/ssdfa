
import tensorflow as tf
import numpy as np

###################################################################

# TODO account for sparsity
# DONT USE THE SAME NAME FOR THE CLASS METHOD AS THE COUNTER
# >>> send(), send.

class Compute:

    def __init__(self):
        super().__init__()

    def mac(self, W, X):
        pass

    def add(self, X):
        pass

###################################################################           

class CMOS(Compute):

    def __init__(self):
        self.mac_count = 0
        self.add_count = 0

    def mac(self, X, Y):
        macs = tf.multiply(tf.multiply(tf.shape(X)[0], tf.shape(X)[1]), tf.shape(Y)[1])
        self.mac_count = tf.add(self.mac_count, macs)

    def add(self, X):
        self.add_count = tf.add(self.add_count, tf.reduce_prod(tf.shape(X)))

###################################################################           

class RRAM(Compute):

    def __init__(self):
        self.mac_count = 0
        self.add_count = 0

    def mac(self, W, X):
        macs = tf.multiply(tf.reduce_prod(tf.shape(W)), tf.shape(X)[0])
        self.mac_count = tf.add(self.mac_count, macs)

    def add(self, X):
        self.add_count = tf.add(self.add_count, tf.reduce_prod(tf.shape(X)))

###################################################################           

