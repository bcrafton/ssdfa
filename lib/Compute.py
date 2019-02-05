
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
        self.mac_count = tf.Variable(0, dtype=tf.int64)
        self.add_count = tf.Variable(0, dtype=tf.int64)

    def mac(self, X, Y):
        macs = tf.cast(tf.multiply(tf.multiply(tf.shape(X)[0], tf.shape(X)[1]), tf.shape(Y)[1]), tf.int64)
        self.mac_count = tf.assign_add(self.mac_count, macs)
        # self.mac_count = tf.add(self.mac_count, macs)

    def add(self, X):
        adds = tf.cast(tf.reduce_prod(tf.shape(X)), tf.int64)
        self.add_count = tf.assign_add(self.add_count, adds)
        # self.add_count = tf.add(self.add_count, adds)

###################################################################           

class RRAM(Compute):

    def __init__(self):
        self.mac_count = tf.Variable(0, dtype=tf.int64)
        self.add_count = tf.Variable(0, dtype=tf.int64)

    def mac(self, X, Y):
        macs = tf.cast(tf.multiply(tf.multiply(tf.shape(X)[0], tf.shape(X)[1]), tf.shape(Y)[1]), tf.int64)
        self.mac_count = tf.assign_add(self.mac_count, macs)
        # self.mac_count = tf.add(self.mac_count, macs)

    def add(self, X):
        adds = tf.cast(tf.reduce_prod(tf.shape(X)), tf.int64)
        self.add_count = tf.assign_add(self.add_count, adds)
        # self.add_count = tf.add(self.add_count, adds)

###################################################################           

