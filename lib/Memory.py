
import tensorflow as tf
import numpy as np

###################################################################

# TODO account for sparsity
# DONT USE THE SAME NAME FOR THE CLASS METHOD AS THE COUNTER
# >>> send(), send.

class Memory:

    def __init__(self):
        super().__init__()

    def read(self, X):
        pass

    def write(self, X):
        pass

###################################################################           

# DRAM / SRAM idk dosnt really make sense for SRAM
class DRAM(Memory):

    def __init__(self):
        self.read_count = tf.Variable(0, dtype=tf.int64)
        self.write_count = tf.Variable(0, dtype=tf.int64)

    def read(self, X):
        reads = tf.cast(tf.reduce_prod(tf.shape(X)), tf.int64)
        self.read_count = tf.assign_add(self.read_count, reads)
        # self.read_count = tf.add(self.read_count, reads)

    def write(self, X):
        writes = tf.cast(tf.reduce_prod(tf.shape(X)), tf.int64)
        self.write_count = tf.assign_add(self.write_count, writes)
        # self.write_count = tf.add(self.write_count, writes)

###################################################################           

class RRAM(Memory):

    def __init__(self):
        self.read_count = tf.Variable(0, dtype=tf.int64)
        self.write_count = tf.Variable(0, dtype=tf.int64)

    def read(self, X):
        reads = tf.cast(tf.reduce_prod(tf.shape(X)), tf.int64)
        self.read_count = tf.assign_add(self.read_count, reads)
        # self.read_count = tf.add(self.read_count, reads)

    def write(self, X):
        writes = tf.cast(tf.reduce_prod(tf.shape(X)), tf.int64)
        self.write_count = tf.assign_add(self.write_count, writes)
        # self.write_count = tf.add(self.write_count, writes)

###################################################################           

