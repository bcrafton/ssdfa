
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
        self.read_count = 0
        self.write_count = 0

    def read(self, X):
        self.read_count = tf.add(self.read_count, tf.reduce_prod(tf.shape(X)))

    def write(self, X):
        self.write_count = tf.add(self.write_count, tf.reduce_prod(tf.shape(X)))

###################################################################           

class RRAM(Memory):

    def __init__(self):
        self.read_count = 0
        self.write_count = 0

    def read(self, X):
        self.read_count = tf.add(self.read_count, tf.reduce_prod(tf.shape(X)))

    def write(self, X):
        self.write_count = tf.add(self.write_count, tf.reduce_prod(tf.shape(X)))

###################################################################           

