
import tensorflow as tf
import numpy as np

###################################################################

# TODO account for sparsity
# DONT USE THE SAME NAME FOR THE CLASS METHOD AS THE COUNTER
# >>> send(), send.

class Movement:

    def __init__(self):
        super().__init__()

    def send(self, X):
        pass

    def receive(self, X):
        pass

###################################################################           


class vonNeumann:

    def __init__(self):
        self.send_count = 0
        self.receive_count = 0

    def send(self, X):
        self.send_count = tf.add(self.send_count, tf.reduce_prod(tf.shape(X)))

    def receive(self, X):
        self.receive_count = tf.add(self.receive_count, tf.reduce_prod(tf.shape(X)))

###################################################################           

class Neuromorphic:

    def __init__(self):
        self.send_count = 0
        self.receive_count = 0

    def send(self, X):
        self.send_count = tf.add(self.send_count, tf.reduce_prod(tf.shape(X)))

    def receive(self, X):
        self.receive_count = tf.add(self.receive_count, tf.reduce_prod(tf.shape(X)))

###################################################################           

