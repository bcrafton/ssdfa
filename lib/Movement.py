
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


class vonNeumann(Movement):

    def __init__(self):
        self.send_count = tf.Variable(0, dtype=tf.int64)
        self.receive_count = tf.Variable(0, dtype=tf.int64)

    def send(self, X):
        sends = tf.cast(tf.reduce_prod(tf.shape(X)), tf.int64)
        self.send_count = tf.assign_add(self.send_count, sends)
        # self.send_count = tf.add(self.send_count, sends)

    def receive(self, X):
        receives = tf.cast(tf.reduce_prod(tf.shape(X)), tf.int64)
        self.receive_count = tf.assign_add(self.receive_count, receives)
        # self.receive_count = tf.add(self.receive_count, receives)

###################################################################           

class Neuromorphic(Movement):

    def __init__(self):
        self.send_count = tf.Variable(0, dtype=tf.int64)
        self.receive_count = tf.Variable(0, dtype=tf.int64)

    def send(self, X):
        sends = tf.cast(tf.reduce_prod(tf.shape(X)), tf.int64)
        self.send_count = tf.assign_add(self.send_count, sends)
        # self.send_count = tf.add(self.send_count, sends)

    def receive(self, X):
        receives = tf.cast(tf.reduce_prod(tf.shape(X)), tf.int64)
        self.receive_count = tf.assign_add(self.receive_count, receives)
        # self.receive_count = tf.add(self.receive_count, receives)

###################################################################           





