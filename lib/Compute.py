
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

    def matmult(self, shape_X, shape_Y):
        self.mac_count += shape_X[0] * shape_X[1] * shape_Y[1]

    def conv(self, shape_X, shape_Y):
        self.mac_count += shape_X[0] * shape_X[1] * shape_Y[1]

    def add(self, shape_X, shape_Y):
        self.add_count += np.prod(shape_X) # need to do np.prod(np.max(shape_X, shape_Y)) or something like this

###################################################################           

class RRAM(Compute):

    def __init__(self):
        self.mac_count = 0
        self.add_count = 0

    def mult(self, shape_X, shape_Y):
        self.mac_count += shape_X[0] * shape_X[1] * shape_Y[1]

    def conv(self, shape_X, shape_Y):
        self.mac_count += shape_X[0] * shape_X[1] * shape_Y[1]

    def add(self, shape_X, shape_Y):
        self.add_count += np.prod(shape_X) # need to do np.prod(np.max(shape_X, shape_Y)) or something like this

###################################################################           

