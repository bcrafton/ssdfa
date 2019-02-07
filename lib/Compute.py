
import tensorflow as tf
import numpy as np

###################################################################

# TODO account for sparsity
# DONT USE THE SAME NAME FOR THE CLASS METHOD AS THE COUNTER
# >>> send(), send.

class Compute:

    def __init__(self):
        super().__init__()

    def matmult(self, W, X):
        pass

    def conv(self, W, X):
        pass

    def add(self, X):
        pass

###################################################################           

class CMOS(Compute):

    def __init__(self):
        self.mac_count = 0
        self.add_count = 0

    def matmult(self, shape_X, shape_Y):
        assert(shape_X[1] == shape_Y[0])
        self.mac_count += shape_X[0] * shape_X[1] * shape_Y[1]

    # TODO make this work
    def conv(self, shape_X, shape_Y):
        self.mac_count += shape_X[0] * shape_X[1] * shape_Y[1]

    # TODO this should be add(x, y) where x and y are same length and we pick the max along each index...
    def add(self, shape_X):
        self.add_count += np.prod(shape_X) 

###################################################################           

class RRAM(Compute):

    def __init__(self):
        self.mac_count = 0
        self.add_count = 0

    def matmult(self, W, X):
        pass

    def conv(self, W, X):
        pass

    def add(self, X):
        pass
        
###################################################################           

