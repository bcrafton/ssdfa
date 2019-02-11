
import tensorflow as tf
import numpy as np

###################################################################

# TODO account for sparsity
# DONT USE THE SAME NAME FOR THE CLASS METHOD AS THE COUNTER
# >>> send(), send.

class Movement:

    def __init__(self):
        super().__init__()

    def send(self, shape_X):
        pass

    def receive(self, shape_X):
        pass
        
    def total(self):
        pass

###################################################################           

# we should also be receiving W here. 

class vonNeumann(Movement):

    def __init__(self):
        self.send_count = 0
        self.receive_count = 0

    def send(self, shape_X):
        self.send_count += np.prod(shape_X)

    def receive(self, shape_X):
        self.receive_count += np.prod(shape_X)

    def total(self):
        return {'send': self.send_count, 'receive': self.receive_count}

###################################################################           

class Neuromorphic(Movement):

    def __init__(self):
        self.send_count = 0
        self.receive_count = 0

    def send(self, shape_X):
        self.send_count += np.prod(shape_X)

    def receive(self, shape_X):
        self.receive_count += np.prod(shape_X)
        
    def total(self):
        return {'send': self.send_count, 'receive': self.receive_count}
        
###################################################################           





