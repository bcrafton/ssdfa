
import tensorflow as tf
import numpy as np

###################################################################

# TODO account for sparsity
# DONT USE THE SAME NAME FOR THE CLASS METHOD AS THE COUNTER
# >>> send(), send.

class Memory:

    def __init__(self):
        super().__init__()

    def read(self, shape_X, sparsity_X=0.):
        pass

    def write(self, shape_X, sparsity_X=0.):
        pass

    def total(self):
        pass

###################################################################           

class DRAM(Memory):

    def __init__(self):
        self.read_count = 0
        self.write_count = 0

    def read(self, shape_X, rate_X=1.):
        self.read_count += np.prod(shape_X) * rate_X

    def write(self, shape_X, rate_X=1.):
        self.write_count += np.prod(shape_X) * rate_X
        
    def total(self):
        return {'read': self.read_count, 'write': self.write_count}
        
###################################################################           

class RRAM(Memory):

    def __init__(self):
        self.read_count = 0
        self.write_count = 0
        self.mac_count = 0

    def read(self, shape_X, rate_X=1.):
        self.read_count += np.prod(shape_X) * rate_X

    def write(self, shape_X, rate_X=1.):
        self.write_count += np.prod(shape_X) * rate_X

    def matmult(self, shape_X, shape_Y, rate_X=1., rate_Y=1.):
        assert(shape_X[1] == shape_Y[0])
        self.mac_count += shape_X[0] * shape_X[1] * shape_Y[1] * rate_Y * rate_X

    def conv(self, shape_filters, shape_images, padding, strides):
        assert(False)
        fh, fw, fin, fout = shape_filters
        batch_size, h, w, fin = shape_images
        
        shape_filter = (fh, fw, fin)
    
        output_row = conv_output_length(h, fh, padding.lower(), strides[1])
        output_col = conv_output_length(w, fw, padding.lower(), strides[2])
        shape_output = (batch_size, output_row, output_col, fout)
    
        self.mac_count += np.prod(shape_output) * np.prod(shape_filter)

    def total(self):
        return {'read': self.read_count, 'write': self.write_count, 'mac': self.mac_count}
        
###################################################################

