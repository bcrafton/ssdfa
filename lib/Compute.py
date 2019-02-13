
import tensorflow as tf
import numpy as np
from lib.conv_utils import conv_output_length
from lib.conv_utils import conv_input_length

###################################################################

# DONT USE THE SAME NAME FOR THE CLASS METHOD AS THE COUNTER
# >>> send(), send.

class Compute:

    def __init__(self):
        super().__init__()

    def matmult(self, shape_X, shape_Y):
        pass

    def conv(self, shape_filters, shape_images, padding, strides):
        pass

    def add(self, X):
        pass
        
    def total(self):
        pass

###################################################################           

class CMOS(Compute):

    def __init__(self):
        self.mac_count = 0
        self.add_count = 0

    def matmult(self, shape_X, shape_Y, rate_Y=1., rate_X=1.):
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

    # TODO this should be add(x, y) where x and y are same length and we pick the max along each index...
    def add(self, shape_X):
        self.add_count += np.prod(shape_X) 
        
    def total(self):
        return {'mac': self.mac_count, 'add': self.add_count}

###################################################################           

