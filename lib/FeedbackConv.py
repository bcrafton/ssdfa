
import tensorflow as tf
import numpy as np
import math

from lib.Layer import Layer 
from lib.Activation import Activation
from lib.Activation import Sigmoid
from lib.FeedbackMatrix import FeedbackMatrix

class FeedbackConv(Layer):

    def __init__(self, size : tuple, num_classes : int, sparse : int, rank : int, name=None, load=None):
        self.size = size
        self.num_classes = num_classes
        self.sparse = sparse
        self.rank = rank
        self.batch_size, self.h, self.w, self.f = self.size
        self.name = name
        self.num_output = self.h * self.w * self.f

        if load:
            weight_dict = np.load(load).item()
            b = weight_dict[self.name]
            self.B = tf.Variable(b, dtype=tf.float32)
        else:
            b, mask = FeedbackMatrix(size=(self.num_classes, self.num_output), sparse=self.sparse, rank=self.rank)

            print (np.count_nonzero(b), np.prod(np.shape(b)))

            self.B = tf.Variable(b, dtype=tf.float32) 
            self.Mask = tf.Variable(mask, dtype=tf.float32)


    ###################################################################
    
    def get_weights(self):
        return [(self.name, self.B)]
    
    def get_feedback(self):
        return self.B

    def num_params(self):
        return 0
        
    def forward(self, X):
        return X
                
    ###################################################################           
        
    def backward(self, AI, AO, DO):    
        return DO

    def gv(self, AI, AO, DO):    
        return []
        
    def train(self, AI, AO, DO): 
        return []
        
    ###################################################################

    def dfa_backward(self, AI, AO, E, DO):
        E = tf.matmul(E, self.B)
        E = tf.reshape(E, self.size)
        E = tf.multiply(E, DO)
        return E
        
    def dfa_gv(self, AI, AO, E, DO):
        AI = tf.reshape(AI, (-1, self.num_output))
        DW = tf.matmul(tf.transpose(E), AI) * self.Mask
        # DW = tf.reshape(AI, self.size)
        return [(DW, self.B)]

    def dfa(self, AI, AO, E, DO): 
        return []
        
    ###################################################################   
        
    # > https://ml-cheatsheet.readthedocs.io/en/latest/loss_functions.html
    # > https://www.ics.uci.edu/~pjsadows/notes.pdf
    # > https://deepnotes.io/softmax-crossentropy
    def lel_backward(self, AI, AO, E, DO, Y):
        shape = tf.shape(AO)
        N = shape[0]
        AO = tf.reshape(AO, (N, self.num_output))
        S = tf.matmul(AO, tf.transpose(self.B))
        # should be doing cross entropy here.
        # is this right ?
        # just adding softmax ?
        ES = tf.subtract(tf.nn.softmax(S), Y)
        DO = tf.matmul(ES, self.B)
        DO = tf.reshape(DO, self.size)
        # (* activation.gradient) and (* AI) occur in the actual layer itself.
        return DO
        
    def lel_gv(self, AI, AO, E, DO, Y):
        return []
        
    def lel(self, AI, AO, E, DO, Y): 
        return []
        
    ###################################################################
        
        


