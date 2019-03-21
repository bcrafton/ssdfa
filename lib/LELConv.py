
import tensorflow as tf
import numpy as np
import math

from lib.Layer import Layer 
from lib.Activation import Activation
from lib.Activation import Sigmoid
from lib.FeedbackMatrix import FeedbackMatrix

from lib.Model import Model
from lib.Layer import Layer 
from lib.ConvToFullyConnected import ConvToFullyConnected
from lib.FullyConnected import FullyConnected
from lib.Convolution import Convolution
from lib.MaxPool import MaxPool
from lib.Activation import Relu
from lib.Activation import Linear

class LELConv(Layer):

    def __init__(self, batch_size, input_shape, filter_sizes, num_classes, name=None):
        self.input_shape = input_shape
        self.filter_sizes = filter_sizes
        self.fh, self.fw, self.fin, self.fout = self.filter_sizes
        self.h, self.w, self.fin = self.input_shape
        self.batch_size = batch_size 
        self.num_classes = num_classes
        self.name = name

        '''
        if load:
            weight_dict = np.load(load).item()
            self.B = tf.cast(tf.Variable(weight_dict[self.name]), tf.float32)
        else:
            b = FeedbackMatrix(size=(self.num_classes, self.num_output), sparse=self.sparse, rank=self.rank)
            self.B = tf.cast(tf.Variable(b), tf.float32) 
        '''
        
        l0 = Convolution(batch_size=batch_size, input_shape=self.input_shape, filter_sizes=self.filter_sizes, init='alexnet', strides=[1, 1], padding="SAME", activation=Relu(), bias=0., name=self.name)
        l1 = MaxPool(batch_size=batch_size, input_shape=l0.output_shape(), ksize=[2, 2], strides=[2, 2], padding="SAME")
        l2 = ConvToFullyConnected(input_shape=l1.output_shape())
        # print ("FC SIZE", l2.output_shape())
        l3 = FullyConnected(input_shape=l2.output_shape(), size=self.num_classes, init='alexnet', activation=Linear(), bias=0., name=self.name)
        self.B = Model(layers=[l0, l1, l2, l3])
        
    ###################################################################
    
    def get_weights(self):
        # return self.B.get_weights()
        return []

    def get_feedback(self):
        return self.B
        
    def output_shape(self):
        return self.input_shape

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
        return DO
        
    def dfa_gv(self, AI, AO, E, DO):
        return []
        
    def dfa(self, AI, AO, E, DO): 
        return []
        
    ###################################################################   
        
    # > https://ml-cheatsheet.readthedocs.io/en/latest/loss_functions.html
    # > https://www.ics.uci.edu/~pjsadows/notes.pdf
    # > https://deepnotes.io/softmax-crossentropy
    def lel_backward(self, AI, AO, E, DO, Y):
        '''
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
        '''
        DO = self.B.backwards(AI, Y)
        return DO
        
    def lel_gv(self, AI, AO, E, DO, Y):
        gvs = self.B.gvs(AI, Y)
        return gvs

    def lel(self, AI, AO, E, DO, Y): 
        return []
        
    ###################################################################
        
        


