
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

np.set_printoptions(threshold=np.inf)

class LELFC(Layer):

    def __init__(self, input_shape, num_classes, name=None):
        self.num_classes = num_classes
        self.input_shape = input_shape
        self.name = name

        '''
        if load:
            weight_dict = np.load(load).item()
            self.B = tf.cast(tf.Variable(weight_dict[self.name]), tf.float32)
        elif std is not None:
            b = np.random.normal(loc=0., scale=std, size=(self.num_classes, self.output_size))
            self.B = tf.cast(tf.Variable(b), tf.float32)
        else:
            # var = 1. / self.output_size
            # std = np.sqrt(var)
            # b = np.random.normal(loc=0., scale=std, size=(self.num_classes, self.output_size))

            b = FeedbackMatrix(size=(self.num_classes, self.output_size), sparse=self.sparse, rank=self.rank)
            self.B = tf.cast(tf.Variable(b), tf.float32) 
        '''

        # THE PROBLEM WAS NEVER THE BIAS ... IT WAS THE FACT WE WERNT DIVIDING BY N
        
        # l0 = FullyConnected(input_shape=input_shape, size=self.input_shape, init='alexnet', activation=Relu(), bias=1., name=self.name)
        self.l0 = FullyConnected(input_shape=input_shape, size=self.num_classes, init='alexnet', activation=Linear(), bias=0., name=self.name)        

        # self.B = Model(layers=[l1])

    def get_weights(self):
        # return self.l0.get_weights()
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
        S = tf.matmul(AO, tf.transpose(self.B))
        # should be doing cross entropy here.
        # is this right ?
        # just adding softmax ?
        ES = tf.subtract(tf.nn.softmax(S), Y)
        DO = tf.matmul(ES, self.B)
        # (* activation.gradient) and (* AI) occur in the actual layer itself.
        return DO
        '''
        # '''
        S = self.l0.forward(AI)
        ES = tf.subtract(tf.nn.softmax(S), Y)
        DI = self.l0.backward(AI, S, ES)
        # '''

        # DI = self.B.backwards(AI, Y)

        return DI

    def lel_gv(self, AI, AO, E, DO, Y):
        # '''
        S = self.l0.forward(AI)
        ES = tf.subtract(tf.nn.softmax(S), Y)
        gvs = self.l0.gv(AI, S, ES)
        # '''

        # gvs = self.B.gvs(AI, Y)

        return gvs
        
    def lel(self, AI, AO, E, DO, Y): 
        assert(False)
        
    ###################################################################  
        
        
        
