
import tensorflow as tf
import numpy as np
import math

from lib.Layer import Layer 
from lib.Activation import Activation
from lib.Activation import Sigmoid
from lib.FeedbackMatrix import FeedbackMatrix

from lib.Memory import Memory
from lib.Memory import DRAM
from lib.Memory import RRAM

from lib.Compute import Compute
from lib.Compute import CMOS

from lib.Movement import Movement
from lib.Movement import vonNeumann
from lib.Movement import Neuromorphic

from lib.add_dict import add_dict

np.set_printoptions(threshold=np.inf)

class FeedbackFC(Layer):
    num = 0
    def __init__(self, size : tuple, num_classes : int, sparse : int, rank : int, name=None, load=None, std=None):
        self.size = size
        self.num_classes = num_classes
        self.sparse = sparse
        self.rank = rank
        self.input_size, self.output_size = self.size
        self.name = name

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
        E = tf.multiply(E, DO)

        # mean, var = tf.nn.moments(E, axes=[0, 1])
        # E = tf.Print(E, [var], message="std: ")

        return E
        
    def dfa_gv(self, AI, AO, E, DO):
        return []
        
    def dfa(self, AI, AO, E, DO): 
        return []
        
    ###################################################################  
        
    # > https://ml-cheatsheet.readthedocs.io/en/latest/loss_functions.html
    # > https://www.ics.uci.edu/~pjsadows/notes.pdf
    # > https://deepnotes.io/softmax-crossentropy
    def lel_backward(self, AI, AO, E, DO, Y):
        S = tf.matmul(AO, tf.transpose(self.B))
        # should be doing cross entropy here.
        # is this right ?
        # just adding softmax ?
        ES = tf.subtract(tf.nn.softmax(S), Y)
        DO = tf.matmul(ES, self.B)
        # (* activation.gradient) and (* AI) occur in the actual layer itself.
        return DO
        
    def lel_gv(self, AI, AO, E, DO, Y):
        return []
        
    def lel(self, AI, AO, E, DO, Y): 
        return []
        
    ###################################################################  
        
    def metrics(self, dfa=False, memory=None, compute=None, movement=None, examples=1, epochs=1):
        
        if not dfa:
            return {}

        memory = DRAM() if memory is None else memory
        compute = CMOS() if compute is None else compute
        movement = vonNeumann() if movement is None else movement

        total_examples = examples * epochs

        size = (self.num_classes, self.output_size)
        input_size = (total_examples, self.num_classes)
        output_size = (total_examples, self.output_size)

        rate = 1.0 - 1.0 * self.sparse / self.num_classes

        #############################

        if type(memory) in [DRAM]:
            memory.read(size, rate_X=rate)
            compute.matmult(input_size, size, rate_Y=rate)
        elif type(memory) in [RRAM]:
            memory.matmult(input_size, size, rate_Y=rate)
        else:
            assert(False)

        movement.receive(input_size)
        movement.send(output_size)
        
        #############################
    
        total = {}
        total = add_dict(total, memory.total())
        total = add_dict(total, compute.total())
        total = add_dict(total, movement.total())

        #############################
        
        return total
        
