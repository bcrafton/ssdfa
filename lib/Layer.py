
import tensorflow as tf
import numpy as np

class Layer:

    def __init__(self):
        super().__init__()

    ###################################################################

    def get_weights(self):
        assert(False)
        
    def num_params(self):
        assert(False)

    def forward(self, X):
        assert(False)

    ###################################################################    
        
    def bp(self, AI, AO, DO, cache):    
        assert(False)

    def dfa(self, AI, AO, E, DO, cache):
        assert(False)

    def lel(self, AI, AO, DO, Y, cache):
        assert(False)
        
    ###################################################################   
