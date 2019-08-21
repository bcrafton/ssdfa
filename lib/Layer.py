
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

    def ss(self, AI, AO, DO, cache):    
        return self.bp(AI, AO, DO, cache)

    def dfa(self, AI, AO, E, DO, cache):
        return self.bp(AI, AO, DO, cache)
        
    def lel(self, AI, AO, DO, Y, cache):
        return self.bp(AI, AO, DO, cache)
        
    ###################################################################   
