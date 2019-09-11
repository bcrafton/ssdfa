
import tensorflow as tf
import numpy as np
np.set_printoptions(threshold=1000)

class Block:
    def __init__(self, layers : tuple):
        self.num_layers = len(layers)
        self.layers = layers
        
    def num_params(self):
        param_sum = 0
        for ii in range(self.num_layers):
            l = self.layers[ii]
            param_sum += l.num_params()
        return param_sum

    def get_weights(self):
        weights = {}
        for ii in range(self.num_layers):
            l = self.layers[ii]
            tup = l.get_weights()
            for (key, value) in tup:
                weights[key] = value
            
        return weights

    ####################################################################

    def forward(self, X):
        A = [None] * self.num_layers
        cache = {}
        
        for ii in range(self.num_layers):
            l = self.layers[ii]
            if ii == 0:
                A[ii] = l.forward(X)
            else:
                A[ii] = l.forward(A[ii-1]['aout'])

        return A[self.num_layers-1]['aout'], A
      
    def bp(self, AI, AO, DO, cache):
        A = [None] * self.num_layers
        D = [None] * self.num_layers
        grads_and_vars = []
            
        for ii in range(self.num_layers-1, -1, -1):
            l = self.layers[ii]
            
            if (ii == self.num_layers-1):
                D[ii], gvs = l.bp(cache[ii-1]['aout'], cache[ii]['aout'], E,       cache[ii]['cache'])
                grads_and_vars.extend(gvs)
            elif (ii == 0):
                D[ii], gvs = l.bp(AI,                  cache[ii]['aout'], D[ii+1], cache[ii]['cache'])
                grads_and_vars.extend(gvs)
            else:
                D[ii], gvs = l.bp(cache[ii-1]['aout'], cache[ii]['aout'], D[ii+1], cache[ii]['cache'])
                grads_and_vars.extend(gvs)

        return grads_and_vars, D[-1]

    def ss(self, AI, AO, DO, cache):
        return self.bp(AI, AO, DO, cache)

    def dfa(self, AI, AO, E, DO, cache):
        return self.bp(AI, AO, DO, cache)

    def lel(self, AI, AO, DO, Y, cache):
        return self.bp(AI, AO, DO, cache)

    ####################################################################
    


        
        
        
        
        
        
        
        
