
import tensorflow as tf
import numpy as np
np.set_printoptions(threshold=1000)

class Model:
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
            tups = self.layers[ii].get_weights()
            for (k, v) in tups:
                weights[k] = v

        return weights

    def predict(self, X):
        A = [None] * self.num_layers
        cache = [None] * self.num_layers
        
        for ii in range(self.num_layers):
            l = self.layers[ii]
            if ii == 0:
                A[ii], cache[ii] = l.forward(X)
            else:
                A[ii], cache[ii] = l.forward(A[ii-1])
                
        return A[self.num_layers-1]
        
    def predict1(self, X):
        A = [None] * self.num_layers
        cache = [None] * self.num_layers
        
        for ii in range(self.num_layers):
            l = self.layers[ii]
            if ii == 0:
                A[ii], cache[ii] = l.forward1(X)
            else:
                A[ii], cache[ii] = l.forward1(A[ii-1])
                
        return A, cache
    
    def predict2(self, X):
        A = [None] * self.num_layers
        cache = [None] * self.num_layers
        
        for ii in range(self.num_layers):
            l = self.layers[ii]
            if ii == 0:
                A[ii], cache[ii] = l.forward2(X)
            else:
                A[ii], cache[ii] = l.forward2(A[ii-1])
                
        return A[self.num_layers-1], A
    
    ####################################################################
      
    def gvs(self, X, Y):
        A = [None] * self.num_layers
        cache = [None] * self.num_layers

        D = [None] * self.num_layers
        grads_and_vars = []
        
        for ii in range(self.num_layers):
            l = self.layers[ii]
            if ii == 0:
                A[ii], cache[ii] = l.forward(X)
            else:
                A[ii], cache[ii] = l.forward(A[ii-1])

        E = tf.nn.softmax(A[self.num_layers-1]) - Y
        # N = tf.shape(A[self.num_layers-1])[0]
        # N = tf.cast(N, dtype=tf.float32)
        # E = E / N
            
        for ii in range(self.num_layers-1, -1, -1):
            l = self.layers[ii]
            
            if (ii == self.num_layers-1):
                D[ii], gvs = l.bp(A[ii-1], A[ii], E,       cache[ii])
                grads_and_vars = gvs + grads_and_vars
            elif (ii == 0):
                D[ii], gvs = l.bp(X,       A[ii], D[ii+1], cache[ii])
                grads_and_vars = gvs + grads_and_vars
            else:
                D[ii], gvs = l.bp(A[ii-1], A[ii], D[ii+1], cache[ii])
                grads_and_vars = gvs + grads_and_vars

        return grads_and_vars
    
    def dfa_gvs(self, X, Y):
        assert(False)

    def lel_gvs(self, X, Y):
        assert(False)

    ####################################################################
    


        
        
        
        
        
        
        
        
