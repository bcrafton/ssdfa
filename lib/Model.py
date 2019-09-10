
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
            l = self.layers[ii]
            tup = l.get_weights()
            for (key, value) in tup:
                weights[key] = value
            
        return weights

    def predict(self, X):
        A = [None] * self.num_layers
        
        for ii in range(self.num_layers):
            l = self.layers[ii]
            if ii == 0:
                A[ii] = l.forward(X)
            else:
                A[ii] = l.forward(A[ii-1]['aout'])
                
        return A[self.num_layers-1]['aout']
    
    ####################################################################
      
    def gvs(self, X, Y):
        A = [None] * self.num_layers
        D = [None] * self.num_layers
        grads_and_vars = []
        
        for ii in range(self.num_layers):
            l = self.layers[ii]
            if ii == 0:
                A[ii] = l.forward(X)
            else:
                A[ii] = l.forward(A[ii-1]['aout'])

        E = tf.nn.softmax(A[self.num_layers-1]['aout']) - Y
        N = tf.shape(A[self.num_layers-1]['aout'])[0]
        N = tf.cast(N, dtype=tf.float32)
        E = E / N
            
        for ii in range(self.num_layers-1, -1, -1):
            l = self.layers[ii]
            
            if (ii == self.num_layers-1):
                D[ii], gvs = l.bp(A[ii-1]['aout'], A[ii]['aout'], E,               A[ii]['cache'])
                grads_and_vars.extend(gvs)
            elif (ii == 0):
                D[ii], gvs = l.bp(X,               A[ii]['aout'], D[ii+1]['dout'], A[ii]['cache'])
                grads_and_vars.extend(gvs)
            else:
                D[ii], gvs = l.bp(A[ii-1]['aout'], A[ii]['aout'], D[ii+1]['dout'], A[ii]['cache'])
                grads_and_vars.extend(gvs)

        return grads_and_vars
    
    def dfa_gvs(self, X, Y):
        A = [None] * self.num_layers
        D = [None] * self.num_layers
        grads_and_vars = []
        
        for ii in range(self.num_layers):
            l = self.layers[ii]
            if ii == 0:
                A[ii] = l.forward(X)
            else:
                A[ii] = l.forward(A[ii-1]['aout'])

        E = tf.nn.softmax(A[self.num_layers-1]['aout']) - Y
        N = tf.shape(A[self.num_layers-1]['aout'])[0]
        N = tf.cast(N, dtype=tf.float32)
        E = E / N
            
        for ii in range(self.num_layers-1, -1, -1):
            l = self.layers[ii]

            if (ii == self.num_layers-1):
                D[ii], gvs = l.dfa(A[ii-1]['aout'], A[ii]['aout'], E, E,               A[ii]['cache'])
                grads_and_vars.extend(gvs)
            elif (ii == 0):
                D[ii], gvs = l.dfa(X,               A[ii]['aout'], E, D[ii+1]['dout'], A[ii]['cache'])
                grads_and_vars.extend(gvs)
            else:
                D[ii], gvs = l.dfa(A[ii-1]['aout'], A[ii]['aout'], E, D[ii+1]['dout'], A[ii]['cache'])
                grads_and_vars.extend(gvs)
                
        return grads_and_vars

    def lel_gvs(self, X, Y):
        A = [None] * self.num_layers
        D = [None] * self.num_layers
        grads_and_vars = []
        
        for ii in range(self.num_layers):
            l = self.layers[ii]
            if ii == 0:
                A[ii] = l.forward(X)
            else:
                A[ii] = l.forward(A[ii-1]['aout'])

        E = tf.nn.softmax(A[self.num_layers-1]['aout']) - Y
        N = tf.shape(A[self.num_layers-1]['aout'])[0]
        N = tf.cast(N, dtype=tf.float32)
        E = E / N
            
        for ii in range(self.num_layers-1, -1, -1):
            l = self.layers[ii]

            if (ii == self.num_layers-1):
                D[ii], gvs = l.lel(A[ii-1]['aout'], A[ii]['aout'], E, E,               Y, A[ii]['cache'])
                grads_and_vars.extend(gvs)
            elif (ii == 0):
                D[ii], gvs = l.lel(X,               A[ii]['aout'], E, D[ii+1]['dout'], Y, A[ii]['cache'])
                grads_and_vars.extend(gvs)
            else:
                D[ii], gvs = l.lel(A[ii-1]['aout'], A[ii]['aout'], E, D[ii+1]['dout'], Y, A[ii]['cache'])
                grads_and_vars.extend(gvs)
                
        return grads_and_vars

    ####################################################################
    


        
        
        
        
        
        
        
        
