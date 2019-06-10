
import tensorflow as tf
import numpy as np
np.set_printoptions(threshold=1000)

class Model:
    def __init__(self, layers, shape_y):
        self.num_layers = len(layers)
        self.layers = layers
        self.shape_y = shape_y
        self.bias = tf.Variable(np.zeros(shape=self.shape_y), dtype=tf.float32)
        
    def num_params(self):
        param_sum = 0
        for ii in range(self.num_layers):
            l = self.layers[ii]
            param_sum += l.num_params()
        return param_sum
        
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

        '''
        N = tf.shape(A[self.num_layers-1]['aout'])[0]
        N = tf.cast(N, dtype=tf.float32)
        E = (A[self.num_layers-1]['aout'] - Y) / N
        loss = tf.reduce_sum(tf.pow(E, 2)) 
        '''
        pred = A[self.num_layers-1]['aout'] + self.bias
        loss = tf.losses.mean_squared_error(labels=X, predictions=pred)
        grads = tf.gradients(loss, [self.bias])
        E = grads[0]
         
        for ii in range(self.num_layers-1, -1, -1):
            l = self.layers[ii]

            if (ii == self.num_layers-1):
                D[ii] = l.backward(A[ii-1]['aout'], A[ii]['aout'], E,               A[ii]['cache'])
                gvs =         l.gv(A[ii-1]['aout'], A[ii]['aout'], E,               D[ii]['cache'])
                grads_and_vars.extend(gvs)
            elif (ii == 0):
                D[ii] = l.backward(X,               A[ii]['aout'], D[ii+1]['dout'], A[ii]['cache'])
                gvs =         l.gv(X,               A[ii]['aout'], D[ii+1]['dout'], D[ii]['cache'])
                grads_and_vars.extend(gvs)
            else:
                D[ii] = l.backward(A[ii-1]['aout'], A[ii]['aout'], D[ii+1]['dout'], A[ii]['cache'])
                gvs =         l.gv(A[ii-1]['aout'], A[ii]['aout'], D[ii+1]['dout'], D[ii]['cache'])
                grads_and_vars.extend(gvs)
                
        return grads_and_vars, loss
    
    ####################################################################
    
    def backwards(self, X, Y):
        assert(False)
    
    ####################################################################
    
    def predict(self, X):
        A = [None] * self.num_layers
        
        for ii in range(self.num_layers):
            l = self.layers[ii]
            if ii == 0:
                A[ii] = l.forward(X)
            else:
                A[ii] = l.forward(A[ii-1])
                
        return A[self.num_layers-1]
        
    ####################################################################
    
    def get_weights(self):
        weights = {}
        for ii in range(self.num_layers):
            l = self.layers[ii]
            tup = l.get_weights()
            for (key, value) in tup:
                weights[key] = value
            
        return weights
        
    def up_to(self, X, N):
        A = [None] * (N + 1)
        
        for ii in range(N + 1):
            l = self.layers[ii]
            if ii == 0:
                A[ii] = l.forward(X)
            else:
                A[ii] = l.forward(A[ii-1])
                
        return A[N]
        
        
        
        
        
        
        
        
