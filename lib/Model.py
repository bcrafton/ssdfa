
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
        
    ####################################################################
        
    def train(self, X, Y):
        A = [None] * self.num_layers
        D = [None] * self.num_layers
        grads_and_vars = []
        
        for ii in range(self.num_layers):
            l = self.layers[ii]
            if ii == 0:
                A[ii] = l.forward(X)
            else:
                A[ii] = l.forward(A[ii-1])

        E = tf.nn.softmax(A[self.num_layers-1]) - Y
        N = tf.shape(A[self.num_layers-1])[0]
        N = tf.cast(N, dtype=tf.float32)
        E = E / N

        for ii in range(self.num_layers-1, -1, -1):
            l = self.layers[ii]
            
            if (ii == self.num_layers-1):
                D[ii] = l.backward(A[ii-1], A[ii], E)
                gvs = l.train(A[ii-1], A[ii], E)
                grads_and_vars.extend(gvs)
            elif (ii == 0):
                D[ii] = l.backward(X, A[ii], D[ii+1])
                gvs = l.train(X, A[ii], D[ii+1])
                grads_and_vars.extend(gvs)
            else:
                D[ii] = l.backward(A[ii-1], A[ii], D[ii+1])
                gvs = l.train(A[ii-1], A[ii], D[ii+1])
                grads_and_vars.extend(gvs)

        return grads_and_vars
              
    def dfa(self, X, Y):
        A = [None] * self.num_layers
        D = [None] * self.num_layers
        grads_and_vars = []
        
        for ii in range(self.num_layers):
            l = self.layers[ii]
            if ii == 0:
                A[ii] = l.forward(X)
            else:
                A[ii] = l.forward(A[ii-1])

        E = tf.nn.softmax(A[self.num_layers-1]) - Y
        N = tf.shape(A[self.num_layers-1])[0]
        N = tf.cast(N, dtype=tf.float32)
        E = E / N

        for ii in range(self.num_layers-1, -1, -1):
            l = self.layers[ii]
                
            if (ii == self.num_layers-1):
                D[ii] = l.dfa_backward(A[ii-1], A[ii], E, E)
                gvs = l.dfa(A[ii-1], A[ii], E, E)
                grads_and_vars.extend(gvs)
            elif (ii == 0):
                D[ii] = l.dfa_backward(X, A[ii], E, D[ii+1])
                gvs = l.dfa(X, A[ii], E, D[ii+1])
                grads_and_vars.extend(gvs)
            else:
                D[ii] = l.dfa_backward(A[ii-1], A[ii], E, D[ii+1])
                gvs = l.dfa(A[ii-1], A[ii], E, D[ii+1])
                grads_and_vars.extend(gvs)
                
        return grads_and_vars
    
    def lel(self, X, Y):
        A = [None] * self.num_layers
        D = [None] * self.num_layers
        grads_and_vars = []
        
        for ii in range(self.num_layers):
            l = self.layers[ii]
            if ii == 0:
                A[ii] = l.forward(X)
            else:
                A[ii] = l.forward(A[ii-1])

        E = tf.nn.softmax(A[self.num_layers-1]) - Y
        N = tf.shape(A[self.num_layers-1])[0]
        N = tf.cast(N, dtype=tf.float32)
        E = E / N

        for ii in range(self.num_layers-1, -1, -1):
            l = self.layers[ii]
                
            if (ii == self.num_layers-1):
                D[ii] = l.lel_backward(A[ii-1], A[ii], E, E, Y)
                gvs = l.lel(A[ii-1], A[ii], E, E, Y)
                grads_and_vars.extend(gvs)
            elif (ii == 0):
                D[ii] = l.lel_backward(X, A[ii], E, D[ii+1], Y)
                gvs = l.lel(X, A[ii], E, D[ii+1], Y)
                grads_and_vars.extend(gvs)
            else:
                D[ii] = l.lel_backward(A[ii-1], A[ii], E, D[ii+1], Y)
                gvs = l.lel(A[ii-1], A[ii], E, D[ii+1], Y)
                grads_and_vars.extend(gvs)
                
        return grads_and_vars
    
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
                D[ii] = l.dfa_backward(A[ii-1]['aout'], A[ii]['aout'], E, E,               A[ii]['cache'])
                gvs =         l.dfa_gv(A[ii-1]['aout'], A[ii]['aout'], E, E,               D[ii]['cache'])
                grads_and_vars.extend(gvs)
            elif (ii == 0):
                D[ii] = l.dfa_backward(X,               A[ii]['aout'], E, D[ii+1]['dout'], A[ii]['cache'])
                gvs =         l.dfa_gv(X,               A[ii]['aout'], E, D[ii+1]['dout'], D[ii]['cache'])
                grads_and_vars.extend(gvs)
            else:
                D[ii] = l.dfa_backward(A[ii-1]['aout'], A[ii]['aout'], E, D[ii+1]['dout'], A[ii]['cache'])
                gvs =         l.dfa_gv(A[ii-1]['aout'], A[ii]['aout'], E, D[ii+1]['dout'], D[ii]['cache'])
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

        E = tf.nn.softmax(A[self.num_layers-1]['aout']) - Y[-1]
        N = tf.shape(A[self.num_layers-1]['aout'])[0]
        N = tf.cast(N, dtype=tf.float32)
        E = E / N
            
        for ii in range(self.num_layers-1, -1, -1):
            l = self.layers[ii]

            if (ii == self.num_layers-1):
                D[ii] = l.lel_backward(A[ii-1]['aout'], A[ii]['aout'], E, E,               Y, A[ii]['cache'])
                gvs =         l.lel_gv(A[ii-1]['aout'], A[ii]['aout'], E, E,               Y, D[ii]['cache'])
                grads_and_vars.extend(gvs)
            elif (ii == 0):
                D[ii] = l.lel_backward(X,               A[ii]['aout'], E, D[ii+1]['dout'], Y, A[ii]['cache'])
                gvs =         l.lel_gv(X,               A[ii]['aout'], E, D[ii+1]['dout'], Y, D[ii]['cache'])
                grads_and_vars.extend(gvs)
            else:
                D[ii] = l.lel_backward(A[ii-1]['aout'], A[ii]['aout'], E, D[ii+1]['dout'], Y, A[ii]['cache'])
                gvs =         l.lel_gv(A[ii-1]['aout'], A[ii]['aout'], E, D[ii+1]['dout'], Y, D[ii]['cache'])
                grads_and_vars.extend(gvs)
                
        return grads_and_vars

    ####################################################################
    
    def backwards(self, X, Y):
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
                D[ii] = l.backward(A[ii-1]['aout'], A[ii]['aout'], E, A[ii]['cache'])
            elif (ii == 0):
                D[ii] = l.backward(X, A[ii]['aout'], D[ii+1]['dout'], A[ii]['cache'])
            else:
                D[ii] = l.backward(A[ii-1]['aout'], A[ii]['aout'], D[ii+1]['dout'], A[ii]['cache'])
                
        return D[0]['dout']
    
    def dfa_backwards(self, X, Y):
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
                D[ii] = l.dfa_backward(A[ii-1]['aout'], A[ii]['aout'], E, A[ii]['cache'])
            elif (ii == 0):
                D[ii] = l.dfa_backward(X, A[ii]['aout'], D[ii+1]['dout'], A[ii]['cache'])
            else:
                D[ii] = l.dfa_backward(A[ii-1]['aout'], A[ii]['aout'], D[ii+1]['dout'], A[ii]['cache'])
                
        return D[0]['dout']
    
    ####################################################################
    
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
                A[ii] = l.forward(A[ii-1]['aout'])
                
        return A[N]['aout']
        
        
        
        
        
        
        
        
