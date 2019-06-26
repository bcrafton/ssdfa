
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
from lib.Dropout import Dropout

# from lib.AvgPoolZ import AvgPool
from lib.AvgPool import AvgPool

class LELPool(Layer):

    def __init__(self, input_shape, pool_shape, num_classes, ae_ouput_shape, ae_filter_shape, name=None):
        self.input_shape = input_shape
        self.batch_size, self.h, self.w, self.fin = self.input_shape
        self.pool_shape = pool_shape
        self.num_classes = num_classes
        self.ae_ouput_shape = ae_ouput_shape
        self.ae_filter_shape = ae_filter_shape
        self.name = name

        self.bias = tf.Variable(np.zeros(shape=self.ae_ouput_shape), dtype=tf.float32)

        ###################################################################

        self.pool = AvgPool(size=self.input_shape, ksize=self.pool_shape, strides=self.pool_shape, padding='SAME')

        conv2fc_shape = self.pool.output_shape()
        self.conv2_fc = ConvToFullyConnected(input_shape=conv2fc_shape)
        
        fc_shape = self.conv2_fc.output_shape()
        self.fc = FullyConnected(input_shape=fc_shape, size=self.num_classes, init='alexnet', activation=Linear(), bias=0., name=self.name)

        ###################################################################
        
        # hold up, are we just doing target propagation here ?
        self.decode_conv = ConvBlock(input_shape=self.input_shape, 
                                     filter_shape=self.ae_filter_shape, 
                                     strides=[1,1,1,1], 
                                     init='alexnet', 
                                     name=self.name + '_decoder')
        
        # pretty sure there is no need for upsample ... so what are we doing here ...
        # self.decoder_up = UpSample(input_shape=input_shape, ksize=self.ksize)
        
    ###################################################################
    
    def get_weights(self):
        return []

    def get_feedback(self):
        assert(False)
        
    def output_shape(self):
        return self.input_shape

    def num_params(self):
        return 0
        
    def forward(self, X):
        return {'aout':X, 'cache':{}}
                
    ###################################################################           
        
    def backward(self, AI, AO, DO, cache=None):    
        return {'dout':DO, 'cache':{}}

    def gv(self, AI, AO, DO, cache=None):    
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
        
    def lel_backward(self, AI, AO, E, DO, Y, cache):
    
        ############
    
        pool = self.pool.forward(AI)
        conv2fc = self.conv2fc.forward(pool['aout'])
        fc = self.fc.forward(conv2fc['aout'])

        dfc = self.fc.backward(conv2fc['aout'], fc['aout'], DO)
        dconv2fc = self.conv2fc.backward(pool['aout'], conv2fc['aout'], dfc['dout'])
        dpool = self.pool.backward(AI, pool['aout'], dconv2fc['dout'])

        cache = {'pool':pool['aout'], 'conv2fc':conv2fc['aout'], 'fc':fc['aout']}
        cache.update({'dpool':dpool['dout'], 'dconv2fc':dconv2fc['dout'], 'dfc':dfc['dout']})
        
        ############
        
        AE_X = cache['AE_X']
        
        decode_conv = self.decode_conv.forward(AI)
        
        pred = decode_conv + self.bias
        loss = tf.losses.mean_squared_error(labels=AE_X, predictions=pred)
        grads = tf.gradients(loss, [self.bias])
        grad = grads[0]
        
        ddecode_conv = self.decode_conv.backward(AI, decode_conv, grad)

        cache.update({'decode_conv':decode_conv['aout']})
        cache.update({'ddecode_conv':ddecode_conv['dout']})

        ############

        DI = dpool['dout'] + 0.05 * ddecode_conv['dout']
        return {'dout':DI, 'cache':cache}
        
    def lel_gv(self, AI, AO, E, DO, Y, cache):
    
        ############
    
        pool, conv2fc, fc = cache['pool'], cache['conv2fc'], cache['fc']
        dfc, dconv2fc, dpool = cache['dfc'], cache['dconv2fc'], cache['dpool']
        
        dfc = self.fc.gv(conv2fc, fc, dfc)
        
        ############

        decode_conv = cache['decode_conv']
        ddecode_conv = cache['ddecode_conv']

        ddecode_conv = self.decode_conv.gv(AI, decode_conv, ddecode_conv)

        ############

        grads = []
        grads.extend(dfc)
        grads.extend(ddecode_conv)
        return grads

    def lel(self, AI, AO, E, DO, Y): 
        return []
        
    ###################################################################
        
        
        
        
        
        
        
        
        
        
        
        


