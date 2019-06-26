
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

from lib.AvgPool import AvgPool

from lib.ConvBlock import ConvBlock

class LELPool(Layer):

    def __init__(self, input_shape, pool_shape, num_classes, ae_output_shape, ae_filter_shape, name=None, ae_loss=0):
        self.input_shape = input_shape
        self.batch_size, self.h, self.w, self.fin = self.input_shape
        self.pool_shape = pool_shape
        self.num_classes = num_classes
        self.ae_output_shape = ae_output_shape
        self.ae_filter_shape = ae_filter_shape
        self.name = name
        self.ae_loss = ae_loss

        self.bias = tf.Variable(np.zeros(shape=self.ae_output_shape), dtype=tf.float32)

        ###################################################################

        self.pool = AvgPool(size=self.input_shape, ksize=self.pool_shape, strides=self.pool_shape, padding='SAME')

        conv2fc_shape = self.pool.output_shape()
        self.conv2fc = ConvToFullyConnected(input_shape=conv2fc_shape)
        
        fc_shape = self.conv2fc.output_shape()
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

        E = tf.nn.softmax(fc['aout']) - Y
        E = E / self.batch_size

        dfc = self.fc.backward(conv2fc['aout'], fc['aout'], E)
        dconv2fc = self.conv2fc.backward(pool['aout'], conv2fc['aout'], dfc['dout'])
        dpool = self.pool.backward(AI, pool['aout'], dconv2fc['dout'])

        ############
        
        AE_X = cache['AE_X']
        
        decode_conv = self.decode_conv.forward(AI)
        
        pred = decode_conv['aout'] + self.bias
        loss = tf.losses.mean_squared_error(labels=AE_X, predictions=pred)
        grads = tf.gradients(loss, [self.bias])
        grad = grads[0]
        
        ddecode_conv = self.decode_conv.backward(AI, decode_conv['aout'], grad, decode_conv['cache'])

        ############

        cache = {'pool':pool['aout'], 'conv2fc':conv2fc['aout'], 'fc':fc['aout']}
        cache.update({'dpool':dpool['dout'], 'dconv2fc':dconv2fc['dout'], 'dfc':dfc['dout'], 'dpred':E})
        cache.update({'decode_conv':decode_conv})
        cache.update({'ddecode_conv':ddecode_conv, 'dae':grad})

        ############

        if self.ae_loss:
            DI = dpool['dout'] + ddecode_conv['dout']
        else:
            DI = dpool['dout']

        # DI = tf.Print(DI, [tf.keras.backend.std(dpool['dout']) / tf.keras.backend.std(ddecode_conv['dout'])], message='', summarize=1000)

        return {'dout':DI, 'cache':cache}
        
    def lel_gv(self, AI, AO, E, DO, Y, cache):
    
        ############
    
        pool, conv2fc, fc = cache['pool'], cache['conv2fc'], cache['fc']
        dpred, dfc, dconv2fc, dpool = cache['dpred'], cache['dfc'], cache['dconv2fc'], cache['dpool']
        
        dfc = self.fc.gv(conv2fc, fc, dpred)
        
        ############

        decode_conv = cache['decode_conv']['aout']
        ddecode_conv = cache['ddecode_conv']['dout']
        dae = cache['dae']

        ddecode_conv = self.decode_conv.gv(AI, decode_conv, dae, cache['ddecode_conv']['cache'])

        ############

        grads = []
        grads.extend(dfc)
        grads.extend(ddecode_conv)
        return grads

    def lel(self, AI, AO, E, DO, Y): 
        return []
        
    ###################################################################
        
        
        
        
        
        
        
        
        
        
        
        


