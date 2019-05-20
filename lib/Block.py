
import tensorflow as tf
import numpy as np

from lib.Model import Model

from lib.Layer import Layer 
from lib.ConvToFullyConnected import ConvToFullyConnected
from lib.FullyConnected import FullyConnected
from lib.Convolution import Convolution
from lib.MaxPool import MaxPool
from lib.AvgPool import AvgPool
from lib.Dropout import Dropout
from lib.LELConv import LELConv
from lib.BatchNorm import BatchNorm
from lib.Activation import Activation
from lib.Activation import Relu

class Block(Layer):

    def __init__(self, input_shape, filter_shape, pool_shape, num_classes, init, name):
        self.input_shape = input_shape
        self.batch, self.h, self.w, self.fin = self.input_shape
        self.filter_shape = filter_shape
        self.fh, self.fw, self.fin, self.fout = self.filter_shape
        self.output_shape = [self.batch, self.h, self.w, self.fout]
        self.pool_shape = pool_shape
        self.num_classes = num_classes
        self.init = init
        self.name = name

        self.lel_shape = [self.batch, self.h, self.w, self.fout + self.fin]
        # print (self.input_shape, self.output_shape, self.lel_shape)

        self.l0 = Convolution(input_sizes=self.input_shape, filter_sizes=self.filter_shape, init=self.init, strides=[1,1,1,1], padding="SAME", name=self.name + '_conv')
        self.l1 = BatchNorm(input_size=self.output_shape, name=self.name + '_bn')
        self.l2 = Relu()
        self.l3 = LELConv(input_shape=self.lel_shape, pool_shape=self.pool_shape, num_classes=self.num_classes, name=self.name + '_fb')
        # self.l3 = LELConv(input_shape=self.output_shape, pool_shape=self.pool_shape, num_classes=self.num_classes, name=self.name + '_fb')

        self.block = Model(layers=[self.l0, self.l1, self.l2, self.l3])

    ###################################################################

    def get_weights(self):
        # shud just always return dictionaries and do ".update()" no lists. 
        # return self.block.get_weights()
        return []

    def output_shape(self):
        assert(False)

    def num_params(self):
        return self.block.num_params()

    def forward(self, X):
        return self.block.forward(X)
        
    ###################################################################           
        
    def backward(self, AI, AO, DO):    
        return self.block.backward(AI, AO, DO)
        
    def gv(self, AI, AO, DO):    
        return self.block.gv(AI, AO, DO)
        
    def train(self, AI, AO, DO): 
        assert(False)
        
    ###################################################################

    def dfa_backward(self, AI, AO, E, DO):
        assert(False)
        
    def dfa_gv(self, AI, AO, E, DO):
        assert(False)
        
    def dfa(self, AI, AO, E, DO): 
        assert(False)
        
    ###################################################################   
    
    def lel_backward(self, AI, AO, E, DO, Y):
        
        # DO = tf.zeros_like(AO)
        # DO = tf.Print(DO, [tf.shape(DO), tf.shape(AO)], message='', summarize=1000)

        conv = self.l0.forward(AI)
        bn   = self.l1.forward(conv)
        relu = self.l2.forward(bn)
        res = tf.concat((relu, AI), axis=3)
        # res = tf.concat((relu, tf.zeros_like(AI)), axis=3)
        # res = relu

        dlel  = self.l3.lel_backward(res, res, None, None, Y)
        drelu = self.l2.lel_backward(bn, relu, None, dlel[:, :, :, 0 : self.fout] + DO, Y)
        # drelu = self.l2.lel_backward(bn, relu, None, dlel, Y)
        dbn   = self.l1.lel_backward(conv, bn, None, drelu, Y)
        dconv = self.l0.lel_backward(AI, conv, None, dbn, Y)

        return dlel[:, :, :, self.fout : (self.fout + self.fin)]
        # return tf.zeros_like(dlel[:, :, :, self.fout : (self.fout + self.fin)])
        # return tf.zeros_like(AI)
        # return tf.zeros_like(dlel)

    def lel_gv(self, AI, AO, E, DO, Y):
        
        # DO = tf.zeros_like(AO)
        # DO = tf.Print(DO, [tf.shape(DO), tf.shape(AO)], message='', summarize=1000)
        # DO = tf.Print(DO, [self.name, tf.keras.backend.std(DO), tf.reduce_sum(DO)], message='', summarize=1000)

        conv = self.l0.forward(AI)
        bn   = self.l1.forward(conv)
        relu = self.l2.forward(bn)
        res = tf.concat((relu, AI), axis=3)
        # res = tf.concat((relu, tf.zeros_like(AI)), axis=3)
        # res = relu

        dlel  = self.l3.lel_backward(res, res, None, None, Y)
        drelu = self.l2.lel_backward(bn, relu, None, dlel[:, :, :, 0 : self.fout] + DO, Y)
        # drelu = self.l2.lel_backward(bn, relu, None, dlel, Y)
        dbn   = self.l1.lel_backward(conv, bn, None, drelu, Y)
        dconv = self.l0.lel_backward(AI, conv, None, dbn, Y)

        gvs = []
        
        dconv = self.l0.lel_gv(AI, conv, None, dbn, Y)
        dbn   = self.l1.lel_gv(conv, bn, None, drelu, Y)
        drelu = self.l2.lel_gv(bn, relu, None, dlel[:, :, :, 0 : self.fout] + DO, Y)
        # drelu = self.l2.lel_gv(bn, relu, None, dlel, Y)
        dlel  = self.l3.lel_gv(res, res, None, None, Y)

        gvs.extend(dconv)
        gvs.extend(dbn)
        gvs.extend(drelu)
        gvs.extend(dlel)
        
        return gvs
        
    def lel(self, AI, AO, E, DO, Y): 
        assert(False)
        
    ###################################################################   
    
    
    
    
    
    
    
