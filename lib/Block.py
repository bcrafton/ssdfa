
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
        # self.pad = int((self.fout - self.fin) / 2)
        # print (self.fin, self.fout, self.pad)

        self.res_filter_shape = [1, 1, self.fin, self.fout]
        self.res = Convolution(input_sizes=self.input_shape, filter_sizes=self.res_filter_shape, init=self.init, strides=[1,1,1,1], padding="SAME", name=self.name + '_res_conv')

        self.l0 = Convolution(input_sizes=self.input_shape, filter_sizes=self.filter_shape, init=self.init, strides=[1,1,1,1], padding="SAME", name=self.name + '_conv')
        self.l1 = BatchNorm(input_size=self.output_shape, name=self.name + '_bn')
        self.l2 = Relu()
        self.l3 = LELConv(input_shape=self.output_shape, pool_shape=self.pool_shape, num_classes=self.num_classes, name=self.name + '_fb')

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
        
        res  = self.res.forward(AI)
        conv = self.l0.forward(AI)
        bn   = self.l1.forward(conv)
        relu = self.l2.forward(bn)
        lel  = self.l3.forward(relu)
        
        dlel  = self.l3.lel_backward(res + relu, res + relu, None, None, Y)
        drelu = self.l2.lel_backward(bn, relu, None, dlel + DO, Y)
        dbn   = self.l1.lel_backward(conv, bn, None, drelu, Y)
        dconv = self.l0.lel_backward(AI, conv, None, dbn, Y)
        dres  = self.res.lel_backward(AI, res, None, dlel, Y)

        # dlel = tf.Print(dlel, ['dlel', tf.shape(dlel), 'dres', tf.shape(dres)], message='', summarize=1000)
        # dres = tf.Print(dres, ['dlel', tf.shape(dlel), 'dres', tf.shape(dres)], message='', summarize=1000)

        return dres

    def lel_gv(self, AI, AO, E, DO, Y):
        
        res  = self.res.forward(AI)
        conv = self.l0.forward(AI)
        bn   = self.l1.forward(conv)
        relu = self.l2.forward(bn)
        lel  = self.l3.forward(relu)
        
        dlel  = self.l3.lel_backward(res + relu, res + relu, None, None, Y)
        drelu = self.l2.lel_backward(bn, relu, None, dlel + DO, Y)
        dbn   = self.l1.lel_backward(conv, bn, None, drelu, Y)
        dconv = self.l0.lel_backward(AI, conv, None, dbn, Y)
        dres  = self.res.lel_backward(AI, res, None, dlel, Y)
        
        # dlel  = tf.Print(dlel, [self.name, 'ai', tf.shape(AI), 'ao', tf.shape(AO), 'dlel', tf.shape(dlel), 'dres', tf.shape(dres), 'do', tf.shape(DO)], message='', summarize=1000)

        # dres = DI
        # drelu = tf.Print(drelu, [tf.shape(AI), tf.shape(AO), tf.shape(DO), tf.shape(dres)], message='', summarize=1000)

        gvs = []
        
        dres  = self.res.lel_gv(AI, res, None, dlel, Y)
        dconv = self.l0.lel_gv(AI, conv, None, dbn, Y)
        dbn   = self.l1.lel_gv(conv, bn, None, drelu, Y)
        drelu = self.l2.lel_gv(bn, relu, None, dlel + DO, Y)
        dlel  = self.l3.lel_gv(res + relu, res + relu, None, None, Y)

        gvs.extend(dres)
        gvs.extend(dconv)
        gvs.extend(dbn)
        gvs.extend(drelu)
        gvs.extend(dlel)
        
        return gvs
        
    def lel(self, AI, AO, E, DO, Y): 
        assert(False)
        
    ###################################################################   
    
    
    
    
    
    
    
