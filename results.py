
import numpy as np
import os
import copy
import threading
import argparse

################################################

def get_perms(param):
    params = [param]
    
    for key in param.keys():
        val = param[key]
        if type(val) == list:
            new_params = []
            for ii in range(len(val)):
                for jj in range(len(params)):
                    new_param = copy.copy(params[jj])
                    new_param[key] = val[ii]
                    new_params.append(new_param)
                    
            params = new_params
            
    return params

################################################
# FC
################################################

mnist_fc_bp = {'benchmark':'mnist_fc.py', 'epochs':300, 'batch_size':32, 'alpha':[0.1, 0.05, 0.03, 0.01], 'l2':[0.], 'eps':[1.], 'act':['tanh'], 'bias':[0.0], 'dropout':[0.0], 'dfa':0, 'sparse':0, 'rank':0, 'init':'sqrt_fan_in', 'opt':'adam', 'load':None}
mnist_fc_dfa = {'benchmark':'mnist_fc.py', 'epochs':300, 'batch_size':32, 'alpha':[0.1, 0.05, 0.03, 0.01], 'l2':[0.], 'eps':[1.], 'act':['tanh'], 'bias':[0.0], 'dropout':[0.0], 'dfa':1, 'sparse':0, 'rank':0, 'init':'zero', 'opt':'adam', 'load':None}
mnist_fc_sparse = {'benchmark':'mnist_fc.py', 'epochs':300, 'batch_size':32, 'alpha':[0.1, 0.05, 0.03, 0.01], 'l2':[0.], 'eps':[1.], 'act':['tanh'], 'bias':[0.0], 'dropout':[0.0], 'dfa':1, 'sparse':1, 'rank':0, 'init':'zero', 'opt':'adam', 'load':None}

cifar10_fc_bp = {'benchmark':'cifar10_fc.py', 'epochs':300, 'batch_size':64, 'alpha':[3e-5], 'l2':[0.], 'eps':[1e-6], 'act':['relu'], 'bias':[0.1], 'dropout':[0.25], 'dfa':0, 'sparse':0, 'rank':0, 'init':'sqrt_fan_in', 'opt':['adam'], 'load':None}
cifar10_fc_dfa = {'benchmark':'cifar10_fc.py', 'epochs':300, 'batch_size':64, 'alpha':[1e-4], 'l2':[0.], 'eps':[1e-6], 'act':['relu'], 'bias':[0.1], 'dropout':[0.25], 'dfa':1, 'sparse':0, 'rank':0, 'init':'zero', 'opt':['adam'], 'load':None}
cifar10_fc_sparse = {'benchmark':'cifar10_fc.py', 'epochs':500, 'batch_size':64, 'alpha':[1e-4], 'l2':[0.], 'eps':[1e-4], 'act':['relu'], 'bias':[0.1], 'dropout':[0.25], 'dfa':1, 'sparse':1, 'rank':0, 'init':'zero', 'opt':['adam'], 'load':None}

cifar100_fc_bp = {'benchmark':'cifar100_fc.py', 'epochs':300, 'batch_size':64, 'alpha':[1e-4, 3e-5, 1e-5], 'l2':[0.], 'eps':[1e-4, 1e-5, 1e-6], 'act':['tanh', 'relu'], 'bias':[0.1], 'dropout':[0.25, 0.5], 'dfa':0, 'sparse':[0], 'rank':0, 'init':'sqrt_fan_in', 'opt':['adam'], 'load':None}
cifar100_fc_dfa = {'benchmark':'cifar100_fc.py', 'epochs':300, 'batch_size':64, 'alpha':[1e-4, 3e-5, 1e-5], 'l2':[0.], 'eps':[1e-4, 1e-5, 1e-6], 'act':['tanh', 'relu'], 'bias':[0.1], 'dropout':[0.25, 0.5], 'dfa':1, 'sparse':[0], 'rank':0, 'init':'zero', 'opt':['adam'], 'load':None}
cifar100_fc_sparse = {'benchmark':'cifar100_fc.py', 'epochs':500, 'batch_size':64, 'alpha':[1e-4, 3e-5, 1e-5], 'l2':[0.], 'eps':[1e-4, 1e-5, 1e-6], 'act':['tanh', 'relu'], 'bias':[0.1], 'dropout':[0.25, 0.5], 'dfa':1, 'sparse':[1], 'rank':0, 'init':'zero', 'opt':['adam'], 'load':None}

################################################
# CONV
################################################

mnist_conv_bp = {'benchmark':'mnist_conv.py', 'epochs':300, 'batch_size':64, 'alpha':[0.01, 0.005], 'l2':[0.], 'eps':[1.], 'act':['tanh'], 'bias':[0.0], 'dropout':[0.25], 'dfa':0, 'sparse':0, 'rank':0, 'init':'sqrt_fan_in', 'opt':'adam', 'load':None}
mnist_conv_dfa = {'benchmark':'mnist_conv.py', 'epochs':300, 'batch_size':64, 'alpha':[0.01, 0.005], 'l2':[0.], 'eps':[1.], 'act':['tanh'], 'bias':[0.0], 'dropout':[0.25], 'dfa':1, 'sparse':0, 'rank':0, 'init':'zero', 'opt':'adam', 'load':None}
mnist_conv_sparse = {'benchmark':'mnist_conv.py', 'epochs':300, 'batch_size':64, 'alpha':[0.01, 0.005], 'l2':[0.], 'eps':[1.], 'act':['tanh'], 'bias':[0.0], 'dropout':[0.25], 'dfa':1, 'sparse':1, 'rank':0, 'init':'zero', 'opt':'adam', 'load':None}

cifar10_conv_bp = {'benchmark':'cifar10_conv.py', 'epochs':500, 'batch_size':64, 'alpha':[3e-5], 'l2':[0.], 'eps':[1e-4], 'act':['relu'], 'bias':[0.1], 'dropout':[0.25], 'dfa':0, 'sparse':0, 'rank':0, 'init':'sqrt_fan_in', 'opt':['adam'], 'load':None}
cifar10_conv_dfa = {'benchmark':'cifar10_conv.py', 'epochs':500, 'batch_size':64, 'alpha':[1e-5], 'l2':[0.], 'eps':[1e-5], 'act':['relu'], 'bias':[0.1], 'dropout':[0.5], 'dfa':1, 'sparse':0, 'rank':0, 'init':'zero', 'opt':['adam'], 'load':None}
cifar10_conv_sparse = {'benchmark':'cifar10_conv.py', 'epochs':500, 'batch_size':64, 'alpha':[1e-5], 'l2':[0.], 'eps':[1e-5], 'act':['tanh'], 'bias':[0.0], 'dropout':[0.5], 'dfa':1, 'sparse':1, 'rank':0, 'init':'zero', 'opt':['adam'], 'load':None}

cifar100_conv_bp = {'benchmark':'cifar100_conv.py', 'epochs':300, 'batch_size':64, 'alpha':[3e-5], 'l2':[0.], 'eps':[1e-6], 'act':['relu'], 'bias':[0.1], 'dropout':[0.5], 'dfa':0, 'sparse':0, 'rank':0, 'init':'sqrt_fan_in', 'opt':['adam'], 'load':None}
cifar100_conv_dfa = {'benchmark':'cifar100_conv.py', 'epochs':500, 'batch_size':64, 'alpha':[1e-5], 'l2':[0.], 'eps':[1e-5], 'act':['tanh'], 'bias':[0.0], 'dropout':[0.25], 'dfa':1, 'sparse':0, 'rank':0, 'init':'zero', 'opt':['adam'], 'load':None}
cifar100_conv_sparse = {'benchmark':'cifar100_conv.py', 'epochs':500, 'batch_size':64, 'alpha':[1e-5], 'l2':[0.], 'eps':[1e-4], 'act':['tanh'], 'bias':[0.0], 'dropout':[0.25], 'dfa':1, 'sparse':1, 'rank':0, 'init':'zero', 'opt':['adam'], 'load':None}

################################################
# vgg
################################################

# use act=tanh, bias=0
# use act=relu, bias=1

imagenet_vgg_bp = {'benchmark':'imagenet_vgg.py', 'epochs':10, 'batch_size':32, 'alpha':[1e-2], 'l2':[0.], 'eps':[1.], 'act':['tanh'], 'bias':[0.], 'dropout':[0.5], 'dfa':0, 'sparse':0, 'rank':0, 'init':['alexnet'], 'opt':'adam', 'load':None}
imagenet_vgg_dfa = {'benchmark':'imagenet_vgg.py', 'epochs':1000, 'batch_size':32, 'alpha':[1e-2], 'l2':[0.], 'eps':[1.01], 'act':['tanh'], 'bias':[0.], 'dropout':[0.5], 'dfa':1, 'sparse':0, 'rank':0, 'init':['zero'], 'opt':'adam', 'load':None}

imagenet_vgg_sparse1 = {'benchmark':'vgg_fc.py', 'epochs':100, 'batch_size':64, 'alpha':[0.05], 'l2':[0.], 'eps':[1.], 'act':['tanh'], 'bias':[0.], 'dropout':[0.5], 'dfa':1, 'sparse':1, 'rank':0, 'init':['zero'], 'opt':'adam', 'load':None}
imagenet_vgg_sparse2 = {'benchmark':'vgg_fc.py', 'epochs':100, 'batch_size':32, 'alpha':[0.0001], 'l2':[0.], 'eps':[1.], 'act':['relu'], 'bias':[1.], 'dropout':[0.1], 'dfa':1, 'sparse':1, 'rank':0, 'init':['zero'], 'opt':'adam', 'load':None}
imagenet_vgg_sparse3 = {'benchmark':'vgg_fc.py', 'epochs':100, 'batch_size':32, 'alpha':[0.0001], 'l2':[0.], 'eps':[1.], 'act':['relu'], 'bias':[1.], 'dropout':[0.25], 'dfa':1, 'sparse':1, 'rank':0, 'init':['zero'], 'opt':'adam', 'load':None}

################################################

vgg64 = {'benchmark':'vgg64.py', 'epochs':100, 'batch_size':64, 'alpha':[0.01], 'l2':[0.], 'eps':[1.], 'act':['relu'], 'bias':[0.], 'dropout':[0.5], 'dfa':0, 'sparse':0, 'rank':0, 'init':['alexnet'], 'opt':'adam', 'load':None}
vgg64_mlp = {'benchmark':'vgg64_mlp.py', 'epochs':100, 'batch_size':64, 'alpha':[0.01], 'l2':[0.], 'eps':[1.], 'act':['relu'], 'bias':[0.], 'dropout':[0.25], 'dfa':0, 'sparse':0, 'rank':0, 'init':['sqrt_fan_in'], 'opt':'adam', 'load':None}

################################################

# params = [mnist_fc_bp, mnist_fc_dfa, mnist_fc_sparse]
# params = [cifar10_fc_bp, cifar10_fc_dfa, cifar10_fc_sparse]
# params = [cifar100_fc_bp, cifar100_fc_dfa, cifar100_fc_sparse]

# params = [mnist_conv_bp, mnist_conv_dfa, mnist_conv_sparse]
# params = [cifar10_conv_bp, cifar10_conv_dfa, cifar10_conv_sparse]
# params = [cifar100_conv_bp, cifar100_conv_dfa, cifar100_conv_sparse]

params = [vgg64]
# params = [vgg64_mlp]

################################################

def get_runs():
    runs = []

    for param in params:
        perms = get_perms(param)
        runs.extend(perms)

    return runs

################################################

        
