
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
# use act=tanh, bias=0
# use act=relu, bias=1

imagenet_vgg_bp = {'benchmark':'imagenet_vgg.py', 'epochs':10, 'batch_size':32, 'alpha':[1e-2], 'l2':[0.], 'eps':[1.], 'act':['tanh'], 'bias':[0.], 'dropout':[0.5], 'dfa':0, 'sparse':0, 'rank':0, 'init':['alexnet'], 'opt':'adam', 'load':None}
imagenet_vgg_dfa = {'benchmark':'imagenet_vgg.py', 'epochs':1000, 'batch_size':32, 'alpha':[1e-2], 'l2':[0.], 'eps':[1.01], 'act':['tanh'], 'bias':[0.], 'dropout':[0.5], 'dfa':1, 'sparse':0, 'rank':0, 'init':['zero'], 'opt':'adam', 'load':None}

imagenet_vgg_sparse1 = {'benchmark':'vgg_fc.py', 'epochs':100, 'batch_size':64, 'alpha':[0.05], 'l2':[0.], 'eps':[1.], 'act':['tanh'], 'bias':[0.], 'dropout':[0.5], 'dfa':1, 'sparse':1, 'rank':0, 'init':['zero'], 'opt':'adam', 'load':None}
imagenet_vgg_sparse2 = {'benchmark':'vgg_fc.py', 'epochs':100, 'batch_size':32, 'alpha':[0.0001], 'l2':[0.], 'eps':[1.], 'act':['relu'], 'bias':[1.], 'dropout':[0.1], 'dfa':1, 'sparse':1, 'rank':0, 'init':['zero'], 'opt':'adam', 'load':None}
imagenet_vgg_sparse3 = {'benchmark':'vgg_fc.py', 'epochs':100, 'batch_size':32, 'alpha':[0.0001], 'l2':[0.], 'eps':[1.], 'act':['relu'], 'bias':[1.], 'dropout':[0.25], 'dfa':1, 'sparse':1, 'rank':0, 'init':['zero'], 'opt':'adam', 'load':None}

################################################

# params = [imagenet_vgg_bp]
# params = [imagenet_vgg_dfa]
# params = [imagenet_vgg_sparse]
# params = [imagenet_vgg_bp, imagenet_vgg_dfa, imagenet_vgg_sparse]
# params = [imagenet_vgg_dfa, imagenet_vgg_sparse]
# params = [imagenet_vgg_sparse1, imagenet_vgg_sparse2, imagenet_vgg_sparse3]
params = [imagenet_vgg_sparse2, imagenet_vgg_sparse3]

################################################

def get_runs():
    runs = []

    for param in params:
        perms = get_perms(param)
        runs.extend(perms)

    return runs

################################################

        
