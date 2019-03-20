
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

imagenet = {'benchmark':'imagenet.py', 'epochs':100, 'batch_size':128, 'alpha':[1e-2], 'l2':[0.], 'eps':[1.], 'act':['relu'], 'bias':[0.], 'dropout':[0.5], 'dfa':0, 'fa':1, 'sparse':0, 'rank':0, 'init':['zero'], 'opt':'adam', 'load':None}

cifar10_conv = {'benchmark':'cifar10_conv.py', 'epochs':25, 'batch_size':64, 'alpha':[3e-5, 1e-5, 3e-6, 1e-6], 'l2':[0.], 'eps':[1e-4, 1e-5, 1e-6], 'act':['tanh', 'relu'], 'bias':[0.1, 1.0], 'dropout':[0.25, 0.5], 'dfa':0, 'fa':[1], 'sparse':0, 'rank':0, 'init':['sqrt_fan_in', 'zero'], 'opt':['adam'], 'load':None}

# cifar10_conv = {'benchmark':'cifar10_conv.py', 'epochs':100, 'batch_size':64, 'alpha':[1e-5], 'l2':[0.], 'eps':[1e-5], 'act':['relu'], 'bias':[0.1], 'dropout':[0.5], 'dfa':0, 'fa':1, 'sparse':0, 'rank':0, 'init':'zero', 'opt':['adam'], 'load':None}

################################################

params = [cifar10_conv]

################################################

def get_runs():
    runs = []

    for param in params:
        perms = get_perms(param)
        runs.extend(perms)

    return runs

################################################

        
