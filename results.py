
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

imagenet64_tiny = {'benchmark':'ImageNet64.py', 'model':['tiny'], 'epochs':3, 'batch_size':64, 'lr':[5e-2, 1e-2, 5e-3], 'eps':[1.], 'dropout':0., 'init':['glorot_uniform'], 'load':None, 'fb_conv':['f', 'u01', 'u012'], 'fb_dw':['f'], 'fb_pw':['f']}
imagenet64_mobile = {'benchmark':'ImageNet64.py', 'model':['mobile'], 'epochs':3, 'batch_size':64, 'lr':[5e-2, 1e-2, 5e-3], 'eps':[1.], 'dropout':0., 'init':['glorot_uniform'], 'load':None, 'fb_conv':['f'], 'fb_dw':['u01', 'u012'], 'fb_pw':['u01', 'u012', 'ud01', 'ud012']}

################################################

params = [imagenet64_tiny, imagenet64_mobile]

################################################

def get_runs():
    runs = []

    for param in params:
        perms = get_perms(param)
        runs.extend(perms)

    return runs

################################################

        
