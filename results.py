
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

# NOTE: it makes no sense to use ud01 for pointwise convolutions.

'''
imagenet64_tiny = {'benchmark':'ImageNet64.py', 'model':['tiny'], 'epochs':15, 'batch_size':64, 'lr':[5e-2, 1e-2, 5e-3], 'eps':[1.], 'dropout':0., 'init':['glorot_uniform'], 'load':None, 'fb':['f', 'u01', 'u012'], 'fb_dw':['f'], 'fb_pw':['f']}
imagenet64_mobile = {'benchmark':'ImageNet64.py', 'model':['mobile'], 'epochs':15, 'batch_size':64, 'lr':[5e-2, 1e-2, 5e-3], 'eps':[1.], 'dropout':0., 'init':['glorot_uniform'], 'load':None, 'fb':['f'], 'fb_dw':['u01', 'u012'], 'fb_pw':['u01', 'u012', 'ud0123']}

params = [imagenet64_tiny, imagenet64_mobile]
'''

'''
imagenet64_mobile = {'benchmark':'ImageNet64.py', 'model':['mobile'], 'epochs':5, 'batch_size':64, 'lr':[5e-2, 1e-2], 'eps':[1.], 'dropout':0., 'init':['glorot_uniform'], 'load':None, 'fb':['f'], 'fb_dw':['f', 'u01', 'u012'], 'fb_pw':['f', 'u01', 'u012', 'ud0123']}

params = [imagenet64_mobile]
'''

'''
imagenet64_tiny = {'benchmark':'ImageNet64.py', 'model':['tiny'], 'epochs':50, 'batch_size':64, 'lr':[5e-2, 1e-2], 'eps':[1.], 'dropout':0., 'init':['glorot_uniform'], 'load':None, 'fb':['u01', 'ud01'], 'fb_dw':['f'], 'fb_pw':['f']}
imagenet64_mobile = {'benchmark':'ImageNet64.py', 'model':['mobile'], 'epochs':50, 'batch_size':64, 'lr':[5e-2, 1e-2], 'eps':[1.], 'dropout':0., 'init':['glorot_uniform'], 'load':None, 'fb':['f'], 'fb_dw':['f', 'u01'], 'fb_pw':['f', 'u01']}

params = [imagenet64_tiny, imagenet64_mobile]
'''

'''
imagenet64_vgg = {'benchmark':'ImageNet64.py', 'model':['vgg'], 'epochs':50, 'batch_size':64, 'lr':[5e-2, 1e-2], 'eps':[1.], 'dropout':0., 'init':['glorot_uniform'], 'load':None, 'fb':['u01', 'ud01f'], 'fb_dw':['f'], 'fb_pw':['f']}

params = [imagenet64_vgg]
'''

################################################

imagenet224 =      {'benchmark':'ImageNet224.py', 'model':['dense'], 'epochs':25, 'batch_size':32, 'lr':[5e-2], 'eps':[1.], 'dropout':0.0, 'init':['glorot_uniform'], 'load':None, 
                   'fb':['f_f'], 'fb_dw':['f_f'], 'fb_pw':['f_f']}

imagenet224_usss = {'benchmark':'ImageNet224.py', 'model':['dense'], 'epochs':25, 'batch_size':32, 'lr':[1e-1], 'eps':[1.], 'dropout':0.0, 'init':['glorot_uniform'], 'load':None, 
                   'fb':['mask01_mean01', 'mask01_mean012'], 'fb_dw':['f_f'], 'fb_pw':['mask01_mean012', 'mask01_mean0123']}

################################################

imagenet64 = {'benchmark':'ImageNet64.py', 'model':['dense4'], 'epochs':25, 'batch_size':64, 'lr':[1e-1], 'eps':[1.], 'dropout':0.0, 'init':['glorot_uniform'], 'load':None, 'fb':['f_f'], 'fb_dw':['f_f'], 'fb_pw':['f_f']}

imagenet64_usss = {'benchmark':'ImageNet64.py', 'model':['dense4'], 'epochs':25, 'batch_size':64, 'lr':[1e-1], 'eps':[1.], 'dropout':0.0, 'init':['glorot_uniform'], 'load':None,
                   'fb':['mask01_mean01', 'mean01_mean01'], 'fb_dw':['f_f'], 'fb_pw':['mask01_mean01', 'mean012_mean01']}

################################################

params = [imagenet64_usss]
# params = [imagenet224, imagenet224_usss]

################################################

def get_runs():
    runs = []

    for param in params:
        perms = get_perms(param)
        runs.extend(perms)

    return runs

################################################

        
