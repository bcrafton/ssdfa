
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
# MobileNet
################################################

mobile_net_dw_64 = {'benchmark':'MobileNetDW.py', 'epochs':100, 'batch_size':128, 'alpha':[1e-2, 3e-3, 1e-3], 'l2':[0.], 'eps':[1.], 'act':['relu'], 'bias':[0.], 'dropout':[0.5], 'dfa':0, 'sparse':0, 'rank':0, 'init':['alexnet'], 'opt':'adam', 'load':None}

mobile_net = {'benchmark':'MobileNetImageNet.py', 'epochs':100, 'batch_size':64, 'alpha':[1e-2], 'l2':[0.], 'eps':[1.], 'act':['relu'], 'bias':[0.], 'dropout':[0.5], 'dfa':0, 'sparse':0, 'rank':0, 'init':['alexnet'], 'opt':'adam', 'load':None}

mobile_net_64 = {'benchmark':'MobileNet64.py', 'epochs':100, 'batch_size':128, 'alpha':[5e-2, 1e-2], 'l2':[0.], 'eps':[1.], 'act':['relu'], 'bias':[0.], 'dropout':[0.5], 'dfa':0, 'sparse':0, 'rank':0, 'init':['alexnet'], 'opt':'adam', 'load':None}

mobile_net_224 = {'benchmark':'MobileNet224.py', 'epochs':100, 'batch_size':64, 'alpha':[1e-2], 'l2':[0.], 'eps':[1.], 'act':['relu'], 'bias':[0.], 'dropout':[0.5], 'dfa':0, 'sparse':0, 'rank':0, 'init':['alexnet'], 'opt':'adam', 'load':None}

################################################

vgg64 = {'benchmark':'vgg64.py', 'epochs':100, 'batch_size':64, 'alpha':[0.03, 0.01], 'l2':[0.], 'eps':[1.], 'act':['relu'], 'bias':[0.], 'dropout':[0.5], 'dfa':0, 'sparse':0, 'rank':0, 'init':['alexnet'], 'opt':'adam', 'load':None}

vgg64_stride = {'benchmark':'vgg64_stride.py', 'epochs':5, 'batch_size':64, 'alpha':[0.03, 0.01], 'l2':[0.], 'eps':[1.], 'act':['relu'], 'bias':[0.], 'dropout':[0.5], 'dfa':0, 'sparse':0, 'rank':0, 'init':['alexnet'], 'opt':'adam', 'load':None}

vgg64_lel = {'benchmark':'vgg64_lel.py', 'epochs':5, 'batch_size':64, 'alpha':[0.01, 0.03], 'l2':[0.], 'eps':[1.], 'act':['relu'], 'bias':[0.], 'dropout':[0.5], 'dfa':[0, 1], 'sparse':0, 'rank':0, 'init':['alexnet'], 'opt':'adam', 'load':None}

vgg64_mlp = {'benchmark':'vgg64_mlp.py', 'epochs':5, 'batch_size':64, 'alpha':[0.01], 'l2':[0.], 'eps':[1.], 'act':['relu'], 'bias':[0.], 'dropout':[0.25], 'dfa':0, 'sparse':0, 'rank':0, 'init':['sqrt_fan_in'], 'opt':'adam', 'load':None}

vgg64_2fc = {'benchmark':'vgg64_2fc.py', 'epochs':30, 'batch_size':64, 'alpha':[0.03, 0.01], 'l2':[0.], 'eps':[1.], 'act':['relu'], 'bias':[0.], 'dropout':[0.5], 'dfa':[0, 1], 'sparse':0, 'rank':0, 'init':['alexnet'], 'opt':'adam', 'load':None}

vgg64_lel_2fc = {'benchmark':'vgg64_lel_2fc.py', 'epochs':30, 'batch_size':64, 'alpha':[0.01, 0.03, 0.05], 'l2':[0.], 'eps':[1.], 'act':['relu'], 'bias':[0.], 'dropout':[0.5], 'dfa':[1], 'sparse':0, 'rank':0, 'init':['alexnet'], 'opt':'adam', 'load':None}

################################################

# vgg64_ae = {'benchmark':'vgg64_autoencoder22.py', 'epochs':10, 'batch_size':32, 'alpha':[0.01], 'l2':[0.], 'eps':[1.], 'act':['relu'], 'bias':[0.], 'dropout':[0.5], 'dfa':0, 'sparse':0, 'rank':0, 'init':['alexnet'], 'opt':'adam', 'load':['vgg64_bp.npy', 'vgg64_lel.npy']}

# vgg64_ae_scratch = {'benchmark':'vgg64_autoencoder18.py', 'epochs':10, 'batch_size':32, 'alpha':[0.001, 0.0001], 'l2':[0.], 'eps':[1.], 'act':['relu'], 'bias':[0.], 'dropout':[0.5], 'dfa':0, 'sparse':0, 'rank':0, 'init':['alexnet'], 'opt':'adam', 'load':None}

vgg64_pool = {'benchmark':'vgg64_pool_block1.py', 'epochs':20, 'batch_size':64, 'alpha':[0.05], 'l2':[0.], 'eps':[1.], 'act':['relu'], 'bias':[0.], 'dropout':[0.5], 'dfa':[1], 'sparse':0, 'rank':0, 'init':['alexnet'], 'opt':'adam', 'ae_loss':[0.0], 'load':None}

################################################

# mobile224 = {'benchmark':'MobileNet224.py', 'epochs':100, 'batch_size':64, 'alpha':[5e-2], 'l2':[0.], 'eps':[1.], 'act':['relu'], 'bias':[0.], 'dropout':[0.5], 'dfa':1, 'sparse':0, 'rank':0, 'init':['alexnet'], 'opt':'adam', 'load':None}

vgg224 = {'benchmark':'VGG224_2.py', 'epochs':100, 'batch_size':32, 'alpha':[5e-2], 'l2':[0.], 'eps':[1.], 'act':['relu'], 'bias':[0.], 'dropout':[0.5], 'dfa':[1], 'sparse':0, 'rank':0, 'init':['alexnet'], 'opt':'adam', 'ae_loss':[1.0], 'load':None}

# vgg224_ae = {'benchmark':'VGG224_AE.py', 'epochs':10, 'batch_size':32, 'alpha':[0.07], 'l2':[0.], 'eps':[1.], 'act':['relu'], 'bias':[0.], 'dropout':[0.5], 'dfa':0, 'sparse':0, 'rank':0, 'init':['alexnet'], 'opt':'adam', 'load':['vgg224_bp.npy', 'vgg224_lel.npy']}

mobile64 = {'benchmark':'MobileNet64.py', 'epochs':20, 'batch_size':64, 'alpha':[0.05], 'l2':[0.], 'eps':[1.], 'act':['relu'], 'bias':[0.], 'dropout':[0.5], 'dfa':[0, 1], 'sparse':0, 'rank':0, 'init':['alexnet'], 'opt':'adam', 'ae_loss':[0.0], 'load':None}

################################################

# params = [vgg64_ae, vgg64_pool]
params = [vgg64_pool]
# params = [vgg64_ae]
# params = [vgg64_ae_scratch]
# params = [mobile64]

# params = [mobile224, vgg224]
# params = [vgg224]
# params = [mobile224]
# params = [vgg224_ae]

################################################

def get_runs():
    runs = []

    for param in params:
        perms = get_perms(param)
        runs.extend(perms)

    return runs

################################################

        
