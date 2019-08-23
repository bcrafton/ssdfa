
import numpy as np
import os
import copy
import threading
import argparse
from results import get_runs

##############################################

runs = get_runs()

##############################################

results = {}

num_runs = len(runs)
for ii in range(num_runs):
    param = runs[ii]

    name = '%s_%f_%f_%s_%f_%f_%d_%d_%s.npy' % (param['benchmark'], 
                                               param['lr'], 
                                               param['eps'], 
                                               param['act'], 
                                               param['bias'], 
                                               param['dropout'], 
                                               param['dfa'], 
                                               param['sparse'], 
                                               param['init']
                                               )

    res = np.load(name, allow_pickle=True).item()
    key = (param['benchmark'], param['dfa'], param['sparse'])
    val = max(res['test_acc'])

    print (name, val)
    
    if key in results.keys():
        if results[key][0] < val:
            results[key] = (val, param['benchmark'], param['lr'], param['eps'], param['act'], param['dfa'], param['sparse'], param['init'], name)
    else:
        results[key] = (val, param['benchmark'], param['lr'], param['eps'], param['act'], param['dfa'], param['sparse'], param['init'], name)
            
for key in sorted(results.keys()):   
    print (key, results[key])





        
