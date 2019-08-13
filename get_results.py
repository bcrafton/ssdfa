
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

    name = '%s_%s_%f_%f_%f_%s.npy' % (
            param['benchmark'], 
            param['model'], 
            param['lr'], 
            param['eps'],
            param['dropout'], 
            param['init']
            )

    res = np.load(name, allow_pickle=True).item()
    key = (param['benchmark'], param['model'], param['lr'])
    val = max(res['val_acc'])

    print (name, val)
    
    if key in results.keys():
        if results[key][0] < val:
            results[key] = (val, param['benchmark'], param['model'])
    else:
        results[key] = (val, param['benchmark'], param['model'])
            
for key in sorted(results.keys()):   
    print (key, results[key])





        
