
import numpy as np
import os
import copy
import threading
import argparse

from results import get_runs

##############################################

parser = argparse.ArgumentParser()
parser.add_argument('--print', type=int, default=0)
cmd_args = parser.parse_args()

##############################################

num_gpus = 5
counter = 0

def run_command(param):
    global num_gpus, counter

    if num_gpus == 0:
        gpu = -1
    else:
        gpu = counter % num_gpus
        counter = counter + 1
    
    name = '%s_%s_%f_%f_%f_%s_%s_%s_%s' % (
            param['benchmark'], 
            param['model'], 
            param['lr'], 
            param['eps'],
            param['dropout'], 
            param['init'],
            param['fb'],
            param['fb_dw'], 
            param['fb_pw']
            )
             
    cmd = "python36 %s --model %s --gpu %d --epochs %d --batch_size %d --lr %f --eps %f --dropout %f --init %s --save %d --name %s --fb %s --fb_dw %s --fb_pw %s" % (
           param['benchmark'], 
           param['model'], 
           gpu, 
           param['epochs'], 
           param['batch_size'], 
           param['lr'], 
           param['eps'], 
           param['dropout'], 
           param['init'], 
           1, 
           name,
           param['fb'],
           param['fb_dw'], 
           param['fb_pw']
           )

    if cmd_args.print:
        print (cmd)
    else:
        os.system(cmd)

    return

##############################################

runs = get_runs()

##############################################

num_runs = len(runs)
parallel_runs = num_gpus

for run in range(0, num_runs, parallel_runs):
    threads = []
    for parallel_run in range( min(parallel_runs, num_runs - run)):
        args = runs[run + parallel_run]
        t = threading.Thread(target=run_command, args=(args,))
        threads.append(t)
        t.start()
    for t in threads:
        t.join()
        
