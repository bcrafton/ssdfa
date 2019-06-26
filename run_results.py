
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

num_gpus = 4
counter = 2

def run_command(param):
    global num_gpus, counter

    if num_gpus == 0:
        gpu = -1
    else:
        gpu = counter % num_gpus
        counter = counter + 1
    
    name = '%s_%f_%f_%f_%s_%f_%f_%d_%d_%s_%s' % (param['benchmark'], param['alpha'], param['l2'], param['eps'], param['act'], param['bias'], param['dropout'], param['dfa'], param['sparse'], param['init'], param['opt'], param['ae_loss'])
    if param['load']:
        name += '_' + param['load']
        cmd = "python %s --gpu %d --epochs %d --batch_size %d --alpha %f --l2 %f --eps %f --act %s --bias %f --dropout %f --dfa %d --sparse %d --rank %d --init %s --opt %s --ae_loss %d --save %d --name %s --load %s" % \
              (param['benchmark'], gpu, param['epochs'], param['batch_size'], param['alpha'], param['l2'], param['eps'], param['act'], param['bias'], param['dropout'], param['dfa'], param['sparse'], param['rank'], param['init'], param['opt'], param['ae_loss'], 1, name, param['load'])
    else:
        cmd = "python %s --gpu %d --epochs %d --batch_size %d --alpha %f --l2 %f --eps %f --act %s --bias %f --dropout %f --dfa %d --sparse %d --rank %d --init %s --opt %s --ae_loss %d --save %d --name %s" % \
              (param['benchmark'], gpu, param['epochs'], param['batch_size'], param['alpha'], param['l2'], param['eps'], param['act'], param['bias'], param['dropout'], param['dfa'], param['sparse'], param['rank'], param['init'], param['opt'], param['ae_loss'], 1, name)

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
        
