

import numpy as np
import matplotlib.pyplot as plt

bp = np.load('vgg64_bp.npy').item()
lel = np.load('vgg64_lel.npy').item()

layers = ['conv1', 'conv2', 'conv3', 'conv4', 'conv5', 'conv6', 'conv7', 'conv8']
# layers = ['conv1', 'conv2']

for l in layers:
    lel_l = lel[l]
    bp_l = bp[l]

    bp_std = np.std(bp_l)
    lel_std = np.std(lel_l)

    bp_mean = np.mean(bp_l)
    lel_mean = np.mean(lel_l)

    print (bp_std, lel_std, bp_mean, lel_mean)
