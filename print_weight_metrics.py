

import numpy as np
# import matplotlib.pyplot as plt

bp = np.load('vgg224_bp.npy', allow_pickle=True).item()
lel = np.load('vgg224_lel.npy', allow_pickle=True).item()
# ae = np.load('vgg64_ae.npy').item()

layers = ['conv1', 'conv2', 'conv3', 'conv4', 'conv5', 'conv6', 'conv7', 'conv8']
# layers = ['conv1_bn_beta', 'conv2_bn_beta', 'conv3_bn_beta', 'conv4_bn_beta', 'conv5_bn_beta', 'conv6_bn_beta', 'conv7_bn_beta', 'conv8_bn_beta']
# layers = ['conv1_bn_gamma', 'conv2_bn_gamma', 'conv3_bn_gamma', 'conv4_bn_gamma', 'conv5_bn_gamma', 'conv6_bn_gamma', 'conv7_bn_gamma', 'conv8_bn_gamma']

layers = [
'block1_conv_block_conv',
'block2_conv_block_conv',
'block3_conv_block_conv',
'block4_conv_block_conv',
'block5_conv_block_conv',
'block6_conv_block_conv',
'block7_conv_block_conv',
'block8_conv_block_conv',
'block9_conv_block_conv',
'block10_conv_block_conv',
'block11_conv_block_conv',
'block12_conv_block_conv',
'block13_conv_block_conv',
'block14_conv_block_conv',
'block15_conv_block_conv',
]

for l in layers:
    lel_l = lel[l]
    bp_l = bp[l]
    # ae_l = ae[l]

    bp_std = np.std(bp_l)
    lel_std = np.std(lel_l)
    # ae_std = np.std(ae_l)

    bp_mean = np.mean(bp_l)
    lel_mean = np.mean(lel_l)
    # ae_mean = np.mean(ae_l)

    print (bp_std, lel_std)
    # print (ae_std, ae_mean)
