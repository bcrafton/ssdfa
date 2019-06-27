
import numpy as np

x = np.load('vgg64_pool_block.py_0.050000_0.000000_1.000000_relu_0.000000_0.500000_1_0_alexnet_adam_3.000000.npy').item()

for key in x.keys():
    print (key)

x['conv1'] = x['block1_conv_block_conv']
x['conv1_bias'] = x['block1_conv_block_conv_bias']

x['conv2'] = x['block2_conv_block_conv']
x['conv2_bias'] = x['block2_conv_block_conv_bias']

x['conv3'] = x['block3_conv_block_conv']
x['conv3_bias'] = x['block3_conv_block_conv_bias']

x['conv4'] = x['block4_conv_block_conv']
x['conv4_bias'] = x['block4_conv_block_conv_bias']

x['conv5'] = x['block5_conv_block_conv']
x['conv5_bias'] = x['block5_conv_block_conv_bias']

x['conv6'] = x['block6_conv_block_conv']
x['conv6_bias'] = x['block6_conv_block_conv_bias']

x['conv7'] = x['block7_conv_block_conv']
x['conv7_bias'] = x['block7_conv_block_conv_bias']

x['conv8'] = x['block8_conv_block_conv']
x['conv8_bias'] = x['block8_conv_block_conv_bias']

np.save('vgg64_lel_3', x)
