

import numpy as np

def conv_weights(input_sizes, filter_sizes):
    h, w, fin = input_sizes
    fh, fw, fin, fout = filter_sizes
    sqrt_fan_in = np.sqrt(h * w * fin)
    filters = np.random.uniform(low=-1.0/sqrt_fan_in, high=1.0/sqrt_fan_in, size=filter_sizes)
    bias = np.zeros(shape=fout)
    return filters, bias

def fc_weights(size):
    input_size, output_size = size
    sqrt_fan_in = np.sqrt(input_size)
    weights = np.random.uniform(low=-1.0/sqrt_fan_in, high=1.0/sqrt_fan_in, size=size)
    bias = np.zeros(shape=output_size)
    return weights, bias

conv1, conv1_bias = conv_weights(input_sizes=(32, 32, 3), filter_sizes=(5,5,3,96))
conv2, conv2_bias = conv_weights(input_sizes=(16, 16, 96), filter_sizes=(5,5,96,128))
fc1, fc1_bias = fc_weights(size=(8*8*128, 10))

weights = {}

weights['conv1'] = conv1
weights['conv1_bias'] = conv1_bias

weights['conv2'] = conv2
weights['conv2_bias'] = conv2_bias

weights['fc1'] = fc1
weights['fc1_bias'] = fc1_bias

np.save('weights', weights)
