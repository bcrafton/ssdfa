
# https://pypi.org/project/python-mnist/
# pretty nice.

import numpy as np
from mnist import MNIST

emnist_data = MNIST(path='./gzip/', return_type='numpy')
emnist_data.select_emnist('letters')
x_orig, y_orig = emnist_data.load_training()

print (np.shape(x_orig)) 
