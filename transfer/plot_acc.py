
import numpy as np
import matplotlib.cm as cmap
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

weights = np.load('cifar10_conv_weights.npy').item()
plt.plot(weights['test_acc'])

weights = np.load('cifar10_conv_weights_fa.npy').item()
plt.plot(weights['test_acc'])

weights = np.load('cifar10_conv_weights_random.npy').item()
plt.plot(weights['test_acc'])

plt.savefig('plot_acc.png')
