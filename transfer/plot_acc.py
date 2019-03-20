
import numpy as np
import matplotlib.cm as cmap
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

weights = np.load('cifar10_conv_weights.npy').item()
x = range(0, 100, 2)
y = np.array(weights['test_acc'])[x]
plt.plot(x, y, label='BP', marker='^', linestyle='None')

weights = np.load('cifar10_conv_weights_fa.npy').item()
x = range(0, 100, 2)
y = np.array(weights['test_acc'])[x]
plt.plot(x, y, label='FA Structured', marker='*', linestyle='None')

weights = np.load('cifar10_conv_weights_random.npy').item()
x = range(0, 100, 2)
y = np.array(weights['test_acc'])[x]
plt.plot(x, y, label='FA Random', marker="v", linestyle='None')

plt.rcParams['font.sans-serif'] = "Arial"
plt.rcParams['font.family'] = "sans-serif"
plt.rcParams['font.size'] = 10.

plt.legend()
plt.savefig('plot_acc.png')
