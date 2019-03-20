
import numpy as np
import matplotlib.cm as cmap
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

weights = np.load('./transfer/cifar10_conv_weights.npy').item()
np.savetxt('train.csv', weights['train_acc'], delimiter=',')
np.savetxt('test.csv', weights['test_acc'], delimiter=',')

weights = weights['conv1']
weights = np.transpose(weights)
print (np.shape(weights))
# 288 11x11 filters
# 96x3, 32x9, 16x18
weights = np.reshape(weights, (16, 18, 5, 5))

# does not work
# fig = plt.figure(figsize=(2, 2))

for ii in range(16):
    for jj in range(18):
        if jj == 0:
            row = weights[ii][jj]
        else:
            row = np.concatenate((row, weights[ii][jj]), axis=1)
            
    if ii == 0:
        img = row
    else:
        img = np.concatenate((img, row), axis=0)
  
plt.imsave("cifar10_conv_weights.png", img, cmap="gray", dpi=300)

###

weights = np.load('./transfer/cifar10_conv_weights_fa.npy').item()
np.savetxt('fa_train.csv', weights['train_acc'], delimiter=',')
np.savetxt('fa_test.csv', weights['test_acc'], delimiter=',')

weights = weights['conv1']
weights = np.transpose(weights)
print (np.shape(weights))
# 288 5x5 filters
# 96x3, 32x9, 16x18
weights = np.reshape(weights, (16, 18, 5, 5))

# does not work
# fig = plt.figure(figsize=(2, 2))

for ii in range(16):
    for jj in range(18):
        if jj == 0:
            row = weights[ii][jj]
        else:
            row = np.concatenate((row, weights[ii][jj]), axis=1)

    if ii == 0:
        img = row
    else:
        img = np.concatenate((img, row), axis=0)

plt.imsave("cifar10_conv_weights_fa.png", img, cmap="gray", dpi=300)

###



