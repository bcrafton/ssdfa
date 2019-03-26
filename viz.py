
import numpy as np
import matplotlib.pyplot as plt

def factors(x):
    l = [] 
    for i in range(1, x + 1):
        if x % i == 0:
            l.append(i)
    
    mid = int(len(l) / 2)
    
    if (len(l) % 2 == 1):
        return [l[mid], l[mid]]
    else:
        return l[mid-1:mid+1]


def viz(name, filters):
    fh, fw, fin, fout = np.shape(filters)
    filters = filters.T
    assert(np.shape(filters) == (fout, fin, fw, fh))
    [nrows, ncols] = factors(fin * fout)
    filters = np.reshape(filters, (nrows, ncols, fw, fh))

    for ii in range(nrows):
        for jj in range(ncols):
            if jj == 0:
                row = filters[ii][jj]
            else:
                row = np.concatenate((row, filters[ii][jj]), axis=1)
                
        if ii == 0:
            img = row
        else:
            img = np.concatenate((img, row), axis=0)
            
    plt.imsave(name, img, cmap="gray")

###############################################

# weights = np.load('cifar10_conv_bp.npy').item()
# weights = np.load('imagenet1.npy').item()
# weights = np.load('imagenet.py_0.010000_0.000000_1.000000_relu_0.000000_0.500000_0_0_alexnet_adam.npy').item()
weights = np.load('imagenet_dfa.npy').item()
# weights = np.load('imagenet_dfa1.npy').item()

print (weights.keys())

conv1 = weights['conv1']
viz('conv1', conv1)

conv2 = weights['conv2']
viz('conv2', conv2)

conv3 = weights['conv3']
viz('conv3', conv3)

conv12 = weights['conv12']
viz('conv12', conv12)

###############################################

