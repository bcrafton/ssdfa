
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

def viz_conv(name, weight):
    weights = np.load(name + '.npy').item()
    conv1 = weights[weight]
    viz(name + '_' + weight, conv1)

###############################################

viz_conv('imagenet_bp1', 'conv1')
viz_conv('imagenet_bp2', 'conv1')
viz_conv('imagenet_dfa1', 'conv1')
viz_conv('imagenet_dfa2', 'conv1')

# viz_conv('imagenet_bp1', 'conv2')
viz_conv('imagenet_bp2', 'conv2')
# viz_conv('imagenet_dfa1', 'conv2')
viz_conv('imagenet_dfa2', 'conv2')

viz_conv('imagenet_dfa1_load', 'conv1')

