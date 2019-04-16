
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
            _filter = np.pad(filters[ii][jj], ((1, 1), (1, 1)), 'constant')
            if jj == 0:
                row = _filter
            else:
                row = np.concatenate((row, _filter), axis=1)
                
        if ii == 0:
            img = row
        else:
            img = np.concatenate((img, row), axis=0)
            
    plt.imsave(name, img, cmap="gray")

###############################################
'''
filters = np.load('autoencoder.npy').item()
filters = filters['conv2']
viz('filters2.png', filters)
'''

filters = np.load('autoencoder.npy').item()
filters = filters['conv1']
shape = np.shape(filters)
print (shape)
filters = np.reshape(filters, (shape[0], shape[1], shape[2], shape[3] * shape[4]))
viz('filters1.png', filters)

filters = np.load('autoencoder.npy').item()
filters = filters['conv2']
shape = np.shape(filters)
print (shape)
filters = np.reshape(filters, (shape[0], shape[1], shape[2], shape[3] * shape[4]))
viz('filters2.png', filters)
