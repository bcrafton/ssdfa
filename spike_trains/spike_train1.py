
import numpy as np

cmp_arr = np.random.uniform(low=0., high=1., size=64)
def to_spike_train(x, cmp_arr):
    x = x / np.max(x)
    return cmp_arr < x  
