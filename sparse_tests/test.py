
import itertools
import numpy as np

input_size = 100
output_size = 100
size = (input_size, output_size)

combs = np.array(list(itertools.product(range(input_size), range(output_size))))
choices = range(len(combs))
idx = np.random.choice(a=choices, size=int(0.01 * np.prod(size)), replace=False).tolist()

print (np.shape(combs), np.shape(idx))
idx = combs[idx]
print (idx)
