

import numpy as np

names = ['train_data_batch_1', \
         'train_data_batch_2', \
         'train_data_batch_3', \
         'train_data_batch_4', \
         'train_data_batch_5', \
         'train_data_batch_6', \
         'train_data_batch_7', \
         'train_data_batch_8', \
         'train_data_batch_9', \
         'train_data_batch_10' \
		  ]
'''
data = None
label = None

for name in names:
    chunk = np.load(name)
    _data = chunk['data']
    _label = chunk['labels']
    _mean = chunk['mean']

    if data is None:
        data = _data
        label = _label
    else:
        data = np.append(data, _data, axis=0)
        label = np.append(label, _label)

    # print (np.shape(_label), np.shape(_data))
'''

chunk = np.load('train_data_batch_1')

data = chunk['data']
label = chunk['labels']

print (np.shape(data))
print (np.shape(label))
