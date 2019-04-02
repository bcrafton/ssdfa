
import numpy as np
import tensorflow as tf

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

'''
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

# print (np.shape(data))
# print (np.shape(label))
'''

chunk = np.load('val_data')
data = chunk['data']
label = chunk['labels']

data = np.reshape(data, (-1, 3, 64, 64))
data = np.transpose(data, (0, 2, 3, 1))
data = np.reshape(data, (-1, 64*64*3))

num_examples = np.shape(data)[0]
for ii in range(num_examples):
    print (ii)

    name = '../tfrecord/val/%d.tfrecord' % (int(ii))
    with tf.python_io.TFRecordWriter(name) as writer:
        image_raw = data[ii].tostring()
        _feature={
                'label': _int64_feature(int(label[ii])),
                'image_raw': _bytes_feature(image_raw)
                }
        _features=tf.train.Features(feature=_feature)
        example = tf.train.Example(features=_features)
        writer.write(example.SerializeToString())
