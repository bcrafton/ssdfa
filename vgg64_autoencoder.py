
# https://learningai.io/projects/2017/06/29/tiny-imagenet.html
# not getting that great of acc

import argparse
import os
import sys

##############################################

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--alpha', type=float, default=1e-2)
parser.add_argument('--l2', type=float, default=0.)
parser.add_argument('--decay', type=float, default=1.)
parser.add_argument('--eps', type=float, default=1.)
parser.add_argument('--dropout', type=float, default=0.5)
parser.add_argument('--act', type=str, default='relu')
parser.add_argument('--bias', type=float, default=0.)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--dfa', type=int, default=0)
parser.add_argument('--sparse', type=int, default=0)
parser.add_argument('--rank', type=int, default=0)
parser.add_argument('--init', type=str, default="alexnet")
parser.add_argument('--opt', type=str, default="adam")
parser.add_argument('--save', type=int, default=0)
parser.add_argument('--name', type=str, default="vgg64x64")
parser.add_argument('--load', type=str, default=None)
args = parser.parse_args()

if args.gpu >= 0:
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpu)

##############################################

import keras
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import SGD

import tensorflow as tf
import os
import math
import numpy
import numpy as np
np.set_printoptions(threshold=1000)
import time
import re

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from PIL import Image
import scipy.misc

from lib.ModelMSE import Model

from lib.Layer import Layer 
from lib.ConvToFullyConnected import ConvToFullyConnected
from lib.FullyConnected import FullyConnected
from lib.Convolution import Convolution
from lib.AvgPool import AvgPool
from lib.UpSample import UpSample
from lib.Dropout import Dropout
from lib.FeedbackFC import FeedbackFC
from lib.FeedbackConv import FeedbackConv

from lib.Activation import Activation
from lib.Activation import Sigmoid
from lib.Activation import Relu
from lib.Activation import Tanh
from lib.Activation import Softmax
from lib.Activation import LeakyRelu
from lib.Activation import Linear

##############################################

num_classes = 1000
epochs = args.epochs
data_augmentation = False

MEAN = [122.77093945, 116.74601272, 104.09373519]

##############################################

def in_top_k(x, y, k):
    x = tf.cast(x, dtype=tf.float32)
    y = tf.cast(y, dtype=tf.int32)

    _, topk = tf.nn.top_k(input=x, k=k)
    topk = tf.transpose(topk)
    correct = tf.equal(y, topk)
    correct = tf.cast(correct, dtype=tf.int32)
    correct = tf.reduce_sum(correct, axis=0)
    return correct

##############################################

def parse_function(filename, label):
    conv = tf.read_file(filename)
    return conv, label

##############################################

def preprocess(image):

    means = tf.reshape(tf.constant(MEAN), [1, 1, 3])
    image = image - means

    return image

##############################################

def pre(fn):
    [fn] = re.findall('\d+.tfrecord', fn)
    [fn] = re.findall('\d+', fn)
    return int(fn)

def get_val_filenames():
    val_filenames = []

    print ("building validation dataset")

    for subdir, dirs, files in os.walk('/home/bcrafton3/Data_SSD/64x64/tfrecord/val/'):
        for file in files:
            val_filenames.append(os.path.join('/home/bcrafton3/Data_SSD/64x64/tfrecord/val/', file))

    np.random.shuffle(val_filenames)    

    remainder = len(val_filenames) % args.batch_size
    val_filenames = val_filenames[:(-remainder)]

    return val_filenames
    
def get_train_filenames():
    train_filenames = []

    print ("building training dataset")

    for subdir, dirs, files in os.walk('/home/bcrafton3/Data_SSD/64x64/tfrecord/train/'):
        for file in files:
            train_filenames.append(os.path.join('/home/bcrafton3/Data_SSD/64x64/tfrecord/train/', file))
    
    np.random.shuffle(train_filenames)

    remainder = len(train_filenames) % args.batch_size
    train_filenames = train_filenames[:(-remainder)]

    return train_filenames

def extract_fn(record):
    _feature={
        'label': tf.FixedLenFeature([], tf.int64),
        'image_raw': tf.FixedLenFeature([], tf.string)
    }
    sample = tf.parse_single_example(record, _feature)
    image = tf.decode_raw(sample['image_raw'], tf.uint8)
    # this was tricky ... stored as uint8, not float32.
    image = tf.cast(image, dtype=tf.float32)
    image = tf.reshape(image, (1, 64, 64, 3))

    means = tf.reshape(tf.constant(MEAN), [1, 1, 1, 3])
    image = image - means
    # image = image / 255.

    label = sample['label']
    return [image, label]

###############################################################

train_filenames = get_train_filenames()
val_filenames = get_val_filenames()

filename = tf.placeholder(tf.string, shape=[None])

###############################################################

val_dataset = tf.data.TFRecordDataset(filename)
val_dataset = val_dataset.map(extract_fn, num_parallel_calls=4)
val_dataset = val_dataset.batch(args.batch_size)
val_dataset = val_dataset.repeat()
val_dataset = val_dataset.prefetch(8)

###############################################################

train_dataset = tf.data.TFRecordDataset(filename)
train_dataset = train_dataset.map(extract_fn, num_parallel_calls=4)
train_dataset = train_dataset.batch(args.batch_size)
train_dataset = train_dataset.repeat()
train_dataset = train_dataset.prefetch(8)

###############################################################

handle = tf.placeholder(tf.string, shape=[])
iterator = tf.data.Iterator.from_string_handle(handle, train_dataset.output_types, train_dataset.output_shapes)
features, labels = iterator.get_next()

features = tf.reshape(features, (args.batch_size, 64, 64, 3))
labels = tf.one_hot(labels, depth=num_classes)

train_iterator = train_dataset.make_initializable_iterator()
val_iterator = val_dataset.make_initializable_iterator()

###############################################################

weights_conv = None
weights_fc = None

train_conv = True
train_fc = True

if args.act == 'tanh':
    act = Tanh()
elif args.act == 'relu':
    act = Relu()
else:
    assert(False)

###############################################################

dropout_rate = tf.placeholder(tf.float32, shape=())
learning_rate = tf.placeholder(tf.float32, shape=())

####

l1_1 = Convolution(input_sizes=[args.batch_size, 64, 64, 3], filter_sizes=[3, 3, 3, 64], init=args.init, strides=[1,1,1,1], padding="SAME", name="conv1")
l1_2 = Relu()
l1_3 = Convolution(input_sizes=[args.batch_size, 64, 64, 64], filter_sizes=[3, 3, 64, 64], init=args.init, strides=[1,1,1,1], padding="SAME", name="conv2")
l1_4 = Relu()
l1_5 = AvgPool(size=[args.batch_size, 64, 64, 64], ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

l2_1 = Convolution(input_sizes=[args.batch_size, 32, 32, 64], filter_sizes=[3, 3, 64, 128], init=args.init, strides=[1,1,1,1], padding="SAME", name="conv3")
l2_2 = Relu()
l2_3 = Convolution(input_sizes=[args.batch_size, 32, 32, 128], filter_sizes=[3, 3, 128, 128], init=args.init, strides=[1,1,1,1], padding="SAME", name="conv4")
l2_4 = Relu()
l2_5 = AvgPool(size=[args.batch_size, 32, 32, 128], ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

l3_1 = Convolution(input_sizes=[args.batch_size, 16, 16, 128], filter_sizes=[3, 3, 128, 256], init=args.init, strides=[1,1,1,1], padding="SAME", name="conv5")
l3_2 = Relu()
l3_3 = Convolution(input_sizes=[args.batch_size, 16, 16, 256], filter_sizes=[3, 3, 256, 256], init=args.init, strides=[1,1,1,1], padding="SAME", name="conv6")
l3_4 = Relu()
l3_5 = AvgPool(size=[args.batch_size, 16, 16, 256], ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

#### 

l4_1 = Convolution(input_sizes=[args.batch_size, 8, 8, 256], filter_sizes=[3, 3, 256, 256], init=args.init, strides=[1,1,1,1], padding="SAME", name="conv7")
l4_2 = Convolution(input_sizes=[args.batch_size, 8, 8, 256], filter_sizes=[3, 3, 256, 128], init=args.init, strides=[1,1,1,1], padding="SAME", name="conv8")
l4_3 = UpSample(input_shape=[args.batch_size, 8, 8, 128], ksize=2)

l5_1 = Convolution(input_sizes=[args.batch_size, 16, 16, 128], filter_sizes=[3, 3, 128, 128], init=args.init, strides=[1,1,1,1], padding="SAME", name="conv9")
l5_2 = Convolution(input_sizes=[args.batch_size, 16, 16, 128], filter_sizes=[3, 3, 128, 64], init=args.init, strides=[1,1,1,1], padding="SAME", name="conv10")
l5_3 = UpSample(input_shape=[args.batch_size, 16, 16, 64], ksize=2)

l6_1 = Convolution(input_sizes=[args.batch_size, 32, 32, 64], filter_sizes=[3, 3, 64, 64], init=args.init, strides=[1,1,1,1], padding="SAME", name="conv11")
l6_2 = Convolution(input_sizes=[args.batch_size, 32, 32, 64], filter_sizes=[3, 3, 64, 3], init=args.init, strides=[1,1,1,1], padding="SAME", name="conv12")
l6_3 = UpSample(input_shape=[args.batch_size, 32, 32, 3], ksize=2)

###############################################################

model = Model(layers=[                              \
                      l1_1, l1_2, l1_3, l1_4, l1_5, \
                      l2_1, l2_2, l2_3, l2_4, l2_5, \
                      l3_1, l3_2, l3_3, l3_4, l3_5, \
                      l4_1, l4_2, l4_3,             \
                      l5_1, l5_1, l5_3,             \
                      l6_1, l6_2, l6_3              \
                      ])

grads_and_vars, loss = model.gvs(X=features, Y=features)
train = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.9, beta2=0.999, epsilon=args.eps).apply_gradients(grads_and_vars=grads_and_vars)

###############################################################

config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.allow_growth=True

sess = tf.Session(config=config)
sess.run(tf.global_variables_initializer())

train_handle = sess.run(train_iterator.string_handle())
val_handle = sess.run(val_iterator.string_handle())

###############################################################

results_filename = args.name + '.results'
f = open(results_filename, "w")
f.write(results_filename + "\n")
f.write("total params: " + str(model.num_params()) + "\n")
f.close()

###############################################################

alpha = args.alpha

for ii in range(0, epochs):

    print('epoch {}/{}'.format(ii, epochs))

    ##################################################################

    sess.run(train_iterator.initializer, feed_dict={filename: train_filenames})
    
    losses = []
    for j in range(0, len(train_filenames), args.batch_size):

        [_, _loss] = sess.run([train, loss], feed_dict={handle: train_handle, dropout_rate: args.dropout, learning_rate: alpha})
        losses.append(_loss)
        
        if (j % (args.batch_size * 100) == 0):
            print (np.average(losses))

    ##################################################################


