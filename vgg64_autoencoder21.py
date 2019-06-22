
# https://learningai.io/projects/2017/06/29/tiny-imagenet.html
# not getting that great of acc

import argparse
import os
import sys

##############################################

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--batch_size', type=int, default=64)
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
    
exxact = 0
if exxact:
    val_path = '/home/bcrafton3/Data_SSD/64x64/tfrecord/val/'
    train_path = '/home/bcrafton3/Data_SSD/64x64/tfrecord/train/'
else:
    val_path = '/usr/scratch/64x64/tfrecord/val/'
    train_path = '/usr/scratch/64x64/tfrecord/train/'

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
from lib.FullyConnectedToConv import FullyConnectedToConv
from lib.FullyConnected import FullyConnected
from lib.Convolution import Convolution
from lib.AvgPool import AvgPool
from lib.UpSample import UpSample
from lib.Dropout import Dropout
from lib.BatchNorm import BatchNorm

from lib.Activation import Activation
from lib.Activation import Relu
from lib.Activation import Linear

##############################################

num_classes = 1000
epochs = args.epochs
data_augmentation = False

# MEAN = [122.77093945, 116.74601272, 104.09373519]

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

    for subdir, dirs, files in os.walk(val_path):
        for file in files:
            val_filenames.append(os.path.join(val_path, file))

    # np.random.shuffle(val_filenames)    

    remainder = len(val_filenames) % args.batch_size
    val_filenames = val_filenames[:(-remainder)]

    return val_filenames
    
def get_train_filenames():
    train_filenames = []

    print ("building training dataset")

    for subdir, dirs, files in os.walk(train_path):
        for file in files:
            train_filenames.append(os.path.join(train_path, file))
    
    # np.random.shuffle(train_filenames)

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

    # means = tf.reshape(tf.constant(MEAN), [1, 1, 1, 3])
    # image = image - means
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

weights_conv = args.load
weights_fc = None

if weights_conv:
    train_conv = False
else:
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

# X = tf.map_fn(lambda frame: tf.image.per_image_standardization(frame), features)
# X = features / tf.reduce_max(features, axis=[3], keepdims=True)
X = features / 255.

l1_1 = Convolution(input_sizes=[args.batch_size, 64, 64, 3], filter_sizes=[3, 3, 3, 64], init=args.init, strides=[1,1,1,1], padding="SAME", name="conv1", load=weights_conv, train=train_conv)
l1_2 = BatchNorm(input_size=[args.batch_size, 64, 64, 64], name='conv1_bn')
l1_3 = Relu()
l1_4 = Convolution(input_sizes=[args.batch_size, 64, 64, 64], filter_sizes=[3, 3, 64, 64], init=args.init, strides=[1,1,1,1], padding="SAME", name="conv2", load=weights_conv, train=train_conv)
l1_5 = BatchNorm(input_size=[args.batch_size, 32, 32, 64], name='conv2_bn')
l1_6 = Relu()
l1_7 = AvgPool(size=[args.batch_size, 64, 64, 64], ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

l2_1 = Convolution(input_sizes=[args.batch_size, 32, 32, 64], filter_sizes=[3, 3, 64, 128], init=args.init, strides=[1,1,1,1], padding="SAME", name="conv3", load=weights_conv, train=train_conv)
l2_2 = BatchNorm(input_size=[args.batch_size, 32, 32, 128], name='conv3_bn')
l2_3 = Relu()
l2_4 = Convolution(input_sizes=[args.batch_size, 32, 32, 128], filter_sizes=[3, 3, 128, 128], init=args.init, strides=[1,1,1,1], padding="SAME", name="conv4", load=weights_conv, train=train_conv)
l2_5 = BatchNorm(input_size=[args.batch_size, 16, 16, 128], name='conv4_bn')
l2_6 = Relu()
l2_7 = AvgPool(size=[args.batch_size, 32, 32, 128], ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

l3_1 = Convolution(input_sizes=[args.batch_size, 16, 16, 128], filter_sizes=[3, 3, 128, 256], init=args.init, strides=[1,1,1,1], padding="SAME", name="conv5", load=weights_conv, train=train_conv)
l3_2 = BatchNorm(input_size=[args.batch_size, 16, 16, 256], name='conv5_bn')
l3_3 = Relu()
l3_4 = Convolution(input_sizes=[args.batch_size, 16, 16, 256], filter_sizes=[3, 3, 256, 256], init=args.init, strides=[1,1,1,1], padding="SAME", name="conv6", load=weights_conv, train=train_conv)
l3_5 = BatchNorm(input_size=[args.batch_size, 16, 16, 256], name='conv6_bn')
l3_6 = Relu()
l3_7 = AvgPool(size=[args.batch_size, 16, 16, 256], ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

l4_1 = Convolution(input_sizes=[args.batch_size, 8, 8, 256], filter_sizes=[3, 3, 256, 512], init=args.init, strides=[1,1,1,1], padding="SAME", name="conv7", load=weights_conv, train=train_conv)
l4_2 = BatchNorm(input_size=[args.batch_size, 8, 8, 512], name='conv7_bn')
l4_3 = Relu()
l4_4 = Convolution(input_sizes=[args.batch_size, 8, 8, 512], filter_sizes=[3, 3, 512, 512], init=args.init, strides=[1,1,1,1], padding="SAME", name="conv8", load=weights_conv, train=train_conv)
l4_5 = BatchNorm(input_size=[args.batch_size, 8, 8, 512], name='conv8_bn')
l4_6 = Relu()

l7_1 = Convolution(input_sizes=[args.batch_size, 8, 8, 512], filter_sizes=[3, 3, 512, 512], init=args.init, strides=[1,1,1,1], padding="SAME", name="conv9")
l7_2 = BatchNorm(input_size=[args.batch_size, 8, 8, 512], name='conv9_bn')
l7_3 = Convolution(input_sizes=[args.batch_size, 8, 8, 512], filter_sizes=[3, 3, 512, 256], init=args.init, strides=[1,1,1,1], padding="SAME", name="conv10")
l7_4 = BatchNorm(input_size=[args.batch_size, 8, 8, 256], name='conv10_bn')
l7_5 = UpSample(input_shape=[args.batch_size, 8, 8, 256], ksize=2)

l8_1 = Convolution(input_sizes=[args.batch_size, 16, 16, 256], filter_sizes=[3, 3, 256, 256], init=args.init, strides=[1,1,1,1], padding="SAME", name="conv11")
l8_2 = BatchNorm(input_size=[args.batch_size, 16, 16, 256], name='conv11_bn')
l8_3 = Convolution(input_sizes=[args.batch_size, 16, 16, 256], filter_sizes=[3, 3, 256, 128], init=args.init, strides=[1,1,1,1], padding="SAME", name="conv12")
l8_4 = BatchNorm(input_size=[args.batch_size, 16, 16, 128], name='conv12_bn')
l8_5 = UpSample(input_shape=[args.batch_size, 16, 16, 128], ksize=2)

l9_1 = Convolution(input_sizes=[args.batch_size, 32, 32, 128], filter_sizes=[3, 3, 128, 128], init=args.init, strides=[1,1,1,1], padding="SAME", name="conv13")
l9_2 = BatchNorm(input_size=[args.batch_size, 32, 32, 128], name='conv13_bn')
l9_3 = Convolution(input_sizes=[args.batch_size, 32, 32, 128], filter_sizes=[3, 3, 128, 64], init=args.init, strides=[1,1,1,1], padding="SAME", name="conv14")
l9_4 = BatchNorm(input_size=[args.batch_size, 32, 32, 64], name='conv14_bn')
l9_5 = UpSample(input_shape=[args.batch_size, 32, 32, 64], ksize=2)

l10_1 = Convolution(input_sizes=[args.batch_size, 64, 64, 64], filter_sizes=[3, 3, 64, 64], init=args.init, strides=[1,1,1,1], padding="SAME", name="conv15")
l10_2 = BatchNorm(input_size=[args.batch_size, 64, 64, 64], name='conv15_bn')
l10_3 = Convolution(input_sizes=[args.batch_size, 64, 64, 64], filter_sizes=[3, 3, 64, 3], init=args.init, strides=[1,1,1,1], padding="SAME", name="conv16")
l10_4 = BatchNorm(input_size=[args.batch_size, 64, 64, 3], name='conv16_bn')

###############################################################

layers=[                              
l1_1, l1_2, l1_3, l1_4, l1_5, l1_6, l1_7, 
l2_1, l2_2, l2_3, l2_4, l2_5, l2_6, l2_7, 
l3_1, l3_2, l3_3, l3_4, l3_5, l3_6, l3_7,
l4_1, l4_2, l4_3, l4_4, l4_5, l4_6,

l7_1, l7_2, l7_3, l7_4, l7_5,
l8_1, l8_2, l8_3, l8_4, l8_5,
l9_1, l9_2, l9_3, l9_4, l9_5,
l10_1, l10_2, l10_3, l10_4
]

model = Model(layers=layers, shape_y=[args.batch_size, 64, 64, 3])
predict = model.predict(X=X)

grads_and_vars, loss = model.gvs(X=X, Y=X)
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
    for jj in range(0, len(train_filenames), args.batch_size):
        
        if (jj % (args.batch_size * 100) == 0):
            [_, _loss, _X, _predict] = sess.run([train, loss, X, predict], feed_dict={handle: train_handle, dropout_rate: args.dropout, learning_rate: alpha})
            losses.append(_loss)

            ########################

            if args.load == None:
                ext = 'random'
            else:
                ext = args.load
            
            name = '%d_%s_%f.jpg' % (jj, ext, args.alpha)

            img1 = np.reshape(_X[0], (64, 64, 3))
            img1 = scipy.misc.imresize(img1, 4.)            
            
            img2 = np.reshape(_predict[0], (64, 64, 3))
            img2 = scipy.misc.imresize(img2, 4.)
            
            concat = np.concatenate((img1, img2), axis=1)
            plt.imsave(name, concat)
            
            ########################

            p = "%d: train loss: %f" % (ii, np.average(losses))
            print (p)
            f = open(results_filename, "a")
            f.write(p + "\n")
            f.close()

            losses=[]
        else:
            [_, _loss] = sess.run([train, loss], feed_dict={handle: train_handle, dropout_rate: args.dropout, learning_rate: alpha})
            losses.append(_loss)

    ##################################################################
    
    sess.run(val_iterator.initializer, feed_dict={filename: val_filenames})

    losses = []
    for jj in range(0, len(val_filenames), batch_size):
    
        if (jj % (args.batch_size * 100) == 0):
            [_loss, _X, _predict] = sess.run([loss, X, predict], feed_dict={handle: val_handle, dropout_rate: 0.0, learning_rate: 0.0})
            losses.append(_loss)
            
            p = "%d: val loss: %f" % (ii, np.average(losses))
            print (p)
            f = open(results_filename, "a")
            f.write(p + "\n")
            f.close()
            
            losses=[]
        else:
            [_loss] = sess.run([loss], feed_dict={handle: val_handle, dropout_rate: 0.0, learning_rate: 0.0})
            losses.append(_loss)

    ##################################################################


