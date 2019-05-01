
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

from lib.Model import Model

from lib.Layer import Layer 
from lib.ConvToFullyConnected import ConvToFullyConnected
from lib.FullyConnected import FullyConnected
from lib.Convolution import Convolution
from lib.Convolution3D import Convolution3D
from lib.MaxPool import MaxPool
from lib.Dropout import Dropout
from lib.FeedbackFC import FeedbackFC
from lib.FeedbackConv import FeedbackConv
from lib.BatchNorm import BatchNorm

from lib.Activation import Activation
from lib.Activation import Sigmoid
from lib.Activation import Relu
from lib.Activation import Tanh
from lib.Activation import Softmax
from lib.Activation import LeakyRelu
from lib.Activation import Linear

##############################################

batch_size = args.batch_size
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

    remainder = len(val_filenames) % batch_size
    val_filenames = val_filenames[:(-remainder)]

    return val_filenames
    
def get_train_filenames():
    train_filenames = []

    print ("building training dataset")

    for subdir, dirs, files in os.walk('/home/bcrafton3/Data_SSD/64x64/tfrecord/train/'):
        for file in files:
            train_filenames.append(os.path.join('/home/bcrafton3/Data_SSD/64x64/tfrecord/train/', file))
    
    np.random.shuffle(train_filenames)

    remainder = len(train_filenames) % batch_size
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

    label = sample['label']
    return [image, label]

###############################################################

train_filenames = get_train_filenames()
val_filenames = get_val_filenames()

filename = tf.placeholder(tf.string, shape=[None])

###############################################################

val_dataset = tf.data.TFRecordDataset(filename)
val_dataset = val_dataset.map(extract_fn, num_parallel_calls=4)
val_dataset = val_dataset.batch(batch_size)
val_dataset = val_dataset.repeat()
val_dataset = val_dataset.prefetch(8)

###############################################################

train_dataset = tf.data.TFRecordDataset(filename)
train_dataset = train_dataset.map(extract_fn, num_parallel_calls=4)
train_dataset = train_dataset.batch(batch_size)
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

l0 = Convolution3D(input_sizes=[batch_size, 64, 64, 3], filter_sizes=[3, 3, 3, 1, 32], init_filters=args.init, strides=[1, 1, 1, 1], padding="SAME", alpha=learning_rate, activation=Relu(), bias=args.bias, name="conv1", load=weights_conv, train=train_conv)

l11 = Convolution3D(input_sizes=[batch_size, 64, 64, 32], filter_sizes=[3, 3, 1, 32, 1], init_filters=args.init, strides=[1, 2, 2, 1], padding="SAME", alpha=learning_rate, activation=Relu(), bias=args.bias, name="conv2", load=weights_conv, train=train_conv)
l12 = BatchNorm(size=[32, 32, 32])
l13 = Convolution3D(input_sizes=[batch_size, 32, 32, 32], filter_sizes=[1, 1, 32, 1, 64], init_filters=args.init, strides=[1, 1, 1, 1], padding="SAME", alpha=learning_rate, activation=Relu(), bias=args.bias, name="conv3", load=weights_conv, train=train_conv)
l14 = BatchNorm(size=[32, 32, 64])

l21 = Convolution3D(input_sizes=[batch_size, 32, 32, 64], filter_sizes=[3, 3, 1, 64, 1], init_filters=args.init, strides=[1, 1, 1, 1], padding="SAME", alpha=learning_rate, activation=Relu(), bias=args.bias, name="conv2", load=weights_conv, train=train_conv)
l22 = BatchNorm(size=[32, 32, 64])
l23 = Convolution3D(input_sizes=[batch_size, 32, 32, 64], filter_sizes=[1, 1, 64, 1, 128], init_filters=args.init, strides=[1, 1, 1, 1], padding="SAME", alpha=learning_rate, activation=Relu(), bias=args.bias, name="conv3", load=weights_conv, train=train_conv)
l24 = BatchNorm(size=[32, 32, 128])

l31 = Convolution3D(input_sizes=[batch_size, 32, 32, 128], filter_sizes=[3, 3, 1, 128, 1], init_filters=args.init, strides=[1, 2, 2, 1], padding="SAME", alpha=learning_rate, activation=Relu(), bias=args.bias, name="conv2", load=weights_conv, train=train_conv)
l32 = BatchNorm(size=[16, 16, 128])
l33 = Convolution3D(input_sizes=[batch_size, 16, 16, 128], filter_sizes=[1, 1, 128, 1, 256], init_filters=args.init, strides=[1, 1, 1, 1], padding="SAME", alpha=learning_rate, activation=Relu(), bias=args.bias, name="conv3", load=weights_conv, train=train_conv)
l34 = BatchNorm(size=[16, 16, 256])

l41 = Convolution3D(input_sizes=[batch_size, 16, 16, 256], filter_sizes=[3, 3, 1, 256, 1], init_filters=args.init, strides=[1, 1, 1, 1], padding="SAME", alpha=learning_rate, activation=Relu(), bias=args.bias, name="conv2", load=weights_conv, train=train_conv)
l42 = BatchNorm(size=[16, 16, 256])
l43 = Convolution3D(input_sizes=[batch_size, 16, 16, 256], filter_sizes=[1, 1, 256, 1, 512], init_filters=args.init, strides=[1, 1, 1, 1], padding="SAME", alpha=learning_rate, activation=Relu(), bias=args.bias, name="conv3", load=weights_conv, train=train_conv)
l44 = BatchNorm(size=[16, 16, 512])

l51 = Convolution3D(input_sizes=[batch_size, 16, 16, 512], filter_sizes=[3, 3, 1, 512, 1], init_filters=args.init, strides=[1, 2, 2, 1], padding="SAME", alpha=learning_rate, activation=Relu(), bias=args.bias, name="conv2", load=weights_conv, train=train_conv)
l52 = BatchNorm(size=[8, 8, 512])
l53 = Convolution3D(input_sizes=[batch_size, 8, 8, 512], filter_sizes=[1, 1, 512, 1, 512], init_filters=args.init, strides=[1, 1, 1, 1], padding="SAME", alpha=learning_rate, activation=Relu(), bias=args.bias, name="conv3", load=weights_conv, train=train_conv)
l54 = BatchNorm(size=[8, 8, 512])

l61 = Convolution3D(input_sizes=[batch_size, 8, 8, 512], filter_sizes=[3, 3, 1, 512, 1], init_filters=args.init, strides=[1, 1, 1, 1], padding="SAME", alpha=learning_rate, activation=Relu(), bias=args.bias, name="conv2", load=weights_conv, train=train_conv)
l62 = BatchNorm(size=[8, 8, 512])
l63 = Convolution3D(input_sizes=[batch_size, 8, 8, 512], filter_sizes=[1, 1, 512, 1, 512], init_filters=args.init, strides=[1, 1, 1, 1], padding="SAME", alpha=learning_rate, activation=Relu(), bias=args.bias, name="conv3", load=weights_conv, train=train_conv)
l64 = BatchNorm(size=[8, 8, 512])

l71 = Convolution3D(input_sizes=[batch_size, 8, 8, 512], filter_sizes=[3, 3, 1, 512, 1], init_filters=args.init, strides=[1, 1, 1, 1], padding="SAME", alpha=learning_rate, activation=Relu(), bias=args.bias, name="conv2", load=weights_conv, train=train_conv)
l72 = BatchNorm(size=[8, 8, 512])
l73 = Convolution3D(input_sizes=[batch_size, 8, 8, 512], filter_sizes=[1, 1, 512, 1, 512], init_filters=args.init, strides=[1, 1, 1, 1], padding="SAME", alpha=learning_rate, activation=Relu(), bias=args.bias, name="conv3", load=weights_conv, train=train_conv)
l74 = BatchNorm(size=[8, 8, 512])

l81 = Convolution3D(input_sizes=[batch_size, 8, 8, 512], filter_sizes=[3, 3, 1, 512, 1], init_filters=args.init, strides=[1, 2, 2, 1], padding="SAME", alpha=learning_rate, activation=Relu(), bias=args.bias, name="conv2", load=weights_conv, train=train_conv)
l82 = BatchNorm(size=[4, 4, 512])
l83 = Convolution3D(input_sizes=[batch_size, 4, 4, 512], filter_sizes=[1, 1, 512, 1, 512], init_filters=args.init, strides=[1, 1, 1, 1], padding="SAME", alpha=learning_rate, activation=Relu(), bias=args.bias, name="conv3", load=weights_conv, train=train_conv)
l84 = BatchNorm(size=[4, 4, 512])

l9 = ConvToFullyConnected(shape=[4, 4, 512])

l10 = FullyConnected(size=[4*4*512, 2048], num_classes=num_classes, init_weights=args.init, alpha=learning_rate, activation=Relu(), bias=1.0, last_layer=False, name="fc1", load=weights_fc, train=train_fc)

l11 = Dropout(rate=dropout_rate)

l12 = FullyConnected(size=[2048, num_classes], num_classes=num_classes, init_weights=args.init, alpha=learning_rate, activation=Linear(), bias=1.0, last_layer=True, name="fc2", load=weights_fc, train=train_fc)

###############################################################

model = Model(layers=[l0,                 \
                      l11, l12, l13, l14, \
                      l21, l22, l23, l24, \
                      l31, l32, l33, l34, \
                      l41, l42, l43, l44, \
                      l51, l52, l53, l54, \
                      l61, l62, l63, l64, \
                      l71, l72, l73, l74, \
                      l81, l82, l83, l84, \
                      l9,                 \
                      l10,                \
                      l11,                \
                      l12])

predict = tf.nn.softmax(model.predict(X=features))

if args.opt == "adam" or args.opt == "rms" or args.opt == "decay" or args.opt == "momentum":
    if args.dfa:
        grads_and_vars = model.dfa_gvs(X=features, Y=labels)
    else:
        grads_and_vars = model.gvs(X=features, Y=labels)
        
    if args.opt == "adam":
        train = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.9, beta2=0.999, epsilon=args.eps).apply_gradients(grads_and_vars=grads_and_vars)
    elif args.opt == "rms":
        train = tf.train.RMSPropOptimizer(learning_rate=learning_rate, decay=0.99, epsilon=args.eps).apply_gradients(grads_and_vars=grads_and_vars)
    elif args.opt == "decay":
        train = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).apply_gradients(grads_and_vars=grads_and_vars)
    elif args.opt == "momentum":
        train = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9).apply_gradients(grads_and_vars=grads_and_vars)
    else:
        assert(False)

else:
    if args.dfa:
        train = model.dfa(X=features, Y=labels)
    else:
        train = model.train(X=features, Y=labels)

correct = tf.equal(tf.argmax(predict,1), tf.argmax(labels,1))
total_correct = tf.reduce_sum(tf.cast(correct, tf.float32))

top5 = in_top_k(predict, tf.argmax(labels,1), k=5)
total_top5 = tf.reduce_sum(tf.cast(top5, tf.float32))

weights = model.get_weights()

print (model.num_params())

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

train_accs = []
train_accs_top5 = []

val_accs = []
val_accs_top5 = []

alpha = args.alpha
phase = 0

for ii in range(0, epochs):

    print('epoch {}/{}'.format(ii, epochs))
    
    ##################################################################

    sess.run(train_iterator.initializer, feed_dict={filename: train_filenames})

    train_total = 0.0
    train_correct = 0.0
    train_top5 = 0.0
    
    for j in range(0, len(train_filenames), batch_size):
        print (j)
        
        [_total_correct, _total_top5, _] = sess.run([total_correct, total_top5, train], feed_dict={handle: train_handle, dropout_rate: args.dropout, learning_rate: alpha})

        train_total += batch_size
        train_correct += _total_correct
        train_top5 += _total_top5
        
        train_acc = train_correct / train_total
        train_acc_top5 = train_top5 / train_total
        
        if (j % (100 * batch_size) == 0):
            p = "train accuracy: %f %f" % (train_acc, train_acc_top5)
            print (p)
            f = open(results_filename, "a")
            f.write(p + "\n")
            f.close()

    p = "train accuracy: %f %f" % (train_acc, train_acc_top5)
    print (p)
    f = open(results_filename, "a")
    f.write(p + "\n")
    f.close()

    train_accs.append(train_acc)
    train_accs_top5.append(train_acc_top5)
    
    ##################################################################

    sess.run(val_iterator.initializer, feed_dict={filename: val_filenames})
    
    val_total = 0.0
    val_correct = 0.0
    val_top5 = 0.0
    
    for j in range(0, len(val_filenames), batch_size):
        print (j)

        [_total_correct, _top5] = sess.run([total_correct, total_top5], feed_dict={handle: val_handle, dropout_rate: 0.0, learning_rate: 0.0})
        
        val_total += batch_size
        val_correct += _total_correct
        val_top5 += _top5
        
        val_acc = val_correct / val_total
        val_acc_top5 = val_top5 / val_total
        
        if (j % (100 * batch_size) == 0):
            p = "val accuracy: %f %f" % (val_acc, val_acc_top5)
            print (p)
            f = open(results_filename, "a")
            f.write(p + "\n")
            f.close()

    p = "val accuracy: %f %f" % (val_acc, val_acc_top5)
    print (p)
    f = open(results_filename, "a")
    f.write(p + "\n")
    f.close()

    val_accs.append(val_acc)
    val_accs_top5.append(val_acc_top5)

    if phase == 0:
        phase = 1
    elif phase == 1:
        dacc = train_accs[-1] - train_accs[-2]
        if dacc <= 0.01:
            alpha = 0.1 * args.alpha
            phase = 2
    elif phase == 2:
        dacc = train_accs[-1] - train_accs[-2]
        if dacc <= 0.001:
            alpha = 0.01 * args.alpha
            phase = 3
    elif phase == 3:
        dacc = train_accs[-1] - train_accs[-2]
        if dacc <= 0.0001:
            alpha = 0.001 * args.alpha
            phase = 4
    elif phase == 4:
        dacc = train_accs[-1] - train_accs[-2]
        if dacc <= 0.00001:
            alpha = 0.0001 * args.alpha
            phase = 5

    p = "Phase: %d" % (phase)
    print (p)
    f = open(results_filename, "a")
    f.write(p + "\n")
    f.close()

    if args.save:
        [w] = sess.run([weights], feed_dict={handle: val_handle, dropout_rate: 0.0, learning_rate: 0.0})
        w['train_acc'] = train_accs
        w['train_acc_top5'] = train_accs_top5
        w['val_acc'] = val_accs
        w['val_acc_top5'] = val_accs_top5
        np.save(args.name, w)
