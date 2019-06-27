
# https://learningai.io/projects/2017/06/29/tiny-imagenet.html
# not getting that great of acc

import argparse
import os
import sys

##############################################

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--alpha', type=float, default=5e-2)
parser.add_argument('--l2', type=float, default=0.)
parser.add_argument('--decay', type=float, default=1.)
parser.add_argument('--eps', type=float, default=1.)
parser.add_argument('--dropout', type=float, default=0.5)
parser.add_argument('--act', type=str, default='relu')
parser.add_argument('--bias', type=float, default=0.)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--dfa', type=int, default=1)
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
from lib.MaxPool import MaxPool
from lib.Dropout import Dropout
from lib.FeedbackFC import FeedbackFC
from lib.FeedbackConv import FeedbackConv
from lib.BatchNorm import BatchNorm

from lib.AvgPool import AvgPool
from lib.LELPool import LELPool

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
'''
A = tf.reshape(A, (self.batch, self.h * self.w, 1, self.fin))
A = tf.tile(A, [1, 1, self.ksize * self.ksize, 1])
A = tf.reshape(A, (self.batch, self.h, self.w, self.ksize, self.ksize, self.fin))
A = tf.reshape(A, (self.batch, self.h, self.w * self.ksize, self.ksize, self.fin))
A = tf.transpose(A, (0, 1, 3, 2, 4))
A = tf.reshape(A, (self.batch, self.h * self.ksize, self.w * self.ksize, self.fin))
'''
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

###############################################################

dropout_rate = tf.placeholder(tf.float32, shape=())
learning_rate = tf.placeholder(tf.float32, shape=())

gray = tf.image.rgb_to_grayscale(features)
# X = tf.map_fn(lambda frame: tf.image.per_image_standardization(frame), features)
X = features / 255.

l1_1 = Convolution(input_sizes=[batch_size, 64, 64, 3], filter_sizes=[3, 3, 3, 64], init=args.init, strides=[1, 1, 1, 1], padding="SAME", name="conv1")
l1_2 = BatchNorm(input_size=[args.batch_size, 64, 64, 64], name='conv1_bn')
l1_3 = Relu()
l1_4 = LELPool(input_shape=[args.batch_size, 64, 64, 64], pool_shape=[1, 8, 8, 1], num_classes=1000, name='conv1_fb')
l1_5 = Convolution(input_sizes=[batch_size, 64, 64, 64], filter_sizes=[3, 3, 64, 64], init=args.init, strides=[1, 1, 1, 1], padding="SAME", name="conv2")
l1_6 = BatchNorm(input_size=[args.batch_size, 64, 64, 64], name='conv2_bn')
l1_7 = Relu()
l1_8 = LELPool(input_shape=[args.batch_size, 64, 64, 64], pool_shape=[1, 8, 8, 1], num_classes=1000, name='conv2_fb')
l1_9 = AvgPool(size=[args.batch_size, 64, 64, 64], ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

l2_1 = Convolution(input_sizes=[batch_size, 32, 32, 64], filter_sizes=[3, 3, 64, 128], init=args.init, strides=[1, 1, 1, 1], padding="SAME", name="conv3")
l2_2 = BatchNorm(input_size=[args.batch_size, 32, 32, 128], name='conv3_bn')
l2_3 = Relu()
l2_4 = LELPool(input_shape=[args.batch_size, 32, 32, 128], pool_shape=[1, 4, 4, 1], num_classes=1000, name='conv3_fb')
l2_5 = Convolution(input_sizes=[batch_size, 32, 32, 128], filter_sizes=[3, 3, 128, 128], init=args.init, strides=[1, 1, 1, 1], padding="SAME", name="conv4")
l2_6 = BatchNorm(input_size=[args.batch_size, 32, 32, 128], name='conv4_bn')
l2_7 = Relu()
l2_8 = LELPool(input_shape=[args.batch_size, 32, 32, 128], pool_shape=[1, 4, 4, 1], num_classes=1000, name='conv4_fb')
l2_9 = AvgPool(size=[args.batch_size, 32, 32, 128], ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

l3_1 = Convolution(input_sizes=[batch_size, 16, 16, 128], filter_sizes=[3, 3, 128, 256], init=args.init, strides=[1, 1, 1, 1], padding="SAME", name="conv5")
l3_2 = BatchNorm(input_size=[args.batch_size, 16, 16, 256], name='conv5_bn')
l3_3 = Relu()
l3_4 = LELPool(input_shape=[args.batch_size, 16, 16, 256], pool_shape=[1, 4, 4, 1], num_classes=1000, name='conv5_fb')
l3_5 = Convolution(input_sizes=[batch_size, 16, 16, 256], filter_sizes=[3, 3, 256, 256], init=args.init, strides=[1, 1, 1, 1], padding="SAME", name="conv6")
l3_6 = BatchNorm(input_size=[args.batch_size, 16, 16, 256], name='conv6_bn')
l3_7 = Relu()
l3_8 = LELPool(input_shape=[args.batch_size, 16, 16, 256], pool_shape=[1, 4, 4, 1], num_classes=1000, name='conv6_fb')
l3_9 = AvgPool(size=[args.batch_size, 16, 16, 256], ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

l4_1 = Convolution(input_sizes=[batch_size, 8, 8, 256], filter_sizes=[3, 3, 256, 512], init=args.init, strides=[1, 1, 1, 1], padding="SAME", name="conv7")
l4_2 = BatchNorm(input_size=[args.batch_size, 8, 8, 512], name='conv7_bn')
l4_3 = Relu()
l4_4 = LELPool(input_shape=[args.batch_size, 8, 8, 512], pool_shape=[1, 2, 2, 1], num_classes=1000, name='conv7_fb')
l4_5 = Convolution(input_sizes=[batch_size, 8, 8, 512], filter_sizes=[3, 3, 512, 512], init=args.init, strides=[1, 1, 1, 1], padding="SAME", name="conv8")
l4_6 = BatchNorm(input_size=[args.batch_size, 8, 8, 512], name='conv8_bn')
l4_7 = Relu()
l4_8 = LELPool(input_shape=[args.batch_size, 8, 8, 512], pool_shape=[1, 2, 2, 1], num_classes=1000, name='conv8_fb')
l4_9 = AvgPool(size=[args.batch_size, 8, 8, 512], ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

l5 = ConvToFullyConnected(input_shape=[4, 4, 512])

l6_1 = FullyConnected(input_shape=4*4*512, size=4096, init=args.init, name="fc1")
# def need a batch norm here.
l6_2 = Relu()

l7 = Dropout(rate=dropout_rate)

l8 = FullyConnected(input_shape=4096, size=1000, init=args.init, name="fc2")

###############################################################

model = Model(layers=[                                                      \
                      l1_1, l1_2, l1_3, l1_4, l1_5, l1_6, l1_7, l1_8, l1_9, \
                      l2_1, l2_2, l2_3, l2_4, l2_5, l2_6, l2_7, l2_8, l2_9, \
                      l3_1, l3_2, l3_3, l3_4, l3_5, l3_6, l3_7, l3_8, l3_9, \
                      l4_1, l4_2, l4_3, l4_4, l4_5, l4_6, l4_7, l4_8, l4_9, \
                      l5,                                                   \
                      l6_1, l6_2,                                           \
                      l7,                                                   \
                      l8                                                    \
                      ])

predict = tf.nn.softmax(model.predict(X=X))
weights = model.get_weights()

o00 = model.up_to(X, N=0)
o09 = model.up_to(X, N=9)
o18 = model.up_to(X, N=18)
o27 = model.up_to(X, N=27)

if args.opt == "adam" or args.opt == "rms" or args.opt == "decay" or args.opt == "momentum":
    if args.dfa:
        grads_and_vars = model.lel_gvs(X=X, Y=labels)
    else:
        grads_and_vars = model.gvs(X=X, Y=labels)
        
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
        train = model.lel(X=X, Y=labels)
    else:
        train = model.train(X=X, Y=labels)

correct = tf.equal(tf.argmax(predict,1), tf.argmax(labels,1))
total_correct = tf.reduce_sum(tf.cast(correct, tf.float32))

top5 = in_top_k(predict, tf.argmax(labels,1), k=5)
total_top5 = tf.reduce_sum(tf.cast(top5, tf.float32))

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
    
    train_acc = 0.0
    train_acc_top5 = 0.0

    for jj in range(0, len(train_filenames), batch_size):
        
        if (jj % (100 * batch_size) == 0):
            [_total_correct, _total_top5, _, _gray, _o00, _o09, _o18, _o27] = sess.run([total_correct, total_top5, train, gray, o00, o09, o18, o27], feed_dict={handle: train_handle, dropout_rate: args.dropout, learning_rate: alpha})
            
            p = "train accuracy: %f %f" % (train_acc, train_acc_top5)
            print (p)
            f = open(results_filename, "a")
            f.write(p + "\n")
            f.close()

            ####################################
            '''
            rows = []
            for _ in range(5):
                idx = np.random.randint(low=0, high=64)

                imgray = scipy.misc.imresize(_gray[0, :, :, 0], 4.)
                im00 = scipy.misc.imresize(_o00[0, :, :, idx], 4.)
                im09 = scipy.misc.imresize(_o09[0, :, :, idx], 8.)
                im18 = scipy.misc.imresize(_o18[0, :, :, idx], 16.)
                im27 = scipy.misc.imresize(_o27[0, :, :, idx], 32.)

                row = np.concatenate((imgray, im00, im09, im18, im27), axis=1)
                rows.append(row)
                
            img = np.concatenate(rows, axis=0)
            plt.imsave('%d_%d_%d.jpg' % (args.dfa, jj, ii), img)
            '''
            ####################################

        else:
            [_total_correct, _total_top5, _] = sess.run([total_correct, total_top5, train], feed_dict={handle: train_handle, dropout_rate: args.dropout, learning_rate: alpha})

        train_total += batch_size
        train_correct += _total_correct
        train_top5 += _total_top5

        train_acc = train_correct / train_total
        train_acc_top5 = train_top5 / train_total

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
    
    for jj in range(0, len(val_filenames), batch_size):

        [_total_correct, _top5] = sess.run([total_correct, total_top5], feed_dict={handle: val_handle, dropout_rate: 0.0, learning_rate: 0.0})
        
        val_total += batch_size
        val_correct += _total_correct
        val_top5 += _top5
        
        val_acc = val_correct / val_total
        val_acc_top5 = val_top5 / val_total
        
        if (jj % (100 * batch_size) == 0):
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
        dacc = val_accs[-1] - val_accs[-2]
        if dacc <= 0.01:
            alpha = 0.1 * args.alpha
            phase = 2
    elif phase == 2:
        dacc = val_accs[-1] - val_accs[-2]
        if dacc <= 0.001:
            alpha = 0.01 * args.alpha
            phase = 3
    elif phase == 3:
        dacc = val_accs[-1] - val_accs[-2]
        if dacc <= 0.0001:
            alpha = 0.001 * args.alpha
            phase = 4
    elif phase == 4:
        dacc = val_accs[-1] - val_accs[-2]
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

