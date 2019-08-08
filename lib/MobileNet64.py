
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
parser.add_argument('--ae_loss', type=float, default=0.0)

parser.add_argument('--save', type=int, default=0)
parser.add_argument('--name', type=str, default="mobile64")
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
from lib.Convolution2D import Convolution2D
from lib.Convolution3D import Convolution3D
from lib.ConvolutionDW import ConvolutionDW
from lib.MaxPool import MaxPool
from lib.AvgPool import AvgPool
from lib.Dropout import Dropout
from lib.FeedbackFC import FeedbackFC
from lib.FeedbackConv import FeedbackConv
from lib.BatchNorm import BatchNorm
from lib.ConvBlock import ConvBlock
from lib.MobileBlock import MobileBlock
from lib.LELConv import LELConv

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
train_conv = True

weights_conv_dw = None
train_conv_dw = True

weights_conv_1x1 = None
train_conv_1x1 = True

weights_fc = None
train_fc = True

###############################################################

if args.act == 'tanh':
    act = Tanh()
elif args.act == 'relu':
    act = Relu()
else:
    assert(False)

###############################################################

dropout_rate = tf.placeholder(tf.float32, shape=())
learning_rate = tf.placeholder(tf.float32, shape=())

gray = tf.image.rgb_to_grayscale(features)
# X = features / 255.

l1_1 = ConvBlock(input_shape=[batch_size, 64, 64, 3], filter_shape=[3, 3, 3, 32], strides=[1,1,1,1], init=args.init, name='block1')
l1_2 = LELConv(input_shape=[batch_size, 64, 64, 32], pool_shape=[1,8,8,1], num_classes=1000, name='block1_fb')

l2 = MobileBlock(input_shape=[batch_size, 64, 64, 32],  filter_shape=[32, 64],   strides=[1,2,2,1], init=args.init, pool_shape=[1,8,8,1], num_classes=1000, name='block2')
l3 = MobileBlock(input_shape=[batch_size, 32, 32, 64],  filter_shape=[64, 128],  strides=[1,1,1,1], init=args.init, pool_shape=[1,8,8,1], num_classes=1000, name='block3')
l4 = MobileBlock(input_shape=[batch_size, 32, 32, 128], filter_shape=[128, 256], strides=[1,2,2,1], init=args.init, pool_shape=[1,4,4,1], num_classes=1000, name='block4')
l5 = MobileBlock(input_shape=[batch_size, 16, 16, 256], filter_shape=[256, 512], strides=[1,1,1,1], init=args.init, pool_shape=[1,4,4,1], num_classes=1000, name='block5')
l6 = MobileBlock(input_shape=[batch_size, 16, 16, 512], filter_shape=[512, 512], strides=[1,2,2,1], init=args.init, pool_shape=[1,2,2,1], num_classes=1000, name='block6')

l7 = MobileBlock(input_shape=[batch_size, 8, 8, 512], filter_shape=[512, 512], strides=[1,1,1,1], init=args.init, pool_shape=[1,2,2,1], num_classes=1000, name='block7')
l8 = MobileBlock(input_shape=[batch_size, 8, 8, 512], filter_shape=[512, 512], strides=[1,1,1,1], init=args.init, pool_shape=[1,2,2,1], num_classes=1000, name='block8')
l9 = MobileBlock(input_shape=[batch_size, 8, 8, 512], filter_shape=[512, 512], strides=[1,1,1,1], init=args.init, pool_shape=[1,2,2,1], num_classes=1000, name='block9')

l10 = MobileBlock(input_shape=[batch_size, 8, 8, 512],  filter_shape=[512, 1024],  strides=[1,2,2,1], init=args.init, pool_shape=[1,2,2,1], num_classes=1000, name='block10')
l11 = MobileBlock(input_shape=[batch_size, 4, 4, 1024], filter_shape=[1024, 1024], strides=[1,1,1,1], init=args.init, pool_shape=[1,4,4,1], num_classes=1000, name='block11')

l12 = AvgPool(size=[batch_size, 4, 4, 1024], ksize=[1, 4, 4, 1], strides=[1, 4, 4, 1], padding="SAME")
l13 = ConvToFullyConnected(input_shape=[1, 1, 1024])
l14 = FullyConnected(input_shape=1024, size=1000, init=args.init, name="fc1")

###############################################################

model = Model(layers=[l1_1, l1_2, l2, l3, l4, l5, l6, l7, l8, l9, l10, l11, l12, l13, l14])

predict = tf.nn.softmax(model.predict(X=features))

o0 = model.up_to(features, N=0)
o1 = model.up_to(features, N=2)
o2 = model.up_to(features, N=4)
o3 = model.up_to(features, N=6)

if args.opt == "adam" or args.opt == "rms" or args.opt == "decay" or args.opt == "momentum":
    if args.dfa:
        grads_and_vars = model.lel_gvs(X=features, Y=labels)
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
        train = model.lel(X=features, Y=labels)
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

    # [w] = sess.run([weights], feed_dict={handle: val_handle, dropout_rate: 0.0, learning_rate: 0.0})
    # np.save(args.name, w)
    # print (w.keys())

    print('epoch {}/{}'.format(ii, epochs))
    
    ##################################################################

    sess.run(train_iterator.initializer, feed_dict={filename: train_filenames})

    train_total = 0.0
    train_correct = 0.0
    train_top5 = 0.0

    # train_acc = train_correct / train_total
    # train_acc_top5 = train_top5 / train_total

    train_acc = 0.0
    train_acc_top5 = 0.0

    for jj in range(0, len(train_filenames), batch_size):
        # print (jj)
        
        if (jj % (100 * batch_size) == 0):
            [_total_correct, _total_top5, _, _gray, _o0, _o1, _o2, _o3] = sess.run([total_correct, total_top5, train, gray, o0, o1, o2, o3], feed_dict={handle: train_handle, dropout_rate: args.dropout, learning_rate: alpha})
            
            p = "train accuracy: %f %f" % (train_acc, train_acc_top5)
            print (p)
            f = open(results_filename, "a")
            f.write(p + "\n")
            f.close()

            ####################################

            rows = []
            for _ in range(5):
                idx = np.random.randint(low=0, high=32)

                imgray = scipy.misc.imresize(_gray[0, :, :, 0], 4.)
                im0 = scipy.misc.imresize(_o0[0, :, :, idx], 4.)
                im1 = scipy.misc.imresize(_o1[0, :, :, idx], 8.)
                im2 = scipy.misc.imresize(_o2[0, :, :, idx], 16.)
                im3 = scipy.misc.imresize(_o3[0, :, :, idx], 32.)

                row = np.concatenate((imgray, im0, im1, im2, im3), axis=1)
                rows.append(row)
                
            img = np.concatenate(rows, axis=0)
            plt.imsave('%d_%d.jpg' % (args.dfa, jj), img)

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
        # print (jj)

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

