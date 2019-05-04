
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
parser.add_argument('--act', type=str, default='tanh')
parser.add_argument('--bias', type=float, default=0.)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--dfa', type=int, default=0)
parser.add_argument('--sparse', type=int, default=0)
parser.add_argument('--rank', type=int, default=0)
parser.add_argument('--init', type=str, default="alexnet")
parser.add_argument('--opt', type=str, default="adam")
parser.add_argument('--save', type=int, default=0)
parser.add_argument('--name', type=str, default="imagenet_alexnet")
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

from lib.Activation import Activation
from lib.Activation import Sigmoid
from lib.Activation import Relu
from lib.Activation import Tanh
from lib.Activation import Softmax
from lib.Activation import LeakyRelu
from lib.Activation import Linear

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

batch_size = args.batch_size
num_classes = 1000
epochs = args.epochs
data_augmentation = False

EPOCHS = args.epochs
BATCH_SIZE = args.batch_size

IMAGENET_MEAN = [123.68, 116.78, 103.94]

##############################################

# https://gist.githubusercontent.com/omoindrot/dedc857cdc0e680dfb1be99762990c9c/raw/e560edd240f8b97e1f0483843dc4d64729ce025c/tensorflow_finetune.py

# Preprocessing (for both training and validation):
# (1) Decode the image from jpg format
# (2) Resize the image so its smaller side is 256 pixels long
def parse_function(filename, label):
    image_string = tf.read_file(filename)
    image_decoded = tf.image.decode_jpeg(image_string, channels=3)          # (1)
    image = tf.cast(image_decoded, tf.float32)

    smallest_side = 256.0
    height, width = tf.shape(image)[0], tf.shape(image)[1]
    height = tf.to_float(height)
    width = tf.to_float(width)

    scale = tf.cond(tf.greater(height, width),
                    lambda: smallest_side / width,
                    lambda: smallest_side / height)
    new_height = tf.to_int32(height * scale)
    new_width = tf.to_int32(width * scale)

    resized_image = tf.image.resize_images(image, [new_height, new_width])  # (2)
    return resized_image, label

# Preprocessing (for training)
# (3) Take a random 224x224 crop to the scaled image
# (4) Horizontally flip the image with probability 1/2
# (5) Substract the per color mean `IMAGENET_MEAN`
# Note: we don't normalize the data here, as VGG was trained without normalization
def train_preprocess(image, label):
    crop_image = tf.random_crop(image, [224, 224, 3])                       # (3)
    flip_image = tf.image.random_flip_left_right(crop_image)                # (4)

    means = tf.reshape(tf.constant(IMAGENET_MEAN), [1, 1, 3])
    centered_image = flip_image - means                                     # (5)

    return centered_image, label
    

# Preprocessing (for validation)
# (3) Take a central 224x224 crop to the scaled image
# (4) Substract the per color mean `IMAGENET_MEAN`
# Note: we don't normalize the data here, as VGG was trained without normalization
def val_preprocess(image, label):
    crop_image = tf.image.resize_image_with_crop_or_pad(image, 224, 224)    # (3)

    means = tf.reshape(tf.constant(IMAGENET_MEAN), [1, 1, 3])
    centered_image = crop_image - means                                     # (4)

    return centered_image, label

##############################################

def get_validation_dataset():
    label_counter = 0
    validation_images = []
    validation_labels = []

    print ("building validation dataset")

    for subdir, dirs, files in os.walk('/home/bcrafton3/Data_SSD/ILSVRC2012/val/'):
        for file in files:
            validation_images.append(os.path.join('/home/bcrafton3/Data_SSD/ILSVRC2012/val/', file))
    validation_images = sorted(validation_images)

    validation_labels_file = open('/home/bcrafton3/dfa/imagenet_labels/validation_labels.txt')
    lines = validation_labels_file.readlines()
    for ii in range(len(lines)):
        validation_labels.append(int(lines[ii]))

    print (len(validation_images), len(validation_labels))
    remainder = len(validation_labels) % batch_size
    validation_images = validation_images[:(-remainder)]
    validation_labels = validation_labels[:(-remainder)]

    print("validation data is ready...")

    return validation_images, validation_labels
    
def get_train_dataset():

    label_counter = 0
    training_images = []
    training_labels = []

    print ("making labels dict")

    f = open('/home/bcrafton3/dfa/imagenet_labels/train_labels.txt', 'r')
    lines = f.readlines()

    labels = {}
    for line in lines:
        line = line.split(' ')
        labels[line[0]] = label_counter
        label_counter += 1

    f.close()

    print ("building dataset")

    for subdir, dirs, files in os.walk('/home/bcrafton3/Data_SSD/ILSVRC2012/train/'):
        for folder in dirs:
            for folder_subdir, folder_dirs, folder_files in os.walk(os.path.join(subdir, folder)):
                for file in folder_files:
                    training_images.append(os.path.join(folder_subdir, file))
                    training_labels.append(labels[folder])

    remainder = len(training_labels) % batch_size
    training_images = training_images[:(-remainder)]
    training_labels = training_labels[:(-remainder)]

    print("Data is ready...")

    return training_images, training_labels

###############################################################

filename = tf.placeholder(tf.string, shape=[None])
label = tf.placeholder(tf.int64, shape=[None])

###############################################################

val_imgs, val_labs = get_validation_dataset()

val_dataset = tf.data.Dataset.from_tensor_slices((filename, label))
val_dataset = val_dataset.shuffle(len(val_imgs))
# val_dataset = val_dataset.shuffle(len(val_imgs), reshuffle_each_iteration=False)
val_dataset = val_dataset.map(parse_function, num_parallel_calls=4)
val_dataset = val_dataset.map(val_preprocess, num_parallel_calls=4)
val_dataset = val_dataset.batch(batch_size)
val_dataset = val_dataset.repeat()
val_dataset = val_dataset.prefetch(8)

###############################################################

train_imgs, train_labs = get_train_dataset()

train_dataset = tf.data.Dataset.from_tensor_slices((filename, label))
train_dataset = train_dataset.shuffle(len(train_imgs))
train_dataset = train_dataset.shuffle(len(train_imgs), reshuffle_each_iteration=False)
train_dataset = train_dataset.map(parse_function, num_parallel_calls=4)
train_dataset = train_dataset.map(train_preprocess, num_parallel_calls=4)
train_dataset = train_dataset.batch(batch_size)
train_dataset = train_dataset.repeat()
train_dataset = train_dataset.prefetch(8)

###############################################################

handle = tf.placeholder(tf.string, shape=[])
iterator = tf.data.Iterator.from_string_handle(handle, train_dataset.output_types, train_dataset.output_shapes)
features, labels = iterator.get_next()
features = tf.reshape(features, (-1, 224, 224, 3))
labels = tf.one_hot(labels, depth=num_classes)

train_iterator = train_dataset.make_initializable_iterator()
val_iterator = val_dataset.make_initializable_iterator()

###############################################################

train_conv = False
weights_conv = 'MobileNetWeights.npy'

train_conv_dw = False
weights_conv_dw = 'MobileNetWeights.npy'

train_conv_pw = False
weights_conv_pw = 'MobileNetWeights.npy'

train_fc = True
weights_fc = None

if args.act == 'tanh':
    act = Tanh()
elif args.act == 'relu':
    act = Relu()
else:
    assert(False)

###############################################################
'''
# can make this tighter bc all filters are 3x3, 1x1 ... only need to know like 3 things to make this work.
def chunk(input_size_dw, filter_size_dw, filter_size_1x1, strides_dw, num):
    name_dw = 'conv' + str(num)
    name_1x1 = 'conv' + str(num + 1)

    batch, h_dw, w_dw, c_dw = input_size_dw
    fh_dw, fw_dw, c_dw, o_dw = filter_size_dw
    _, sh_dw, sw_dw, _ = strides_dw
    
    #######################################
    
    input_size_batch_norm_dw = [h_dw/sh_dw, w_dw/sw_dw, o_dw]
    
    #######################################
    
    input_size_1x1 = [batch, h_dw/sh_dw, w_dw/sw_dw, o_dw]
    fh_1x1, fw_1x1, c_1x1, o_1x1 = filter_size_1x1
    
    #######################################
    
    input_size_batch_norm_1x1 = [h_dw/sh_dw, w_dw/sw_dw, o_1x1]
    
    #######################################
    
    l1 = ConvolutionDW(input_sizes=input_size_dw, filter_sizes=filter_size_dw, init=args.init, strides=strides_dw, padding="SAME", alpha=learning_rate, activation=Relu(), bias=args.bias, name=name_dw, load=weights_conv, train=train_conv)
    
    l2 = BatchNorm(size=input_size_batch_norm_dw)
    
    l3 = Convolution2D(input_sizes=input_size_1x1, filter_sizes=filter_size_1x1, init=args.init, strides=[1,1,1,1], padding="SAME", alpha=learning_rate, activation=Relu(), bias=args.bias, name=name_1x1, load=weights_conv, train=train_conv)
    
    l4 = BatchNorm(size=input_size_batch_norm_1x1)

    return [l1, l2, l3, l4]
'''
###############################################################

dropout_rate = tf.placeholder(tf.float32, shape=())
learning_rate = tf.placeholder(tf.float32, shape=())

########################

l0 = Convolution2D(input_sizes=[batch_size, 224, 224, 3], filter_sizes=[3, 3, 3, 24], init=args.init, strides=[1, 2, 2, 1], padding="SAME", alpha=learning_rate, activation=Relu(), bias=args.bias, name="conv1", load=weights_conv, train=train_conv)

########################

l1_1 = ConvolutionDW(input_sizes=[batch_size, 112, 112, 24], filter_sizes=[3, 3, 24, 1], init=args.init, strides=[1, 1, 1, 1], padding="SAME", alpha=learning_rate, activation=Relu(), bias=args.bias, name="conv_dw_1", load=weights_conv, train=train_conv)
l1_2 = BatchNorm(size=[112, 112, 24])
l1_3 = Convolution2D(input_sizes=[batch_size, 112, 112, 24], filter_sizes=[1, 1, 24, 64], init=args.init, strides=[1, 1, 1, 1], padding="SAME", alpha=learning_rate, activation=Relu(), bias=args.bias, name="conv_pw_1", load=weights_conv, train=train_conv)
l1_4 = BatchNorm(size=[112, 112, 64])

l2_1 = ConvolutionDW(input_sizes=[batch_size, 112, 112, 64], filter_sizes=[3, 3, 64, 1], init=args.init, strides=[1, 2, 2, 1], padding="SAME", alpha=learning_rate, activation=Relu(), bias=args.bias, name="conv_dw_2", load=weights_conv, train=train_conv)
l2_2 = BatchNorm(size=[56, 56, 64])
l2_3 = Convolution2D(input_sizes=[batch_size, 56, 56, 64], filter_sizes=[1, 1, 64, 128], init=args.init, strides=[1, 1, 1, 1], padding="SAME", alpha=learning_rate, activation=Relu(), bias=args.bias, name="conv_pw_2", load=weights_conv, train=train_conv)
l2_4 = BatchNorm(size=[56, 56, 128])

l3_1 = ConvolutionDW(input_sizes=[batch_size, 56, 56, 128], filter_sizes=[3, 3, 128, 1], init=args.init, strides=[1, 1, 1, 1], padding="SAME", alpha=learning_rate, activation=Relu(), bias=args.bias, name="conv_dw_3", load=weights_conv, train=train_conv)
l3_2 = BatchNorm(size=[56, 56, 128])
l3_3 = Convolution2D(input_sizes=[batch_size, 56, 56, 128], filter_sizes=[1, 1, 128, 128], init=args.init, strides=[1, 1, 1, 1], padding="SAME", alpha=learning_rate, activation=Relu(), bias=args.bias, name="conv_pw_3", load=weights_conv, train=train_conv)
l3_4 = BatchNorm(size=[56, 56, 128])

l4_1 = ConvolutionDW(input_sizes=[batch_size, 56, 56, 128], filter_sizes=[3, 3, 128, 1], init=args.init, strides=[1, 2, 2, 1], padding="SAME", alpha=learning_rate, activation=Relu(), bias=args.bias, name="conv_dw_4", load=weights_conv, train=train_conv)
l4_2 = BatchNorm(size=[28, 28, 128])
l4_3 = Convolution2D(input_sizes=[batch_size, 28, 28, 128], filter_sizes=[1, 1, 128, 256], init=args.init, strides=[1, 1, 1, 1], padding="SAME", alpha=learning_rate, activation=Relu(), bias=args.bias, name="conv_pw_4", load=weights_conv, train=train_conv)
l4_4 = BatchNorm(size=[28, 28, 256])

l5_1 = ConvolutionDW(input_sizes=[batch_size, 28, 28, 256], filter_sizes=[3, 3, 256, 1], init=args.init, strides=[1, 1, 1, 1], padding="SAME", alpha=learning_rate, activation=Relu(), bias=args.bias, name="conv_dw_5", load=weights_conv, train=train_conv)
l5_2 = BatchNorm(size=[28, 28, 256])
l5_3 = Convolution2D(input_sizes=[batch_size, 28, 28, 256], filter_sizes=[1, 1, 256, 256], init=args.init, strides=[1, 1, 1, 1], padding="SAME", alpha=learning_rate, activation=Relu(), bias=args.bias, name="conv_pw_5", load=weights_conv, train=train_conv)
l5_4 = BatchNorm(size=[28, 28, 256])

l6_1 = ConvolutionDW(input_sizes=[batch_size, 28, 28, 256], filter_sizes=[3, 3, 256, 1], init=args.init, strides=[1, 2, 2, 1], padding="SAME", alpha=learning_rate, activation=Relu(), bias=args.bias, name="conv_dw_6", load=weights_conv, train=train_conv)
l6_2 = BatchNorm(size=[14, 14, 256])
l6_3 = Convolution2D(input_sizes=[batch_size, 14, 14, 256], filter_sizes=[1, 1, 256, 512], init=args.init, strides=[1, 1, 1, 1], padding="SAME", alpha=learning_rate, activation=Relu(), bias=args.bias, name="conv_pw_6", load=weights_conv, train=train_conv)
l6_4 = BatchNorm(size=[14, 14, 512])

########################

l7_1_1 = ConvolutionDW(input_sizes=[batch_size, 14, 14, 512], filter_sizes=[3, 3, 512, 1], init=args.init, strides=[1, 1, 1, 1], padding="SAME", alpha=learning_rate, activation=Relu(), bias=args.bias, name="conv_dw_7", load=weights_conv, train=train_conv)
l7_1_2 = BatchNorm(size=[14, 14, 512])
l7_1_3 = Convolution2D(input_sizes=[batch_size, 14, 14, 512], filter_sizes=[1, 1, 512, 512], init=args.init, strides=[1, 1, 1, 1], padding="SAME", alpha=learning_rate, activation=Relu(), bias=args.bias, name="conv_pw_7", load=weights_conv, train=train_conv)
l7_1_4 = BatchNorm(size=[14, 14, 512])

l7_2_1 = ConvolutionDW(input_sizes=[batch_size, 14, 14, 512], filter_sizes=[3, 3, 512, 1], init=args.init, strides=[1, 1, 1, 1], padding="SAME", alpha=learning_rate, activation=Relu(), bias=args.bias, name="conv_dw_8", load=weights_conv, train=train_conv)
l7_2_2 = BatchNorm(size=[14, 14, 512])
l7_2_3 = Convolution2D(input_sizes=[batch_size, 14, 14, 512], filter_sizes=[1, 1, 512, 512], init=args.init, strides=[1, 1, 1, 1], padding="SAME", alpha=learning_rate, activation=Relu(), bias=args.bias, name="conv_pw_8", load=weights_conv, train=train_conv)
l7_2_4 = BatchNorm(size=[14, 14, 512])

l7_3_1 = ConvolutionDW(input_sizes=[batch_size, 14, 14, 512], filter_sizes=[3, 3, 512, 1], init=args.init, strides=[1, 1, 1, 1], padding="SAME", alpha=learning_rate, activation=Relu(), bias=args.bias, name="conv_dw_9", load=weights_conv, train=train_conv)
l7_3_2 = BatchNorm(size=[14, 14, 512])
l7_3_3 = Convolution2D(input_sizes=[batch_size, 14, 14, 512], filter_sizes=[1, 1, 512, 512], init=args.init, strides=[1, 1, 1, 1], padding="SAME", alpha=learning_rate, activation=Relu(), bias=args.bias, name="conv_pw_9", load=weights_conv, train=train_conv)
l7_3_4 = BatchNorm(size=[14, 14, 512])

l7_4_1 = ConvolutionDW(input_sizes=[batch_size, 14, 14, 512], filter_sizes=[3, 3, 512, 1], init=args.init, strides=[1, 1, 1, 1], padding="SAME", alpha=learning_rate, activation=Relu(), bias=args.bias, name="conv_dw_10", load=weights_conv, train=train_conv)
l7_4_2 = BatchNorm(size=[14, 14, 512])
l7_4_3 = Convolution2D(input_sizes=[batch_size, 14, 14, 512], filter_sizes=[1, 1, 512, 512], init=args.init, strides=[1, 1, 1, 1], padding="SAME", alpha=learning_rate, activation=Relu(), bias=args.bias, name="conv_pw_10", load=weights_conv, train=train_conv)
l7_4_4 = BatchNorm(size=[14, 14, 512])

l7_5_1 = ConvolutionDW(input_sizes=[batch_size, 14, 14, 512], filter_sizes=[3, 3, 512, 1], init=args.init, strides=[1, 1, 1, 1], padding="SAME", alpha=learning_rate, activation=Relu(), bias=args.bias, name="conv_dw_11", load=weights_conv, train=train_conv)
l7_5_2 = BatchNorm(size=[14, 14, 512])
l7_5_3 = Convolution2D(input_sizes=[batch_size, 14, 14, 512], filter_sizes=[1, 1, 512, 512], init=args.init, strides=[1, 1, 1, 1], padding="SAME", alpha=learning_rate, activation=Relu(), bias=args.bias, name="conv_pw_11", load=weights_conv, train=train_conv)
l7_5_4 = BatchNorm(size=[14, 14, 512])

########################

l8_1 = ConvolutionDW(input_sizes=[batch_size, 14, 14, 512], filter_sizes=[3, 3, 512, 1], init=args.init, strides=[1, 2, 2, 1], padding="SAME", alpha=learning_rate, activation=Relu(), bias=args.bias, name="conv_dw_12", load=weights_conv, train=train_conv)
l8_2 = BatchNorm(size=[7, 7, 512])
l8_3 = Convolution2D(input_sizes=[batch_size, 7, 7, 512], filter_sizes=[1, 1, 512, 1024], init=args.init, strides=[1, 1, 1, 1], padding="SAME", alpha=learning_rate, activation=Relu(), bias=args.bias, name="conv_pw_12", load=weights_conv, train=train_conv)
l8_4 = BatchNorm(size=[7, 7, 1024])

l9_1 = ConvolutionDW(input_sizes=[batch_size, 7, 7, 1024], filter_sizes=[3, 3, 1024, 1], init=args.init, strides=[1, 1, 1, 1], padding="SAME", alpha=learning_rate, activation=Relu(), bias=args.bias, name="conv_dw_13", load=weights_conv, train=train_conv)
l9_2 = BatchNorm(size=[7, 7, 1024])
l9_3 = Convolution2D(input_sizes=[batch_size, 7, 7, 1024], filter_sizes=[1, 1, 1024, 1024], init=args.init, strides=[1, 1, 1, 1], padding="SAME", alpha=learning_rate, activation=Relu(), bias=args.bias, name="conv_pw_13", load=weights_conv, train=train_conv)
l9_4 = BatchNorm(size=[7, 7, 1024])

########################

l10 = AvgPool(size=[batch_size, 7, 7, 1024], ksize=[1, 7, 7, 1], strides=[1, 7, 7, 1], padding="SAME")

########################

l11 = ConvToFullyConnected(shape=[1, 1, 1024])

l12 = FullyConnected(size=[1024, 1000], num_classes=num_classes, init_weights=args.init, alpha=learning_rate, activation=Relu(), bias=args.bias, last_layer=False, name="fc1", load=weights_fc, train=train_fc)

l13 = Dropout(rate=dropout_rate)

l14 = FullyConnected(size=[1000, 1000], num_classes=num_classes, init_weights=args.init, alpha=learning_rate, activation=Linear(), bias=args.bias, last_layer=True, name="fc2", load=weights_fc, train=train_fc)

###############################################################

model = Model(layers=[l0,                             \
                      l1_1, l1_2, l1_3, l1_4,         \
                      l2_1, l2_2, l2_3, l2_4,         \
                      l3_1, l3_2, l3_3, l3_4,         \
                      l4_1, l4_2, l4_3, l4_4,         \
                      l5_1, l5_2, l5_3, l5_4,         \
                      l6_1, l6_2, l6_3, l6_4,         \
                      l7_1_1, l7_1_2, l7_1_3, l7_1_4, \
                      l7_2_1, l7_2_2, l7_2_3, l7_2_4, \
                      l7_3_1, l7_3_2, l7_3_3, l7_3_4, \
                      l7_4_1, l7_4_2, l7_4_3, l7_4_4, \
                      l7_5_1, l7_5_2, l7_5_3, l7_5_4, \
                      l8_1, l8_2, l8_3, l8_4,         \
                      l9_1, l9_2, l9_3, l9_4,         \
                      l10,                            \
                      l11,                            \
                      l12,                            \
                      l13,                            \
                      l14])

###############################################################

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

# top5 = tf.nn.in_top_k(predictions=predict, targets=tf.argmax(labels,1), k=5)
top5 = in_top_k(predict, tf.argmax(labels,1), k=5)
total_top5 = tf.reduce_sum(tf.cast(top5, tf.float32))

weights = model.get_weights()

print (model.num_params())

###############################################################

config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.allow_growth=True

# sess = tf.InteractiveSession(config=config)
# tf.global_variables_initializer().run()
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

    lr = alpha
    print (ii)

    ##################################################################

    sess.run(train_iterator.initializer, feed_dict={filename: train_imgs, label: train_labs})

    train_total = 0.0
    train_correct = 0.0
    train_top5 = 0.0
    
    for j in range(0, len(train_imgs), batch_size):
        print (j)
        
        _total_correct, _top5, _ = sess.run([total_correct, total_top5, train], feed_dict={handle: train_handle, dropout_rate: args.dropout, learning_rate: lr})
        
        train_total += batch_size
        train_correct += _total_correct
        train_top5 += _top5
        
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
    
    sess.run(val_iterator.initializer, feed_dict={filename: val_imgs, label: val_labs})
    
    val_total = 0.0
    val_correct = 0.0
    val_top5 = 0.0
    
    for j in range(0, len(val_imgs), batch_size):
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
        print ('phase 1')
    elif phase == 1:
        dacc = train_accs[-1] - train_accs[-2]
        if dacc <= 0.01:
            alpha = 0.1 * args.alpha
            phase = 2
            print ('phase 2')
    elif phase == 2:
        dacc = train_accs[-1] - train_accs[-2]
        if dacc <= 0.005:
            alpha = 0.05 * args.alpha
            phase = 3
            print ('phase 3')

    if args.save:
        [w] = sess.run([weights], feed_dict={handle: val_handle, dropout_rate: 0.0, learning_rate: 0.0})

        w['train_acc'] = train_accs
        w['train_acc_top5'] = train_accs_top5
        w['val_acc'] = val_accs
        w['val_acc_top5'] = val_accs_top5

        np.save(args.name, w)

    print('epoch {}/{}'.format(ii, epochs))
    
    
    
    
    
    
    
    

