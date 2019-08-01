
import argparse
import os
import sys

##############################################

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--alpha', type=float, default=1e-2)
parser.add_argument('--decay', type=float, default=1.)
parser.add_argument('--eps', type=float, default=1.)
parser.add_argument('--act', type=str, default='tanh')
parser.add_argument('--bias', type=float, default=0.)
parser.add_argument('--dropout', type=float, default=0.5)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--dfa', type=int, default=0)
parser.add_argument('--sparse', type=int, default=0)
parser.add_argument('--rank', type=int, default=0)
parser.add_argument('--init', type=str, default="sqrt_fan_in")
parser.add_argument('--opt', type=str, default="gd")
parser.add_argument('--save', type=int, default=0)
parser.add_argument('--name', type=str, default="imagenet_vgg")
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

from Model import Model

from Layer import Layer 
from ConvToFullyConnected import ConvToFullyConnected
from FullyConnected import FullyConnected
from Convolution import Convolution
from MaxPool import MaxPool
from Dropout import Dropout
from FeedbackFC import FeedbackFC
from FeedbackConv import FeedbackConv

from Activation import Activation
from Activation import Sigmoid
from Activation import Relu
from Activation import Tanh
from Activation import Softmax
from Activation import LeakyRelu
from Activation import Linear

##############################################

batch_size = args.batch_size
num_classes = 1000
epochs = args.epochs
data_augmentation = False

EPOCHS = args.epochs
BATCH_SIZE = args.batch_size

IMAGENET_MEAN = [123.68, 116.78, 103.94]

##############################################

def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

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

    for subdir, dirs, files in os.walk('/home/bcrafton3/ILSVRC2012/val/'):
        for file in files:
            validation_images.append(os.path.join('/home/bcrafton3/ILSVRC2012/val/', file))
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

    for subdir, dirs, files in os.walk('/home/bcrafton3/ILSVRC2012/train/'):
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
# val_dataset = val_dataset.shuffle(len(val_imgs))
val_dataset = val_dataset.map(parse_function, num_parallel_calls=4)
val_dataset = val_dataset.map(val_preprocess, num_parallel_calls=4)
val_dataset = val_dataset.batch(batch_size)
val_dataset = val_dataset.repeat()
val_dataset = val_dataset.prefetch(8)

###############################################################

train_imgs, train_labs = get_train_dataset()

train_dataset = tf.data.Dataset.from_tensor_slices((filename, label))
# train_dataset = train_dataset.shuffle(len(train_imgs))
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
# labels = tf.one_hot(labels, depth=num_classes)

train_iterator = train_dataset.make_initializable_iterator()
val_iterator = val_dataset.make_initializable_iterator()

###############################################################

train_conv=False
weights_conv='../vgg_weights/vgg_weights.npy'

###############################################################

dropout_rate = tf.placeholder(tf.float32, shape=())
learning_rate = tf.placeholder(tf.float32, shape=())

l0 = Convolution(input_sizes=[batch_size, 224, 224, 3], filter_sizes=[3, 3, 3, 64], num_classes=num_classes, init_filters=args.init, strides=[1, 1, 1, 1], padding="SAME", alpha=learning_rate, activation=Relu(), bias=0.0, last_layer=False, name="conv1", load=weights_conv, train=train_conv)
l1 = Convolution(input_sizes=[batch_size, 224, 224, 64], filter_sizes=[3, 3, 64, 64], num_classes=num_classes, init_filters=args.init, strides=[1, 1, 1, 1], padding="SAME", alpha=learning_rate, activation=Relu(), bias=0.0, last_layer=False, name="conv2", load=weights_conv, train=train_conv)
l2 = MaxPool(size=[batch_size, 224, 224, 64], ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="VALID")

l3 = Convolution(input_sizes=[batch_size, 112, 112, 64], filter_sizes=[3, 3, 64, 128], num_classes=num_classes, init_filters=args.init, strides=[1, 1, 1, 1], padding="SAME", alpha=learning_rate, activation=Relu(), bias=0.0, last_layer=False, name="conv3", load=weights_conv, train=train_conv)
l4 = Convolution(input_sizes=[batch_size, 112, 112, 128], filter_sizes=[3, 3, 128, 128], num_classes=num_classes, init_filters=args.init, strides=[1, 1, 1, 1], padding="SAME", alpha=learning_rate, activation=Relu(), bias=0.0, last_layer=False, name="conv4", load=weights_conv, train=train_conv)
l5 = MaxPool(size=[batch_size, 112, 112, 128], ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="VALID")

l6 = Convolution(input_sizes=[batch_size, 56, 56, 128], filter_sizes=[3, 3, 128, 256], num_classes=num_classes, init_filters=args.init, strides=[1, 1, 1, 1], padding="SAME", alpha=learning_rate, activation=Relu(), bias=0.0, last_layer=False, name="conv5", load=weights_conv, train=train_conv)
l7 = Convolution(input_sizes=[batch_size, 56, 56, 256], filter_sizes=[3, 3, 256, 256], num_classes=num_classes, init_filters=args.init, strides=[1, 1, 1, 1], padding="SAME", alpha=learning_rate, activation=Relu(), bias=0.0, last_layer=False, name="conv6", load=weights_conv, train=train_conv)
l8 = Convolution(input_sizes=[batch_size, 56, 56, 256], filter_sizes=[3, 3, 256, 256], num_classes=num_classes, init_filters=args.init, strides=[1, 1, 1, 1], padding="SAME", alpha=learning_rate, activation=Relu(), bias=0.0, last_layer=False, name="conv7", load=weights_conv, train=train_conv)
l9 = MaxPool(size=[batch_size, 56, 56, 256], ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="VALID")

l10 = Convolution(input_sizes=[batch_size, 28, 28, 256], filter_sizes=[3, 3, 256, 512], num_classes=num_classes, init_filters=args.init, strides=[1, 1, 1, 1], padding="SAME", alpha=learning_rate, activation=Relu(), bias=0.0, last_layer=False, name="conv8", load=weights_conv, train=train_conv)
l11 = Convolution(input_sizes=[batch_size, 28, 28, 512], filter_sizes=[3, 3, 512, 512], num_classes=num_classes, init_filters=args.init, strides=[1, 1, 1, 1], padding="SAME", alpha=learning_rate, activation=Relu(), bias=0.0, last_layer=False, name="conv9", load=weights_conv, train=train_conv)
l12 = Convolution(input_sizes=[batch_size, 28, 28, 512], filter_sizes=[3, 3, 512, 512], num_classes=num_classes, init_filters=args.init, strides=[1, 1, 1, 1], padding="SAME", alpha=learning_rate, activation=Relu(), bias=0.0, last_layer=False, name="conv10", load=weights_conv, train=train_conv)
l13 = MaxPool(size=[batch_size, 28, 28, 512], ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="VALID")

l14 = Convolution(input_sizes=[batch_size, 14, 14, 512], filter_sizes=[3, 3, 512, 512], num_classes=num_classes, init_filters=args.init, strides=[1, 1, 1, 1], padding="SAME", alpha=learning_rate, activation=Relu(), bias=0.0, last_layer=False, name="conv11", load=weights_conv, train=train_conv)
l15 = Convolution(input_sizes=[batch_size, 14, 14, 512], filter_sizes=[3, 3, 512, 512], num_classes=num_classes, init_filters=args.init, strides=[1, 1, 1, 1], padding="SAME", alpha=learning_rate, activation=Relu(), bias=0.0, last_layer=False, name="conv12", load=weights_conv, train=train_conv)
l16 = Convolution(input_sizes=[batch_size, 14, 14, 512], filter_sizes=[3, 3, 512, 512], num_classes=num_classes, init_filters=args.init, strides=[1, 1, 1, 1], padding="SAME", alpha=learning_rate, activation=Relu(), bias=0.0, last_layer=False, name="conv13", load=weights_conv, train=train_conv)
l17 = MaxPool(size=[batch_size, 14, 14, 512], ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="VALID")

###############################################################

model = Model(layers=[l0, l1, l2, l3, l4, l5, l6, l7, l8, l9, l10, l11, l12, l13, l14, l15, l16, l17])
predict = model.predict(X=features)

###############################################################

config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.InteractiveSession(config=config)
tf.global_variables_initializer().run()

train_handle = sess.run(train_iterator.string_handle())
val_handle = sess.run(val_iterator.string_handle())

###############################################################
'''
sess.run(train_iterator.initializer, feed_dict={filename: train_imgs, label: train_labs})

for i in range(0, len(train_imgs), batch_size):
    print (i)
    [_predict, _labels] = sess.run([predict, labels], feed_dict={handle: train_handle, dropout_rate: 0.0, learning_rate: 0.0})
    _predict = np.reshape(_predict, (batch_size, 7 * 7 * 512))
    for j in range(batch_size):
        name = './tfrecord/train/%d.tfrecord' % (int(i + j))
        with tf.python_io.TFRecordWriter(name) as writer:
            image_raw = _predict[j].tostring()
            _feature={
                    'label': _int64_feature(int(_labels[j])),
                    'image_raw': _bytes_feature(image_raw)
                    }
            _features=tf.train.Features(feature=_feature)
            example = tf.train.Example(features=_features)
            writer.write(example.SerializeToString())
'''
##################################################################

sess.run(val_iterator.initializer, feed_dict={filename: val_imgs, label: val_labs})

for i in range(0, len(val_imgs), batch_size):
    print (i)
    [_predict, _labels] = sess.run([predict, labels], feed_dict={handle: val_handle, dropout_rate: 0.0, learning_rate: 0.0})
    _predict = np.reshape(_predict, (batch_size, 7 * 7 * 512))
    for j in range(batch_size):
        name = './tfrecord/test/%d.tfrecord' % (int(i + j))
        with tf.python_io.TFRecordWriter(name) as writer:
            image_raw = _predict[j].tostring()
            _feature={
                    'label': _int64_feature(int(_labels[j])),
                    'image_raw': _bytes_feature(image_raw)
                    }
            _features=tf.train.Features(feature=_feature)
            example = tf.train.Example(features=_features)
            writer.write(example.SerializeToString())





