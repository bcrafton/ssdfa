
import argparse
import os
import sys

##############################################

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--alpha', type=float, default=1e-4)
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
parser.add_argument('--name', type=str, default="imagenet_vgg")
parser.add_argument('--load', type=str, default=None)
args = parser.parse_args()

if args.gpu >= 0:
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpu)
    
exxact = 0
if exxact:
    val_data_path = '/home/bcrafton3/Data_SSD/ILSVRC2012/val/'
    val_label_path = '/home/bcrafton3/dfa/imagenet_labels/validation_labels.txt'
    train_data_path = '/home/bcrafton3/Data_SSD/ILSVRC2012/train/'
    train_label_path = '/home/bcrafton3/dfa/imagenet_labels/train_labels.txt'
else:
    val_data_path = '/usr/scratch/ILSVRC2012/val/'
    val_label_path = '/usr/scratch/ILSVRC2012/validation_labels.txt'
    train_data_path = '/usr/scratch/ILSVRC2012/train/'
    train_label_path = '/usr/scratch/ILSVRC2012/train_labels.txt'

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

from lib.mobilenet_preprocessing import preprocess_for_train
from lib.mobilenet_preprocessing import preprocess_for_eval

from lib.ModelMSE import Model

from lib.Layer import Layer 
from lib.ConvToFullyConnected import ConvToFullyConnected
from lib.FullyConnected import FullyConnected
from lib.Convolution2D import Convolution2D
from lib.Convolution3D import Convolution3D
from lib.ConvolutionDW import ConvolutionDW
from lib.MaxPool import MaxPool
from lib.AvgPool import AvgPool
from lib.UpSample import UpSample
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

from lib.ConvBlock import ConvBlock
from lib.VGG_Block_AE import VGGBlock

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

data_augmentation = False
# IMAGENET_MEAN = [123.68, 116.78, 103.94]

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
    return flip_image, label

    # means = tf.reshape(tf.constant(IMAGENET_MEAN), [1, 1, 3])
    # centered_image = flip_image - means                                     # (5)

    # return centered_image, label
    

# Preprocessing (for validation)
# (3) Take a central 224x224 crop to the scaled image
# (4) Substract the per color mean `IMAGENET_MEAN`
# Note: we don't normalize the data here, as VGG was trained without normalization
def val_preprocess(image, label):
    crop_image = tf.image.resize_image_with_crop_or_pad(image, 224, 224)    # (3)
    return crop_image, label

    # means = tf.reshape(tf.constant(IMAGENET_MEAN), [1, 1, 3])
    # centered_image = crop_image - means                                     # (4)

    # return centered_image, label

##############################################

def get_validation_dataset():
    label_counter = 0
    validation_images = []
    validation_labels = []

    print ("building validation dataset")

    for subdir, dirs, files in os.walk(val_data_path):
        for file in files:
            validation_images.append(os.path.join(val_data_path, file))
    validation_images = sorted(validation_images)

    validation_labels_file = open(val_label_path)
    lines = validation_labels_file.readlines()
    for ii in range(len(lines)):
        validation_labels.append(int(lines[ii]))

    print (len(validation_images), len(validation_labels))
    remainder = len(validation_labels) % args.batch_size
    validation_images = validation_images[:(-remainder)]
    validation_labels = validation_labels[:(-remainder)]

    print("validation data is ready...")

    return validation_images, validation_labels
    
def get_train_dataset():

    label_counter = 0
    training_images = []
    training_labels = []

    print ("making labels dict")

    f = open(train_label_path)
    lines = f.readlines()

    labels = {}
    for line in lines:
        line = line.split(' ')
        labels[line[0]] = label_counter
        label_counter += 1

    f.close()

    print ("building dataset")

    for subdir, dirs, files in os.walk(train_data_path):
        for folder in dirs:
            for folder_subdir, folder_dirs, folder_files in os.walk(os.path.join(subdir, folder)):
                for file in folder_files:
                    training_images.append(os.path.join(folder_subdir, file))
                    training_labels.append(labels[folder])

    remainder = len(training_labels) % args.batch_size
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
val_dataset = val_dataset.map(parse_function, num_parallel_calls=4)
val_dataset = val_dataset.map(val_preprocess, num_parallel_calls=4)
val_dataset = val_dataset.batch(args.batch_size)
val_dataset = val_dataset.repeat()
val_dataset = val_dataset.prefetch(8)

###############################################################

train_imgs, train_labs = get_train_dataset()

train_dataset = tf.data.Dataset.from_tensor_slices((filename, label))
train_dataset = train_dataset.shuffle(len(train_imgs))
train_dataset = train_dataset.shuffle(len(train_imgs), reshuffle_each_iteration=False)
train_dataset = train_dataset.map(parse_function, num_parallel_calls=4)
train_dataset = train_dataset.map(train_preprocess, num_parallel_calls=4)
train_dataset = train_dataset.batch(args.batch_size)
train_dataset = train_dataset.repeat()
train_dataset = train_dataset.prefetch(8)

###############################################################

handle = tf.placeholder(tf.string, shape=[])
iterator = tf.data.Iterator.from_string_handle(handle, train_dataset.output_types, train_dataset.output_shapes)
features, labels = iterator.get_next()
features = tf.reshape(features, (-1, 224, 224, 3))
labels = tf.one_hot(labels, depth=1000)

train_iterator = train_dataset.make_initializable_iterator()
val_iterator = val_dataset.make_initializable_iterator()

###############################################################

weights_conv = args.load

if weights_conv:
    train_conv = False
else:
    train_conv = True
    
###############################################################

dropout_rate = tf.placeholder(tf.float32, shape=())
learning_rate = tf.placeholder(tf.float32, shape=())

X = features / 255.

##########
# encoder.

l1_1 = VGGBlock(input_shape=[args.batch_size, 224, 224, 3],  filter_shape=[3, 32],      strides=[1,1,1,1], init=args.init, name='block1', load=weights_conv, train=train_conv)
l1_2 = VGGBlock(input_shape=[args.batch_size, 224, 224, 32], filter_shape=[32, 32],     strides=[1,1,1,1], init=args.init, name='block2', load=weights_conv, train=train_conv)
l1_3 = AvgPool(size=[args.batch_size, 224, 224, 32], ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

l2_1 = VGGBlock(input_shape=[args.batch_size, 112, 112, 32], filter_shape=[32, 64],     strides=[1,1,1,1], init=args.init, name='block3', load=weights_conv, train=train_conv)
l2_2 = VGGBlock(input_shape=[args.batch_size, 112, 112, 64], filter_shape=[64, 64],     strides=[1,1,1,1], init=args.init, name='block4', load=weights_conv, train=train_conv)
l2_3 = AvgPool(size=[args.batch_size, 112, 112, 64], ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

l3_1 = VGGBlock(input_shape=[args.batch_size, 56, 56, 64],   filter_shape=[64, 128],    strides=[1,1,1,1], init=args.init, name='block5', load=weights_conv, train=train_conv)
l3_2 = VGGBlock(input_shape=[args.batch_size, 56, 56, 128],  filter_shape=[128, 128],   strides=[1,1,1,1], init=args.init, name='block6', load=weights_conv, train=train_conv)
l3_3 = VGGBlock(input_shape=[args.batch_size, 56, 56, 128],  filter_shape=[128, 128],   strides=[1,1,1,1], init=args.init, name='block7', load=weights_conv, train=train_conv)
l3_4 = AvgPool(size=[args.batch_size, 56, 56, 128], ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

l4_1 = VGGBlock(input_shape=[args.batch_size, 28, 28, 128],  filter_shape=[128, 256],   strides=[1,1,1,1], init=args.init, name='block8', load=weights_conv, train=train_conv)
l4_2 = VGGBlock(input_shape=[args.batch_size, 28, 28, 256],  filter_shape=[256, 256],   strides=[1,1,1,1], init=args.init, name='block9', load=weights_conv, train=train_conv)
l4_3 = VGGBlock(input_shape=[args.batch_size, 28, 28, 256],  filter_shape=[256, 256],   strides=[1,1,1,1], init=args.init, name='block10', load=weights_conv, train=train_conv)
l4_4 = AvgPool(size=[args.batch_size, 28, 28, 256], ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

l5_1 = VGGBlock(input_shape=[args.batch_size, 14, 14, 256],  filter_shape=[256, 512],   strides=[1,1,1,1], init=args.init, name='block11', load=weights_conv, train=train_conv)
l5_2 = VGGBlock(input_shape=[args.batch_size, 14, 14, 512],  filter_shape=[512, 512],   strides=[1,1,1,1], init=args.init, name='block12', load=weights_conv, train=train_conv)
l5_3 = VGGBlock(input_shape=[args.batch_size, 14, 14, 512],  filter_shape=[512, 512],   strides=[1,1,1,1], init=args.init, name='block13', load=weights_conv, train=train_conv)
l5_4 = AvgPool(size=[args.batch_size, 14, 14, 512], ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

l6_1 = VGGBlock(input_shape=[args.batch_size, 7, 7, 512],    filter_shape=[512, 1024],  strides=[1,1,1,1], init=args.init, name='block14', load=weights_conv, train=train_conv)
l6_2 = VGGBlock(input_shape=[args.batch_size, 7, 7, 1024],   filter_shape=[1024, 1024], strides=[1,1,1,1], init=args.init, name='block15', load=weights_conv, train=train_conv)

##########
# decoder.

l7_1 = VGGBlock(input_shape=[args.batch_size, 7, 7, 1024],   filter_shape=[1024, 1024], strides=[1,1,1,1], init=args.init, name='block16')
l7_2 = VGGBlock(input_shape=[args.batch_size, 7, 7, 1024],   filter_shape=[1024, 512],  strides=[1,1,1,1], init=args.init, name='block17')
l7_3 = UpSample(input_shape=[args.batch_size, 7, 7, 512], ksize=2)

l8_1 = VGGBlock(input_shape=[args.batch_size, 14, 14, 512],  filter_shape=[512, 512],   strides=[1,1,1,1], init=args.init, name='block18')
l8_2 = VGGBlock(input_shape=[args.batch_size, 14, 14, 512],  filter_shape=[512, 512],   strides=[1,1,1,1], init=args.init, name='block19')
l8_3 = VGGBlock(input_shape=[args.batch_size, 14, 14, 512],  filter_shape=[512, 256],   strides=[1,1,1,1], init=args.init, name='block20')
l8_4 = UpSample(input_shape=[args.batch_size, 14, 14, 256], ksize=2)

l9_1 = VGGBlock(input_shape=[args.batch_size, 28, 28, 256],  filter_shape=[256, 256],   strides=[1,1,1,1], init=args.init, name='block21')
l9_2 = VGGBlock(input_shape=[args.batch_size, 28, 28, 256],  filter_shape=[256, 256],   strides=[1,1,1,1], init=args.init, name='block22')
l9_3 = VGGBlock(input_shape=[args.batch_size, 28, 28, 256],  filter_shape=[256, 128],   strides=[1,1,1,1], init=args.init, name='block23')
l9_4 = UpSample(input_shape=[args.batch_size, 28, 28, 128], ksize=2)

l10_1 = VGGBlock(input_shape=[args.batch_size, 56, 56, 128],  filter_shape=[128, 128],   strides=[1,1,1,1], init=args.init, name='block24')
l10_2 = VGGBlock(input_shape=[args.batch_size, 56, 56, 128],  filter_shape=[128, 128],   strides=[1,1,1,1], init=args.init, name='block25')
l10_3 = VGGBlock(input_shape=[args.batch_size, 56, 56, 128],  filter_shape=[128, 64],    strides=[1,1,1,1], init=args.init, name='block26')
l10_4 = UpSample(input_shape=[args.batch_size, 56, 56, 64], ksize=2)

l11_1 = VGGBlock(input_shape=[args.batch_size, 112, 112, 64], filter_shape=[64, 64],     strides=[1,1,1,1], init=args.init, name='block27')
l11_2 = VGGBlock(input_shape=[args.batch_size, 112, 112, 64], filter_shape=[64, 32],     strides=[1,1,1,1], init=args.init, name='block28')
l11_3 = UpSample(input_shape=[args.batch_size, 112, 112, 32], ksize=2)

l12_1 = VGGBlock(input_shape=[args.batch_size, 224, 224, 32], filter_shape=[32, 32],     strides=[1,1,1,1], init=args.init, name='block29')
l12_2 = VGGBlock(input_shape=[args.batch_size, 224, 224, 32], filter_shape=[32, 3],      strides=[1,1,1,1], init=args.init, name='block30')

##########

layers=[
l1_1, l1_2, l1_3,
l2_1, l2_2, l2_3,
l3_1, l3_2, l3_3, l3_4,
l4_1, l4_2, l4_3, l4_4,
l5_1, l5_2, l5_3, l5_4,
l6_1, l6_2, 

l7_1, l7_2, l7_3,
l8_1, l8_2, l8_3, l8_4,
l9_1, l9_2, l9_3, l9_4,
l10_1, l10_2, l10_3, l10_4,
l11_1, l11_2, l11_3,
l12_1, l12_2
]

model = Model(layers=layers, shape_y=[args.batch_size, 224, 224, 3])
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

for ii in range(0, args.epochs):

    print('epoch {}/{}'.format(ii, args.epochs))

    ##################################################################

    sess.run(train_iterator.initializer, feed_dict={filename: train_imgs, label: train_labs})

    losses = []
    for jj in range(0, len(train_imgs), args.batch_size):

        if (jj % (args.batch_size * 100) == 0):
            [_, _loss, _X, _predict] = sess.run([train, loss, X, predict], feed_dict={handle: train_handle, dropout_rate: args.dropout, learning_rate: alpha})
            losses.append(_loss)

            ########################

            if args.load == None:
                ext = 'random'
            else:
                ext = args.load
            
            name = '%d_%s_%f.jpg' % (jj, ext, args.alpha)

            img1 = np.reshape(_X[0], (224, 224, 3))
            # img1 = scipy.misc.imresize(img1, 4.)            
            
            img2 = np.reshape(_predict[0], (224, 224, 3))
            # img2 = scipy.misc.imresize(img2, 4.)
            
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
    
    sess.run(val_iterator.initializer, feed_dict={filename: val_imgs, label: val_labs})
        
    for jj in range(0, len(val_imgs), args.batch_size):

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

    if args.save:
        [w] = sess.run([weights], feed_dict={handle: val_handle, dropout_rate: 0.0, learning_rate: 0.0})
        np.save(args.name, w)

    
    
