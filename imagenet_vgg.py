
import argparse
import os
import sys

##############################################

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--lr', type=float, default=1e-2)
parser.add_argument('--eps', type=float, default=1.)
parser.add_argument('--dropout', type=float, default=0.5)
parser.add_argument('--act', type=str, default='relu')
parser.add_argument('--bias', type=float, default=0.)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--dfa', type=int, default=0)
parser.add_argument('--sparse', type=int, default=0)
parser.add_argument('--rank', type=int, default=0)
parser.add_argument('--init', type=str, default="glorot_uniform")
parser.add_argument('--save', type=int, default=0)
parser.add_argument('--name', type=str, default="imagenet_vgg")
parser.add_argument('--load', type=str, default=None)
args = parser.parse_args()

if args.gpu >= 0:
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpu)

exxact = 0
if exxact:
    val_path = '/home/bcrafton3/Data_SSD/ILSVRC2012/val/'
    train_path = '/home/bcrafton3/Data_SSD/ILSVRC2012/train/'
else:
    val_path = '/usr/scratch/bcrafton/ILSVRC2012/val/'
    train_path = '/usr/scratch/bcrafton/ILSVRC2012/train/'

val_labels = './imagenet_labels/validation_labels.txt'
train_labels = './imagenet_labels/train_labels.txt'

IMAGENET_MEAN = [123.68, 116.78, 103.94]

##############################################

import keras
import tensorflow as tf
import numpy as np
np.set_printoptions(threshold=1000)

from lib.Model import Model

from lib.Layer import Layer 
from lib.ConvToFullyConnected import ConvToFullyConnected
from lib.FullyConnected import FullyConnected
from lib.Convolution import Convolution
from lib.MaxPool import MaxPool
from lib.Dropout import Dropout
from lib.FeedbackFC import FeedbackFC
from lib.FeedbackConv import FeedbackConv

from lib.Activation import Activation
from lib.Activation import Relu

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

    for subdir, dirs, files in os.walk(val_path):
        for file in files:
            validation_images.append(os.path.join(val_path, file))
    validation_images = sorted(validation_images)

    validation_labels_file = open(val_labels)
    lines = validation_labels_file.readlines()
    for ii in range(len(lines)):
        validation_labels.append(int(lines[ii]))

    remainder = len(validation_labels) % args.batch_size
    validation_images = validation_images[:(-remainder)]
    validation_labels = validation_labels[:(-remainder)]

    return validation_images, validation_labels
    
def get_train_dataset():

    label_counter = 0
    training_images = []
    training_labels = []

    f = open(train_labels, 'r')
    lines = f.readlines()

    labels = {}
    for line in lines:
        line = line.split(' ')
        labels[line[0]] = label_counter
        label_counter += 1

    f.close()

    print ("building train dataset")

    for subdir, dirs, files in os.walk(train_path):
        for folder in dirs:
            for folder_subdir, folder_dirs, folder_files in os.walk(os.path.join(subdir, folder)):
                for file in folder_files:
                    training_images.append(os.path.join(folder_subdir, file))
                    training_labels.append(labels[folder])

    remainder = len(training_labels) % args.batch_size
    training_images = training_images[:(-remainder)]
    training_labels = training_labels[:(-remainder)]

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

if args.act == 'tanh':
    act = Tanh()
elif args.act == 'relu':
    act = Relu()
else:
    assert(False)

###############################################################

weights_conv = './transfer/vgg_weights.npy'
weights_fc = None

###############################################################

batch_size = tf.placeholder(tf.int32, shape=())
dropout_rate = tf.placeholder(tf.float32, shape=())
lr = tf.placeholder(tf.float32, shape=())

l0 = Convolution(input_shape=[batch_size, 224, 224, 3],  filter_sizes=[3, 3, 3, 64],  init=args.init, padding="SAME", activation=act, bias=args.bias, load=weights_conv, name='conv1', train=train_conv)
l1 = Convolution(input_shape=[batch_size, 224, 224, 64], filter_sizes=[3, 3, 64, 64], init=args.init, padding="SAME", activation=act, bias=args.bias, load=weights_conv, name='conv2', train=train_conv)
l2 = MaxPool(size=[batch_size, 224, 224, 64], ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

l3 = Convolution(input_shape=[batch_size, 112, 112, 64],  filter_sizes=[3, 3, 64, 128],  init=args.init, padding="SAME", activation=act, bias=args.bias, load=weights_conv, name='conv3', train=train_conv)
l4 = Convolution(input_shape=[batch_size, 112, 112, 128], filter_sizes=[3, 3, 128, 128], init=args.init, padding="SAME", activation=act, bias=args.bias, load=weights_conv, name='conv4', train=train_conv)
l5 = MaxPool(size=[batch_size, 112, 112, 128], ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

l6 = Convolution(input_shape=[batch_size, 56, 56, 128], filter_sizes=[3, 3, 128, 256], init=args.init, padding="SAME", activation=act, bias=args.bias, load=weights_conv, name='conv5', train=train_conv)
l7 = Convolution(input_shape=[batch_size, 56, 56, 256], filter_sizes=[3, 3, 256, 256], init=args.init, padding="SAME", activation=act, bias=args.bias, load=weights_conv, name='conv6', train=train_conv)
l8 = Convolution(input_shape=[batch_size, 56, 56, 256], filter_sizes=[3, 3, 256, 256], init=args.init, padding="SAME", activation=act, bias=args.bias, load=weights_conv, name='conv7', train=train_conv)
l9 = MaxPool(size=[batch_size, 56, 56, 256], ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

l10 = Convolution(input_shape=[batch_size, 28, 28, 256], filter_sizes=[3, 3, 256, 512], init=args.init, padding="SAME", activation=act, bias=args.bias, load=weights_conv, name='conv8', train=train_conv)
l11 = Convolution(input_shape=[batch_size, 28, 28, 512], filter_sizes=[3, 3, 512, 512], init=args.init, padding="SAME", activation=act, bias=args.bias, load=weights_conv, name='conv9', train=train_conv)
l12 = Convolution(input_shape=[batch_size, 28, 28, 512], filter_sizes=[3, 3, 512, 512], init=args.init, padding="SAME", activation=act, bias=args.bias, load=weights_conv, name='conv10', train=train_conv)
l13 = MaxPool(size=[batch_size, 28, 28, 512], ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

l14 = Convolution(input_shape=[batch_size, 14, 14, 512], filter_sizes=[3, 3, 512, 512], init=args.init, padding="SAME", activation=act, bias=args.bias, load=weights_conv, name='conv11', train=train_conv)
l15 = Convolution(input_shape=[batch_size, 14, 14, 512], filter_sizes=[3, 3, 512, 512], init=args.init, padding="SAME", activation=act, bias=args.bias, load=weights_conv, name='conv12', train=train_conv)
l16 = Convolution(input_shape=[batch_size, 14, 14, 512], filter_sizes=[3, 3, 512, 512], init=args.init, padding="SAME", activation=act, bias=args.bias, load=weights_conv, name='conv13', train=train_conv)
l17 = MaxPool(size=[batch_size, 14, 14, 512], ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

l18 = ConvToFullyConnected(input_shape=[7, 7, 512])

l19 = FullyConnected(input_shape=7*7*512, size=4096, init=args.init, activation=act, bias=args.bias, load=weights_fc, name='fc1', train=train_fc)
l20 = Dropout(rate=dropout_rate)
l21 = FeedbackFC(size=[7*7*512, 4096], num_classes=1000, sparse=args.sparse, rank=args.rank, name='fc1_fb')

l22 = FullyConnected(input_shape=4096, size=4096, init=args.init, activation=act, bias=args.bias, load=weights_fc, name='fc2', train=train_fc)
l23 = Dropout(rate=dropout_rate)
l24 = FeedbackFC(size=[4096, 4096], num_classes=1000, sparse=args.sparse, rank=args.rank, name='fc2_fb')

l25 = FullyConnected(input_shape=4096, size=1000, init=args.init, bias=args.bias, load=weights_fc, name='fc3', train=train_fc)

###############################################################

model = Model(layers=[l0, l1, l2, l3, l4, l5, l6, l7, l8, l9, l10, l11, l12, l13, l14, l15, l16, l17, l18, l19, l20, l21, l22, l23, l24, l25])
predict = tf.nn.softmax(model.predict(X=features))
weights = model.get_weights()

if args.dfa:
    grads_and_vars = model.dfa_gvs(X=features, Y=labels)
else:
    grads_and_vars = model.gvs(X=features, Y=labels)
        
train = tf.train.AdamOptimizer(learning_rate=lr, epsilon=args.eps).apply_gradients(grads_and_vars=grads_and_vars)

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

phase = 0
lr_decay = args.lr

for ii in range(args.epochs):

    sess.run(train_iterator.initializer, feed_dict={filename: train_imgs, label: train_labs})

    train_total = 0.0
    train_correct = 0.0
    train_top5 = 0.0
    
    for j in range(0, len(train_imgs), args.batch_size):
        [_total_correct, _top5, _] = sess.run([total_correct, total_top5, train], feed_dict={handle: train_handle, batch_size: args.batch_size, dropout_rate: args.dropout, lr: lr_decay})
        
        train_total += args.batch_size
        train_correct += _total_correct
        train_top5 += _top5
        
        train_acc = train_correct / train_total
        train_acc_top5 = train_top5 / train_total
        
        if (j % (1000 * args.batch_size) == 0):
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
    
    for j in range(0, len(val_imgs), args.batch_size):
        [_total_correct, _top5] = sess.run([total_correct, total_top5], feed_dict={handle: val_handle, batch_size: args.batch_size, dropout_rate: 0.0, lr: 0.0})
        
        val_total += args.batch_size
        val_correct += _total_correct
        val_top5 += _top5
        
        val_acc = val_correct / val_total
        val_acc_top5 = val_top5 / val_total
        
        if (j % (1000 * args.batch_size) == 0):
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
        dacc = val_accs[-1] - val_accs[-2]
        if dacc <= 0.01:
            lr_decay = 0.1 * args.lr
            phase = 2
            print ('phase 2')
    elif phase == 2:
        dacc = val_accs[-1] - val_accs[-2]
        if dacc <= 0.005:
            lr_decay = 0.05 * args.lr
            phase = 3
            print ('phase 3')

    if args.save:
        [w] = sess.run([weights], feed_dict={handle: val_handle, dropout_rate: 0.0, learning_rate: 0.0})
        w['train_acc'] = train_accs
        w['train_acc_top5'] = train_accs_top5
        w['val_acc'] = val_accs
        w['val_acc_top5'] = val_accs_top5
        np.save(args.name, w)

    print('epoch %d/%d' % (ii, args.epochs))
    
    
    
    
    
    
    
    

