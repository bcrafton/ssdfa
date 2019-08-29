
import argparse
import os
import sys

##############################################

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default="mobile")
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--lr', type=float, default=5e-2)
parser.add_argument('--eps', type=float, default=1.)
parser.add_argument('--dropout', type=float, default=0.)
parser.add_argument('--init', type=str, default="alexnet")
parser.add_argument('--save', type=int, default=0)
parser.add_argument('--name', type=str, default="imagenet64")
parser.add_argument('--load', type=str, default=None)

parser.add_argument('--fb', type=str, default="f")
parser.add_argument('--fwd', type=str, default="f")

args = parser.parse_args()

if args.gpu >= 0:
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpu)

exxact = 0
if exxact:
    val_path = '/home/bcrafton3/Data_SSD/64x64/tfrecord/val/'
    train_path = '/home/bcrafton3/Data_SSD/64x64/tfrecord/train/'
else:
    val_path = '/usr/scratch/bcrafton/64x64/tfrecord/val/'
    train_path = '/usr/scratch/bcrafton/64x64/tfrecord/train/'

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
from lib.AvgPool import AvgPool
from lib.Dropout import Dropout
from lib.FeedbackFC import FeedbackFC
from lib.FeedbackConv import FeedbackConv

from lib.ConvBlock import ConvBlock
from lib.VGGBlock import VGGBlock
from lib.MobileBlock import MobileBlock
from lib.BatchNorm import BatchNorm

from lib.VGGNet import VGGNetTiny
from lib.VGGNet import VGGNet64
from lib.MobileNet import MobileNet64

from collections import deque
import matplotlib.pyplot as plt

##############################################

# MEAN = [122.77093945, 116.74601272, 104.09373519]

##############################################

def unit_vector(vector):
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

def factors(x):
    l = [] 
    for i in range(1, x + 1):
        if x % i == 0:
            l.append(i)
    
    mid = int(len(l) / 2)
    
    if (len(l) % 2 == 1):
        return [l[mid], l[mid]]
    else:
        return l[mid-1:mid+1]

def viz_fmaps(name, fmaps):
    b, h, w, c = np.shape(fmaps)
    fmaps = np.transpose(fmaps, [0,3,1,2])
    [nrows, ncols] = factors(b * c)
    fmaps = np.reshape(fmaps, (nrows, ncols, h, w))

    for ii in range(nrows):
        for jj in range(ncols):
            if jj == 0:
                row = fmaps[ii][jj]
            else:
                row = np.concatenate((row, fmaps[ii][jj]), axis=1)
                
        if ii == 0:
            img = row
        else:
            img = np.concatenate((img, row), axis=0)
            
    plt.imsave(name, img, cmap='gray')

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

def get_val_filenames():
    val_filenames = []

    print ("building validation dataset")

    for subdir, dirs, files in os.walk(val_path):
        for file in files:
            val_filenames.append(os.path.join(val_path, file))

    np.random.shuffle(val_filenames)    

    remainder = len(val_filenames) % args.batch_size
    val_filenames = val_filenames[:(-remainder)]

    return val_filenames
    
def get_train_filenames():
    train_filenames = []

    print ("building training dataset")

    for subdir, dirs, files in os.walk(train_path):
        for file in files:
            train_filenames.append(os.path.join(train_path, file))
    
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

    # means = tf.reshape(tf.constant(MEAN), [1, 1, 1, 3])
    # image = image - means

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
labels = tf.one_hot(labels, depth=1000)

X = tf.concat((features, -1. * features), axis=3) / 255.
Y = labels

train_iterator = train_dataset.make_initializable_iterator()
val_iterator = val_dataset.make_initializable_iterator()

###############################################################

batch_size = tf.placeholder(tf.int32, shape=())
dropout_rate = tf.placeholder(tf.float32, shape=())
lr = tf.placeholder(tf.float32, shape=())

###############################################################

if args.model == 'vgg':
    model = VGGNet64(batch_size=batch_size, dropout_rate=dropout_rate, init=args.init)
elif args.model == 'tiny':
    model = VGGNetTiny(batch_size=batch_size, dropout_rate=dropout_rate, init=args.init)
elif args.model == 'mobile':
    model = MobileNet64(batch_size=batch_size, dropout_rate=dropout_rate, init=args.init)
else:
    assert (False)

###############################################################

predict = tf.nn.softmax(model.predict(X=X))
weights = model.get_weights()

bp_gvs, bp_derivs = model.gvs(X=X, Y=Y)
ss_gvs, ss_derivs = model.ss_gvs(X=X, Y=Y)

train = tf.train.AdamOptimizer(learning_rate=lr, epsilon=args.eps).apply_gradients(grads_and_vars=ss_gvs)

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

    print('epoch %d/%d' % (ii, args.epochs))

    sess.run(train_iterator.initializer, feed_dict={filename: train_filenames})

    train_total = 0.0
    train_correct = 0.0
    train_top5 = 0.0
    
    for jj in range(0, len(train_filenames), args.batch_size):
        if (jj % (100 * args.batch_size) == 0):
            [bp_gv, ss_gv, ss_deriv, bp_deriv, _total_correct, _total_top5, _] = sess.run([bp_gvs, ss_gvs, ss_derivs, bp_derivs, total_correct, total_top5, train], feed_dict={handle: train_handle, batch_size: args.batch_size, dropout_rate: args.dropout, lr: lr_decay})
        else:
            [_total_correct, _total_top5, _] = sess.run([total_correct, total_top5, train], feed_dict={handle: train_handle, batch_size: args.batch_size, dropout_rate: args.dropout, lr: lr_decay})

        train_total += args.batch_size
        train_correct += _total_correct
        train_top5 += _total_top5
        
        train_acc = train_correct / train_total
        train_acc_top5 = train_top5 / train_total
        
        if (jj % (100 * args.batch_size) == 0):
            # gradients
            num_gv = len(ss_gv)
            angles_gv = [None] * num_gv
            matches_gv = [None] * num_gv
            for kk in range(num_gv):
                angles_gv[kk] = deque(maxlen=250)
                matches_gv[kk] = deque(maxlen=250)

            for kk in range(num_gv):
                ss = np.reshape(ss_gv[kk], -1)
                bp = np.reshape(bp_gv[kk], -1)
                angle = angle_between(ss, bp) * (180. / 3.14)
                match = np.count_nonzero(np.sign(ss) == np.sign(bp)) / np.prod(np.shape(ss))
                angles_gv[kk].append(angle)
                matches_gv[kk].append(match)

            # derivatives
            num_deriv = len(ss_deriv)
            angles_deriv = [None] * num_deriv
            matches_deriv = [None] * num_deriv
            for kk in range(num_deriv):
                angles_deriv[kk] = deque(maxlen=250)
                matches_deriv[kk] = deque(maxlen=250)

            for kk in range(num_deriv):
                for ll in range(args.batch_size):
                    ss = np.reshape(ss_deriv[kk][ll], -1)
                    bp = np.reshape(bp_deriv[kk][ll], -1)
                    angle = angle_between(ss, bp) * (180. / 3.14)
                    match = np.count_nonzero(np.sign(ss) == np.sign(bp)) / np.prod(np.shape(ss))
                    angles_deriv[kk].append(angle)
                    matches_deriv[kk].append(match)

            p = "train accuracy: %f %f" % (train_acc, train_acc_top5)
            print (p)
            f = open(results_filename, "a")
            f.write(p + "\n")
            f.close()

            angles_gv = np.average(angles_gv, axis=1)               
            matches_gv = np.average(matches_gv, axis=1) * 100.      
            angles_deriv = np.average(angles_deriv, axis=1)         
            matches_deriv = np.average(matches_deriv, axis=1) * 100.

            print ('gv angles',     int(np.max(angles_gv)),     int(np.average(angles_gv)),     int(np.min(angles_gv)))
            print ('gv matches',    int(np.max(matches_gv)),    int(np.average(matches_gv)),    int(np.min(matches_gv)))
            print ('deriv angles',  int(np.max(angles_deriv)),  int(np.average(angles_deriv)),  int(np.min(angles_deriv)))
            print ('deriv matches', int(np.max(matches_deriv)), int(np.average(matches_deriv)), int(np.min(matches_deriv)))

    train_accs.append(train_acc)
    train_accs_top5.append(train_acc_top5)
    
    ##################################################################

    sess.run(val_iterator.initializer, feed_dict={filename: val_filenames})
    
    val_total = 0.0
    val_correct = 0.0
    val_top5 = 0.0
    
    for jj in range(0, len(val_filenames), args.batch_size):
        [_total_correct, _top5] = sess.run([total_correct, total_top5], feed_dict={handle: val_handle, batch_size: args.batch_size, dropout_rate: 0.0, lr: 0.0})
        
        val_total += args.batch_size
        val_correct += _total_correct
        val_top5 += _top5
        
        val_acc = val_correct / val_total
        val_acc_top5 = val_top5 / val_total
        
        if (jj % (100 * args.batch_size) == 0):
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
            lr_decay = 0.1 * args.lr
            phase = 2
    elif phase == 2:
        dacc = val_accs[-1] - val_accs[-2]
        if dacc <= 0.005:
            lr_decay = 0.05 * args.lr
            phase = 3

    p = "phase: %d" % (phase)
    print (p)
    f = open(results_filename, "a")
    f.write(p + "\n")
    f.close()

    [w] = sess.run([weights], feed_dict={})
    np.save(args.name, w)
    








