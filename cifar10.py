
import argparse
import os
import sys

##############################################

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=5)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--eps', type=float, default=1.)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--init', type=str, default="glorot_uniform")
parser.add_argument('--save', type=int, default=1)
parser.add_argument('--name', type=str, default="cifar10_conv")
parser.add_argument('--load', type=str, default=None)
args = parser.parse_args()

if args.gpu >= 0:
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpu)

##############################################

import tensorflow as tf
import keras
import numpy as np

from lib.Model import Model

from lib.Layer import Layer 
from lib.ConvToFullyConnected import ConvToFullyConnected
from lib.FullyConnected import FullyConnected
from lib.Convolution import Convolution
from lib.Activation import Relu

from lib.cifar_models import cifar_conv
# from lib.cifar_models import cifar_conv_bn

##############################################

def quantize_activations(a):
  # scale = (15 - 0) / (np.percentile(a, 95) - np.percentile(a, 5))
  scale = (15 - 0) / (np.max(a) - np.min(a))
  # scale = (15 - 0) / (2 * np.std(a))

  a = scale * a
  a = np.floor(a)
  a = np.clip(a, 0, 15)
  return a, scale

##############################################

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

train_examples = 50000
test_examples = 10000

assert(np.shape(x_train) == (50000, 32, 32, 3))
# x_train = x_train - np.mean(x_train, axis=0, keepdims=True)
# x_train = x_train / np.std(x_train, axis=0, keepdims=True)
x_train, _ = quantize_activations(x_train)
y_train = keras.utils.to_categorical(y_train, 10)

assert(np.shape(x_test) == (10000, 32, 32, 3))
# x_test = x_test - np.mean(x_test, axis=0, keepdims=True)
# x_test = x_test / np.std(x_test, axis=0, keepdims=True)
x_test, _ = quantize_activations(x_test)
y_test = keras.utils.to_categorical(y_test, 10)

##############################################

tf.set_random_seed(0)
tf.reset_default_graph()

batch_size = tf.placeholder(tf.int32, shape=())
lr = tf.placeholder(tf.float32, shape=())

X = tf.placeholder(tf.float32, [None, 32, 32, 3])
Y = tf.placeholder(tf.float32, [None, 10])

scale = tf.placeholder(tf.float32, [8])

model = cifar_conv(batch_size=batch_size, scale=scale)

##############################################

predict = model.predict(X=X)
predict1 = model.predict1(X=X)
predict2 = model.predict2(X=X)

weights = model.get_weights()

grads_and_vars = model.gvs(X=X, Y=Y)
train = tf.train.AdamOptimizer(learning_rate=lr, epsilon=args.eps).apply_gradients(grads_and_vars=grads_and_vars)

correct = tf.equal(tf.argmax(predict,1), tf.argmax(Y,1))
total_correct = tf.reduce_sum(tf.cast(correct, tf.float32))

correct2 = tf.equal(tf.argmax(predict2,1), tf.argmax(Y,1))
total_correct2 = tf.reduce_sum(tf.cast(correct2, tf.float32))

##############################################

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()
tf.local_variables_initializer().run()

##############################################

filename = args.name + '.results'
f = open(filename, "w")
f.write(filename + "\n")
f.write("total params: " + str(model.num_params()) + "\n")
f.close()

##############################################

train_accs = []
test_accs = []

for ii in range(args.epochs):

    #############################
    
    _total_correct = 0
    for jj in range(0, train_examples, args.batch_size):    
        s = jj
        e = min(jj + args.batch_size, train_examples)
        b = e - s
        
        xs = x_train[s:e]
        ys = y_train[s:e]
        
        _correct, _ = sess.run([total_correct, train], feed_dict={batch_size: b, lr: args.lr, X: xs, Y: ys})
        _total_correct += _correct
  
    train_acc = 1.0 * _total_correct / (train_examples - (train_examples % args.batch_size))
    train_accs.append(train_acc)

    #############################

    _total_correct = 0
    for jj in range(0, test_examples, args.batch_size):
        s = jj
        e = min(jj + args.batch_size, test_examples)
        b = e - s
        
        xs = x_test[s:e]
        ys = y_test[s:e]
        
        _correct = sess.run(total_correct, feed_dict={batch_size: b, lr: 0.0, X: xs, Y: ys})
        _total_correct += _correct
        
    test_acc = 1.0 * _total_correct / (test_examples - (test_examples % args.batch_size))
    test_accs.append(test_acc)
    
    #############################
            
    p = "%d | train acc: %f | test acc: %f" % (ii, train_acc, test_acc)
    print (p)
    f = open(filename, "a")
    f.write(p + "\n")
    f.close()

##############################################

# ConvRelu:
# cache = (conv, conv_cache, relu, relu_cache)

# Dense:
# cache = (scale,)

scales = [[], [], [], [], [], [], [], []]

for jj in range(0, train_examples, args.batch_size):    
    s = jj
    e = min(jj + args.batch_size, train_examples)
    b = e - s
    
    xs = x_train[s:e]
    ys = y_train[s:e]
    
    A, cache = sess.run(predict1, feed_dict={batch_size: b, lr: 0., X: xs, Y: ys})

    conv1, conv2, conv3, conv4, conv5, conv6, conv7, conv2dense, dense = cache
    convs = [conv1, conv2, conv3, conv4, conv5, conv6, conv7]

    for kk in range(7):
        sw, sa = convs[kk][1], convs[kk][3]
        # print (sw, sa)
        scales[kk].append(sa)
    
    # print (dense)
    sa = dense
    scales[7].append(sa)

# print (np.shape(scales))
scale_np = np.mean(scales, axis=(1,2))
scale_np = 1. / np.floor(1. / scale_np)

##############################################

_total_correct = 0
for jj in range(0, test_examples, args.batch_size):
    s = jj
    e = min(jj + args.batch_size, test_examples)
    b = e - s
    
    xs = x_test[s:e]
    ys = y_test[s:e]
    
    _correct = sess.run(total_correct2, feed_dict={batch_size: b, lr: 0.0, X: xs, Y: ys, scale: scale_np})
    _total_correct += _correct
    
test_acc = 1.0 * _total_correct / (test_examples - (test_examples % args.batch_size))
test_accs.append(test_acc)

print (test_acc)

##############################################

[w] = sess.run([weights], feed_dict={})
'''
for key in w.keys():
    print (key, np.shape(weights[key]))
'''
w['train_acc'] = train_accs
w['test_acc'] = test_accs
np.save(args.name, w)

##############################################


