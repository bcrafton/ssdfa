
import argparse
import os
import sys

##############################################

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=1000)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--eps', type=float, default=1.)
parser.add_argument('--dropout', type=float, default=0.0)
parser.add_argument('--bias', type=float, default=0.)
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--dfa', type=int, default=0)
parser.add_argument('--sparse', type=int, default=0)
parser.add_argument('--rank', type=int, default=0)
parser.add_argument('--init', type=str, default="glorot_normal")
parser.add_argument('--save', type=int, default=0)
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
from lib.MaxPool import MaxPool
from lib.Dropout import Dropout
from lib.FeedbackFC import FeedbackFC
from lib.FeedbackConv import FeedbackConv

from lib.Activation import Relu
from lib.Activation import Tanh

from lib.cifar_models import cifar_conv
from lib.cifar_models import cifar_fc

##############################################

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

train_examples = 50000
test_examples = 10000

assert(np.shape(x_train) == (50000, 32, 32, 3))
x_train = x_train / np.std(x_train, axis=0, keepdims=True)
x_train = np.concatenate((x_train, -1. * x_train), axis=3)
y_train = keras.utils.to_categorical(y_train, 10)

assert(np.shape(x_test) == (10000, 32, 32, 3))
x_test = x_test / np.std(x_test, axis=0, keepdims=True)
x_test = np.concatenate((x_test, -1. * x_test), axis=3)
y_test = keras.utils.to_categorical(y_test, 10)

def unit_vector(vector):
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

##############################################

tf.set_random_seed(0)
tf.reset_default_graph()

batch_size = tf.placeholder(tf.int32, shape=())
dropout_rate = tf.placeholder(tf.float32, shape=())
lr = tf.placeholder(tf.float32, shape=())

X = tf.placeholder(tf.float32, [None, 32, 32, 6])
Y = tf.placeholder(tf.float32, [None, 10])

model = cifar_conv(batch_size=batch_size, dropout_rate=dropout_rate, init='glorot_uniform')
# model = cifar_fc(batch_size=batch_size, dropout_rate=dropout_rate)

##############################################
predict = model.predict(X=X)
weights = model.get_weights()
##############################################
#'''
if args.dfa:
    grads_and_vars = model.dfa_gvs(X=X, Y=Y)
else:
    grads_and_vars = model.gvs(X=X, Y=Y)

[d1, db1, c3, cg3, cb3, c2, cg2, cb2, c1, cg1, cb1] = grads_and_vars
#'''
##############################################
#'''
params = tf.trainable_variables()
loss = tf.nn.softmax_cross_entropy_with_logits_v2(labels=Y, logits=predict)
grads = tf.gradients(loss, params)
grads_and_vars_bp = zip(grads, params)
[c1_bp, cg1_bp, cb1_bp, c2_bp, cg2_bp, cb2_bp, c3_bp, cg3_bp, cb3_bp, d1_bp, db1_bp] = grads_and_vars_bp
#'''
##############################################
train1 = tf.train.AdamOptimizer(learning_rate=lr, epsilon=args.eps).apply_gradients(grads_and_vars=[c1, cg1, cb1, d1, db1])
train2 = tf.train.AdamOptimizer(learning_rate=lr, epsilon=args.eps).apply_gradients(grads_and_vars=[c2, cg2, cb2, d1, db1])
train3 = tf.train.AdamOptimizer(learning_rate=lr, epsilon=args.eps).apply_gradients(grads_and_vars=[c3, cg3, cb3, d1, db1])
train4 = tf.train.AdamOptimizer(learning_rate=lr, epsilon=args.eps).apply_gradients(grads_and_vars=[c1, cg1, cb1, c2, cg2, cb2, c3, cg3, cb3, d1, db1])

correct = tf.equal(tf.argmax(predict,1), tf.argmax(Y,1))
sum_correct = tf.reduce_sum(tf.cast(correct, tf.float32))
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
    
    total_correct = 0
    matches = []
    angles = []
    for jj in range(0, train_examples, args.batch_size):
        s = jj
        e = min(jj + args.batch_size, train_examples)
        b = e - s
        
        xs = x_train[s:e]
        ys = y_train[s:e]
        
        ######################################################

        '''
        [params] = sess.run([params], feed_dict={batch_size: b, dropout_rate: 0.0, lr: args.lr, X: xs, Y: ys})
        for p in params:
            print (np.shape(p))
        assert(False)
        '''

        '''
        [gvs] = sess.run([grads_and_vars], feed_dict={batch_size: b, dropout_rate: 0.0, lr: args.lr, X: xs, Y: ys})
        for gv in grads_and_vars:
            print (np.shape(gv[1]))
        assert(False)
        '''

        ######################################################

        '''
        if ii < 10:
            _sum_correct, _ = sess.run([sum_correct, train1], feed_dict={batch_size: b, dropout_rate: 0.0, lr: args.lr, X: xs, Y: ys})
        elif ii < 20:
            _sum_correct, _ = sess.run([sum_correct, train2], feed_dict={batch_size: b, dropout_rate: 0.0, lr: args.lr, X: xs, Y: ys})
        elif ii < 30:
            _sum_correct, _ = sess.run([sum_correct, train3], feed_dict={batch_size: b, dropout_rate: 0.0, lr: args.lr, X: xs, Y: ys})
        else:
            _sum_correct, _ = sess.run([sum_correct, train4], feed_dict={batch_size: b, dropout_rate: 0.0, lr: args.lr, X: xs, Y: ys})
        '''

        '''
        if (ii % 20) < 5:
            _sum_correct, _ = sess.run([sum_correct, train1], feed_dict={batch_size: b, dropout_rate: 0.0, lr: args.lr, X: xs, Y: ys})
        elif (ii % 20) < 10:
            _sum_correct, _ = sess.run([sum_correct, train2], feed_dict={batch_size: b, dropout_rate: 0.0, lr: args.lr, X: xs, Y: ys})
        else:
            _sum_correct, _ = sess.run([sum_correct, train3], feed_dict={batch_size: b, dropout_rate: 0.0, lr: args.lr, X: xs, Y: ys})
        '''

        _sum_correct, _ = sess.run([sum_correct, train4], feed_dict={batch_size: b, dropout_rate: 0.0, lr: args.lr, X: xs, Y: ys})

        [ss, bp] = sess.run([c1, c1_bp], feed_dict={batch_size: b, dropout_rate: 0.0, lr: 0.0, X: xs, Y: ys})
        ss = ss[0]
        bp = bp[0]
        top = np.sum(np.sign(ss) == np.sign(bp))
        bot = np.prod(np.shape(bp))
        match = top / bot
        matches.append(match)
        # print (np.shape(ss), np.shape(bp))
        # print (match, top, bot)
        ss = np.reshape(ss, [b, -1])
        bp = np.reshape(bp, [b, -1])

        for kk in range(b):
            angle = angle_between(ss[kk], bp[kk]) * (180. / 3.14)
            angles.append(angle)

        # print (np.average(angles), np.average(matches))

        ######################################################

        total_correct += _sum_correct

        ######################################################

    train_acc = 1.0 * total_correct / (train_examples - (train_examples % args.batch_size))
    train_accs.append(train_acc)

    #############################

    total_correct = 0
    for jj in range(0, test_examples, args.batch_size):
        s = jj
        e = min(jj + args.batch_size, test_examples)
        b = e - s
        
        xs = x_test[s:e]
        ys = y_test[s:e]
        
        _sum_correct = sess.run(sum_correct, feed_dict={batch_size: b, dropout_rate: 0.0, lr: 0.0, X: xs, Y: ys})
        total_correct += _sum_correct
        
    test_acc = 1.0 * total_correct / (test_examples - (test_examples % args.batch_size))
    test_accs.append(test_acc)
    
    #############################
            
    p = "%d | train acc: %f | test acc: %f | sign match: %f | angle: %f" % (ii, train_acc, test_acc, np.average(matches), np.average(angles))
    print (p)
    f = open(filename, "a")
    f.write(p + "\n")
    f.close()

##############################################

if args.save:
    [w] = sess.run([weights], feed_dict={})
    w['train_acc'] = train_accs
    w['test_acc'] = test_accs
    np.save(args.name, w)
    
##############################################


