

import numpy as np
import tensorflow as tf
import argparse
import keras

#######################################

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--batch_size', type=int, default=50)
parser.add_argument('--lr', type=float, default=1e-2)
args = parser.parse_args()

LAYER1 = 784
LAYER2 = 100
LAYER3 = 10

TRAIN_EXAMPLES = 60000
TEST_EXAMPLES = 10000

#######################################

def softmax(x):
    return tf.nn.softmax(x)

def relu(x):
    return tf.nn.relu(x)

def drelu(x):
    return tf.cast(x > 0.0, dtype=tf.float32)

#######################################

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

y_train = keras.utils.to_categorical(y_train, 10)
x_train = x_train.reshape(TRAIN_EXAMPLES, 784)
x_train = x_train.astype('float32')
x_train = x_train / np.max(x_train)

y_test = keras.utils.to_categorical(y_test, 10)
x_test = x_test.reshape(TEST_EXAMPLES, 784)
x_test = x_test.astype('float32')
x_test = x_test / np.max(x_test)

#######################################

X = tf.placeholder(tf.float32, [None, 784])
Y = tf.placeholder(tf.float32, [None, 10])
A1 = X

with tf.device('/device:GPU:0'):
    high = 1. / np.sqrt(LAYER1)
    weights1_init = np.random.uniform(low=-high, high=high, size=(LAYER1, LAYER2))
    weights1 = tf.Variable(weights1_init, dtype=tf.float32)
    Z2 = tf.matmul(A1, weights1)
    A2 = relu(Z2)

with tf.device('/device:GPU:1'):
    high = 1. / np.sqrt(LAYER2)
    weights2_init = np.random.uniform(low=-high, high=high, size=(LAYER2, LAYER3))
    weights2 = tf.Variable(weights2_init, dtype=tf.float32)
    Z3 = tf.matmul(A2, weights2)
    A3 = softmax(Z3)

    E = tf.subtract(A3, Y)

D3 = E
D2 = tf.matmul(D3, tf.transpose(weights2)) * drelu(A2)

DW2 = tf.matmul(tf.transpose(A2), D3) 
DW1 = tf.matmul(tf.transpose(A1), D2)  

train2 = weights2 - args.lr * DW2
train1 = weights1 - args.lr * DW1

correct = tf.equal(tf.argmax(A3,1), tf.argmax(Y,1))
total_correct = tf.reduce_sum(tf.cast(correct, tf.float32))

#######################################

config = tf.ConfigProto(allow_soft_placement=True)
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)
sess.run(tf.global_variables_initializer())

#######################################

for epoch in range(args.epochs):
    print ("epoch: %d/%d" % (epoch, args.epochs))
    
    train_correct = 0
    for ex in range(0, TRAIN_EXAMPLES, args.batch_size):
        start = ex 
        stop = ex + args.batch_size
    
        xs = x_train[start:stop]
        ys = y_train[start:stop]
        
        [correct, _, _] = sess.run([total_correct, train1, train2], feed_dict={X: xs, Y: ys})
        train_correct += correct
        
    print (train_correct * 1.0 / TRAIN_EXAMPLES)
        
    
    
    
