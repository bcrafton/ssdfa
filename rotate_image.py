
import tensorflow as tf
import keras
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

img = x_train[0]
img = sp.ndimage.rotate(img, 30.)

plt.imshow(img)
plt.show()
