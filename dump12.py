
import numpy as np
import tensorflow as tf

(x_train, y_train), (_, _) = tf.keras.datasets.mnist.load_data()

ones = np.where(y_train == 1)
twos = np.where(y_train == 2)

ones_img = x_train[ones]
twos_img = x_train[twos]

np.save('ones', ones_img)
np.save('twos', twos_img)
