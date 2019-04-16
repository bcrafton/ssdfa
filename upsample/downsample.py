
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

#####

x = range(4 * 4)
x = np.reshape(x, (16, 1))
x = np.concatenate((x, x, x, x), axis=1)
x = np.reshape(x, (4, 4, 2, 2))
x = np.reshape(x, (4, 8, 2))
x = np.transpose(x, (0, 2, 1))
x = np.reshape(x, (8, 8))
print (x)

x = np.reshape(x, (4, 2, 4, 2))
x = np.transpose(x, (0, 2, 1, 3))
x = np.reshape(x, (4, 4, 4))

####

image = Image.open('laska.png')
image.load()
image = np.asarray(image, dtype="float32")
image = image[0:225, 0:225, 0:3]

x = np.copy(image)
print (np.shape(x), np.prod(np.shape(x)))

x = np.reshape(x, (225*225, 1, 3))
x = np.concatenate((x, x, x, x), axis=1)
x = np.reshape(x, (225, 225, 2, 2, 3))
x = np.reshape(x, (225, 450, 2, 3))
x = np.transpose(x, (0, 2, 1, 3))
x = np.reshape(x, (450, 450, 3))

'''
img = x / 255.
plt.imshow(img)
plt.show()
'''

x = np.reshape(x, (225, 2, 225, 2, 3))
x = np.transpose(x, (0, 2, 1, 3, 4))
x = np.reshape(x, (225, 225, 4, 3))
x = np.mean(x, axis=2)

img = x / 255.
plt.imshow(img)
plt.show()
