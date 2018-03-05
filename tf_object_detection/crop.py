from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import scipy as sp
import os

# crop BB from image
def crop(image, BB):
	x0 = (BB[0] * 5184).astype(int)
	if x0 < 0:
		x0 = 0
	y0 = (BB[1] * 3456).astype(int)
	if y0 < 0:
		y0 = 0
	width = (BB[2] * 5184).astype(int)
	height = (BB[3] * 3456).astype(int)
	# target 4 has off image coordinates?
	return image[y0:y0+height , x0:x0+width, :]

# image size 5184x3456
image = Image.open('image1.jpg')
convert = np.asarray(image)

# Box coordinates are normalized
BB = np.loadtxt('BBtest.txt', delimiter=' ')

# crop 1st BB
bird = crop(convert, BB[0])

plt.imshow(bird)
plt.show()
