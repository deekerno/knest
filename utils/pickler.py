# UCF Senior Design 2017-18
# Group 38

import cv2
import os
import numpy as np
import pandas as pd

IMAGE_DIR = '/users/ayylmao/desktop/dataset/'
LABELS = ['bird', 'not_bird']
NUM_IMAGES = 6087
IMAGES = 'dataset.csv'
IMAGE_SIZE = 448


def compile_data():
	image_shape = (IMAGE_SIZE, IMAGE_SIZE, 3)
	x = np.zeros((NUM_IMAGES,) + image_shape, dtype=np.uint8)
	y = np.zeros((NUM_IMAGES, len(LABELS)), dtype=np.uint8)
	index = 0

	for row in images.itertuples(index=False):
		filename = images.iloc[index, 0]
		label = images.iloc[index, 1]
		x[index] = cv2.imread(os.path.join(IMAGE_DIR, filename))
		y[index, label] = 1
		index += 1

	return x, y


# Read the image dataset into a pandas dataframe
images = pd.read_csv(IMAGES)

x, y = compile_data()

data = {'x' : x, 'y' : y}

with open('leavens.pkl', 'wb') as f:
	pickle.dump(data, f)
