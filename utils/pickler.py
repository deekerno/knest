# UCF Senior Design 2017-18
# Group 38

import cv2
import os
import numpy as np
import pandas as pd
import pickle

IMAGE_DIR = '/users/ad/project/knest/dataset/'
LABELS = ['bird', 'not_bird']
NUM_IMAGES = 6087
IMAGES = 'dataset.csv'
IMAGE_SIZE = 448
FINAL_DATASET = 'leavens2.pkl'

def compile_data():
    """
        Returns two arrays, x and y, that are used to create a pickle file.
    """

    # Set the image shape along with the 3 color dimensions.
    image_shape = (IMAGE_SIZE, IMAGE_SIZE, 3)

    # Create an array of sub-arrays with the same shape as the image.
    x = np.zeros((NUM_IMAGES,) + image_shape, dtype=np.float32)

    # Create an array of sub-arrays that contain an element for each label.
    y = np.zeros((NUM_IMAGES, len(LABELS)), dtype=np.float32)

    index = 0

    # For each row in the dataframe, get the filename and label. A label of
    # 0 is bird, 1 is not_bird. Then, load the array representaion of the image
    # into x and fill the corresponding image label with a 1.
    for row in images.itertuples(index=False):
        filename = images.iloc[index, 0]
        label = images.iloc[index, 1]
        x[index] = cv2.imread(os.path.join(IMAGE_DIR, filename))
        y[index, label] = 1
        index += 1

    return x, y


# Read the image dataset into a pandas dataframe.
images = pd.read_csv(IMAGES)

x, y = compile_data()

data = {'x': x, 'y': y}

bytes_out = pickle.dumps(data, protocol=4)
max_bytes = 2**31 - 1
n_bytes = 2**31

with open(FINAL_DATASET, 'wb') as f:
    for idx in range(0, len(bytes_out), max_bytes):
            f.write(bytes_out[idx:idx+max_bytes])
    #pickle.dump(data, f, protocol=4)
    
