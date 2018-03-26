# UCF Senior Design 2017-18
# Group 38

import cv2
import os
import numpy as np
import h5py
from sklearn.model_selection import train_test_split as tts

BIRD_DIR = '/users/ad/projects/knest/crops'
NONBIRD_DIR = '/users/ad/projects/knest/nonbird_crops'
LABELS = ['bird', 'not_bird']
IMAGE_SIZE = 400
FINAL_DATASET = 'leavens_real.h5'
NUM_IMAGES = 3848
SEED = 418791


def gen_resized_list(folder):
    resized_images = []
    images = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith('.JPG')]
    for image in images:
        img = cv2.imread(image)
        resized_image = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
        resized_images.append(resized_image)

    return resized_images


def compile_data(birds, nonbirds):

    image_shape = (IMAGE_SIZE, IMAGE_SIZE, 3)

    x = np.zeros((NUM_IMAGES,) + image_shape)
    y = np.zeros((NUM_IMAGES, len(LABELS)))

    index = 0

    for bird in birds:
        x[index] = bird
        y[index, 0] = 1
        index += 1

    for nonbird in nonbirds:
        x[index] = nonbird
        y[index, 1] = 1
        index += 1

    return x, y


birds = gen_resized_list(BIRD_DIR)
nonbirds = gen_resized_list(NONBIRD_DIR)

print("Compiling dataset...")

x, y = compile_data(birds, nonbirds)

print("Dataset compiled.")

print("Splitting dataset...")

(X, X_test, Y, Y_test) = tts(x, y, test_size=0.2, random_state=SEED)

print("Dataset split.")

h5f = h5py.File(FINAL_DATASET, 'w')
h5f.create_dataset('X', data=X)
h5f.create_dataset('Y', data=Y)
h5f.create_dataset('X_test', data=X_test)
h5f.create_dataset('Y_test', data=Y_test)
h5f.close()
