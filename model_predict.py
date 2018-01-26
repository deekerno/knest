# UCF Senior Design 2017-18
# Group 38

from model import Model
import tflearn
import pickle
import json
import os
import scipy as sc
import numpy as np
import argparse
from PIL import Image
import glob

# Your chosen hyperparamters should be contained in a JSON
HYPERPARAMETERS = "config/ad_hp.json"

# Decode into a JSON object
with open(HYPERPARAMETERS) as json_file:
    hp_content = json.load(json_file)

# Instantiate a model object with the hyperparameters
test = Model('CNNTest', hp_content=hp_content)

# Build the network through TFLearn
model = tflearn.DNN(test.network, tensorboard_verbose=3,
                    checkpoint_path=os.path.join(test.checkpoint_dir, 'test.ckpt'))

model.load("saved/train1/bird-classifier.tfl")

for filename in glob.glob('leavens/*.jpg'):
    img = Image.open(filename)
    img = sc.ndimage.imread(filename, mode="RGB")
    img = sc.misc.imresize(img, (32,32), interp="bicubic").astype(np.float32, casting='unsafe')
    prediction = model.predict([img])
    is_bird = np.argmax(prediction[0]) == 1
    if is_bird:
        print("found bird!")
    else:
        print("bird not detected!")