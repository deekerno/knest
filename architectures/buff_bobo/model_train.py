# UCF Senior Design 2017-18
# Group 38

from sklearn.model_selection import train_test_split as tts
import buffed_resnet
import os
import h5py
import tflearn
import yaml

DATASET = 'leavens_resized_112.h5'
OUTPUT_FOLDER = 'output/buff_bobo_resnet_112'
SEED = 798547

# Residual blocks
# 32 layers: n=5, 56 layers: n=9, 110 layers: n=18
n = 5

# Real-time data preprocessing.
img_prep = tflearn.ImagePreprocessing()
img_prep.add_featurewise_zero_center(per_channel=True)

# Real-time data augmentation.
img_aug = tflearn.ImageAugmentation()
img_aug.add_random_flip_leftright()

# Create the "buffed" bobo architecture.
bb = buffed_resnet.BuffBobo(n, img_aug, img_prep)

# Build the model.
model = tflearn.DNN(bb.network, checkpoint_path=OUTPUT_FOLDER,
                    max_checkpoints=10, tensorboard_verbose=3,
                    clip_gradients=0.)

# Load the dataset.
h5f = h5py.File(DATASET, 'r')

# Take the dataset representation, and load into arrays and labels.
X = h5f['X']
Y = h5f['Y']
X_test = h5f['X_test']
Y_test = h5f['Y_test']

# Split the dataset into training and testing sets.
#(X, X_test, Y, Y_test) = tts(x, y, test_size=0.3, random_state=SEED)

# Shuffle the data in place.
X, Y = tflearn.data_utils.shuffle(X, Y)

# Train the network.
model.fit(X, Y, n_epoch=100, validation_set=(X_test, Y_test),
          show_metric=True, batch_size=32, shuffle=True,
          run_id='buff_bobo')

# Save the weights.
model_name = "buff_bobo_resnet_112"  + ".tfl"
model_path = os.path.join(OUTPUT_FOLDER, model_name)
model.save(model_path)
