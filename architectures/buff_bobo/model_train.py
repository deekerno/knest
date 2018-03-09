# UCF Senior Design 2017-18
# Group 38

from sklearn.model_selection import train_test_split as tts
import architectures.buff_bobo.buff_bobo
import os
import pickle
import tflearn
import yaml

DATASET = 'leavens.pkl'
OUTPUT_FOLDER = '/output/buff_bobo'
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
bb = buff_bobo.BuffBobo()

# Build the model.
model = tflearn.DNN(bb.network, checkpoint_path=OUTPUT_FOLDER,
                    max_checkpoints=10, tensorboard_verbose=3,
                    clip_gradients=0.)

# Load the dataset.
with open(DATASET, 'rb') as f:
    pickle = pickle.load(f, encoding='latin1')

# Take the dataset representation, and load into arrays and labels.
x = pickle['x']
y = pickle['y']

# Split the dataset into training and testing sets.
(X, X_test, Y, Y_test) = tts(x, y, test_size=0.3, random_state=SEED)

# Shuffle the data in place.
X, Y = tflearn.data_utils.shuffle(X, Y)

# Train the network.
model.fit(X, Y, n_epoch=200, validation_set=(X_test, Y_test),
          show_metric=True, batch_size=32, shuffle=True,
          run_id='buff_bobo')

# Save the weights.
model_name = "buff_bobo"  + ".tfl"
model_path = os.path.join(OUTPUT_FOLDER, model_name)
model.save(model_path)
