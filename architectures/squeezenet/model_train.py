# UCF Senior Design 2017-18
# Group 38

import h5py
import os
import squeezenet
import tflearn

CONFIG = 'config/squeezenet/config.yaml'
DATASET = 'leavens_real.h5'
OUTPUT_FOLDER = 'output/squeezenet/'
SEED = 552353

# Load the dataset.
print("Loading dataset...")
h5f = h5py.File(DATASET, 'r')

# Take the dataset representation, and load into arrays and labels.
X = h5f['X']
Y = h5f['Y']
X_test = h5f['X_test']
Y_test = h5f['Y_test']

# Shuffle the data in place.
print("Shuffling dataset...")
X, Y = tflearn.data_utils.shuffle(X, Y)

# Create the SqueezeNet architecture using defined hyperparameters.
sq = squeezenet.Model(CONFIG)

# Build the model.
model = tflearn.DNN(sq.network, checkpoint_path=OUTPUT_FOLDER,
                    max_checkpoints=10, tensorboard_verbose=3,
                    clip_gradients=0.)

# Train the network.
model.fit(X, Y, n_epoch=100, validation_set=(X_test, Y_test),
          show_metric=True, batch_size=30, shuffle=True,
          run_id='squeezenet')

# Save the weights.
model_name = "squeezenet" + ".tfl"
model_path = os.path.join(OUTPUT_FOLDER, model_name)
model.save(model_path)
