# UCF Senior Design 2017-18
# Group 38

from sklearn.model_selection import train_test_split as tts
import squeezenet
import os
import h5py
import tflearn

DATASET = 'leavens.h5'
OUTPUT_FOLDER = 'output/squeezenet_448/'
SEED = 552353

# Create the "buffed" bobo architecture.
sq = squeezenet.Model((448, 448), 2)

# Build the model.
model = tflearn.DNN(sq.network, checkpoint_path=OUTPUT_FOLDER,
                    max_checkpoints=10, tensorboard_verbose=3,
                    clip_gradients=0.)

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

# Train the network.
model.fit(X, Y, n_epoch=100, validation_set=(X_test, Y_test),
          show_metric=True, batch_size=30, shuffle=True,
          run_id='squeezenet_448')

# Save the weights.
model_name = "squeezenet_448"  + ".tfl"
model_path = os.path.join(OUTPUT_FOLDER, model_name)
model.save(model_path)
