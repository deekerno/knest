# UCF Senior Design 2017-18
# Group 38

from test_model import TestModel
import tflearn
import pickle
import json

# Your chosen hyperparamters should be contained in a JSON
HYPERPARAMETERS = "config/ad_hp.json"

# Decode into a JSON object
with open(HYPERPARAMETERS) as json_file:
    hp_content = json.load(json_file)

# Instantiate a model object with the hyperparameters
test = TestModel('CNNTest', hp_content=hp_content)

# Build the network through TFLearn
model = tflearn.DNN(test.network, tensorboard_verbose=3,
                    checkpoint_path='test.ckpt')

# Open up the pickled dataset (although this may change with TFRecord stuff)
with open('full_dataset.pkl', 'rb') as f:
    X, Y, X_test, Y_test = pickle.load(f, encoding='latin1')

# Shuffle the data in unison
X, Y = tflearn.data_utils.shuffle(X, Y)

# Train the network for however many epochs you feel necessary
model.fit(X, Y, n_epoch=test.hp.num_epochs, shuffle=True,
          validation_set=(X_test, Y_test),
          show_metric=True, batch_size=test.hp.batch_size,
          run_id='bird-classifier')

# Save the model
model.save("bird-classifier.tfl")
