from __future__ import division
from __future__ import print_function
from __future__ import absolute_import

import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
from tflearn.data_preprocessing import ImagePreprocessing
from tflearn.data_augmentation import ImageAugmentation

import numpy as np


OUTPUT_FOLDER = 'output/'


class ClassificationModel(object):
    """
        A model for object classification.
    """

    def __init__(self, weights):
        self.network = self._build_network()
        self.model = tflearn.DNN(self.network, checkpoint_path=OUTPUT_FOLDER,
                                 max_checkpoints=10, tensorboard_verbose=3,
                    clip_gradients=0.)
        self.model.load(weights, weights_only=True)

    def _build_network(self):

        cnet = input_data(shape=[None, 32, 32, 3])
        cnet = conv_2d(cnet, 32, 3, activation='relu')
        cnet = max_pool_2d(cnet, 2)
        cnet = conv_2d(cnet, 64, 3, activation='relu')
        cnet = conv_2d(cnet, 64, 3, activation='relu')
        cnet = max_pool_2d(cnet, 2)
        cnet = fully_connected(cnet, 1024, activation='relu')
        cnet = dropout(cnet, 0.5)
        cnet = fully_connected(cnet, 2, activation='softmax')
        cnet = regression(cnet, optimizer='adam',
                          loss='categorical_crossentropy',
                          learning_rate=0.001)

        return cnet

    def predict(self, image):
        """
            Wraps the TFLearn model prediction function. Returns the
            predicted probabilites in an array.
        """
        return self.model.predict([image])

    def classify(self, prediction):

        is_bird = np.argmax(prediction[0]) == 1
        return is_bird