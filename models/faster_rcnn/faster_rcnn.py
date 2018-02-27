# UCF Senior Design 2017-18
# Group 38

from PIL import Image
import numpy as np
import scipy as sp
import tensorflow as tf
import tflearn
import yaml


class FasterRCNN(object):
    def __init__(self, config, name='FasterRCNN'):
        self.config = self._read_config(config)
        self.classifier, self.bounding_boxes = self._build_network()

    def _build_network():
        """
            Builds the network architecture (VGG16) for the Faster R-CNN.
        """
        network = tflearn.layers.core.input_data(shape=[None, 448, 448, 3])
        network = tflearn.layers.conv.conv_2d(network, 64, 3, activation='relu')
        network = tflearn.layers.conv.conv_2d(network, 64, 3, activation='relu')
        network = tflearn.layers.conv.max_pool_2d(network, 2, strides=2)

        network = tflearn.layers.conv.conv_2d(network, 128, 3, activation='relu')
        network = tflearn.layers.conv.conv_2d(network, 128, 3, activation='relu')
        network = tflearn.layers.conv.max_pool_2d(network, 2, strides=2)

        network = tflearn.layers.conv.conv_2d(network, 256, 3, activation='relu')
        network = tflearn.layers.conv.conv_2d(network, 256, 3, activation='relu')
        network = tflearn.layers.conv.conv_2d(network, 256, 3, activation='relu')
        network = tflearn.layers.conv.max_pool_2d(network, 2, strides=2)

        network = tflearn.layers.conv.conv_2d(network, 512, 3, activation='relu')
        network = tflearn.layers.conv.conv_2d(network, 512, 3, activation='relu')
        network = tflearn.layers.conv.conv_2d(network, 512, 3, activation='relu')
        network = tflearn.layers.conv.max_pool_2d(network, 2, strides=2)
        network = tflearn.layers.conv.conv_2d(network, 512, 3, activation='relu')

        rpn = tflearn.layers.conv.conv_2d(network, 512, 3, padding='valid', activation='relu')
        classify = tflearn.layers.conv.conv_2d(rpn, 9*2, 1, padding='valid', activation='sigmoid')
        bboxes = tflearn.layers.conv.conv_2d(rpn, 9*4, 1, padding='valid', activation='linear')

        return classify, bboxes

    def _read_config(config):
        with open(config, 'r') as ymlfile:
            cfg = yaml.load(ymlfile)
            return cfg
