# UCF Senior Design 2017-18
# Group 38

from .rpn_proposal import RPNProposal
from .rpn_target import RPNTarget
import tensorflow as tf
import tflearn
import yaml


class RPN(object):
    def __init__(self, incoming, config, all_anchors, image_shape, trained_output, name='RPN'):
        self.config = self._read_config(config)
        self.rpn_config = self.config['model']['rpn']
        self._rpn, self._rpn_class, self._rpn_bbox = self._build_network()
        self._proposal = RPNProposal(config)
        self._anchor_target = RPNTarget(config)
        self.prediction_dict = self._create_dict(all_anchors, trained_output)

    def _build_network(self, incoming):

        # Store all the important configuration options
        self._num_channels = self.rpn_config['num_channels']
        self._filter_size = self.rpn_config['kernel_shape']
        self._rpn_init = self.rpn_config['rpn_initializer']['type']
        self._cls_init = self.rpn_config['cls_initializer']['type']
        self._bbox_init = self.rpn_config['bbox_initializer']['type']

        # Instantiate three separate layers that will handle regions (anchors)...
        rpn = tflearn.layers.conv.conv_2d(incoming, self._num_channels,
                                          self._filter_size,
                                          weights_init=self._rpn_init,
                                          padding='valid',
                                          name='rpn_conv')

        # classes (in this use case, just two classes [bird/not bird])...
        rpn_class = tflearn.layers.conv.conv_2d(incoming, self._num_channels,
                                                self._filter_size,
                                                weights_init=self._cls_init,
                                                padding='valid',
                                                name='rpn_class_conv')

        # and bounding box prediction deltas from ground truth.
        rpn_bbox = tflearn.layers.conv.conv_2d(incoming, self._num_channels,
                                               self._filter_size,
                                               weights_init=self._bbox_init,
                                               padding='valid',
                                               name='rpn_bbox_conv')
        return rpn, rpn_class, rpn_bbox

    def _create_dict(self, all_anchors, image_shape, trained_output):

        # Dictionary to hold information about predictions for easy reference
        prediction_dict = {}

        # Apply activation to the output of the region proposal network
        rpn_conv_feature = self._rpn(trained_output)
        rpn_feature = tflearn.activations.relu(rpn_conv_feature)

        # Get the classification scores and bounding box predictions
        rpn_original_cls_score = self._rpn_class(rpn_feature)
        rpn_original_bbox_pred = self._rpn_bbox(rpn_feature)

        rpn_cls_score = tf.reshape(rpn_original_cls_score, [-1, 2])
        rpn_cls_prob = tf.nn.softmax(rpn_cls_score)

        prediction_dict['rpn_cls_prob'] = rpn_cls_prob
        prediction_dict['rpn_cls_score'] = rpn_cls_score

        rpn_bbox_pred = tf.reshape(rpn_original_bbox_pred, [-1, 4])

        prediction_dict['rpn_bbox_pred'] = rpn_bbox_pred

        proposal_prediction = self._proposal(
            rpn_cls_prob, rpn_bbox_pred, all_anchors, image_shape)

        prediction_dict['proposals'] = proposal_prediction['proposals']
        prediction_dict['scores'] = proposal_prediction['scores']

        return prediction_dict

    def _read_config(config):
        with open(config, 'r') as ymlfile:
            cfg = yaml.load(ymlfile)
            return cfg
