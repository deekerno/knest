# UCF Senior Design 2017-18
# Group 38

from .rpn_proposal import RPNProposal
from .rpn_target import RPNTarget
import tensorflow as tf
import tflearn
import models.faster_rcnn.utils as utils


class RPN(object):
    def __init__(self, incoming, config, all_anchors, image_shape, trained_output, name='RPN'):
        self.config = utils.read_config(config)
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

    def _create_dict(self, all_anchors, image_shape, trained_output, gt_boxes):

        # Dictionary to hold information about predictions for easy reference
        prediction_dict = {}

        # Apply activation to the output of the region proposal network
        rpn_conv_feature = self._rpn(trained_output)
        rpn_feature = tflearn.activations.relu(rpn_conv_feature)

        # Get the classification scores and bounding box predictions
        rpn_original_cls_score = self._rpn_class(rpn_feature)
        rpn_original_bbox_pred = self._rpn_bbox(rpn_feature)

        # Get the probabilities for the class labels
        rpn_cls_score = tf.reshape(rpn_original_cls_score, [-1, 2])
        rpn_cls_prob = tf.nn.softmax(rpn_cls_score)

        # Store the scores and probabilites
        prediction_dict['rpn_cls_prob'] = rpn_cls_prob
        prediction_dict['rpn_cls_score'] = rpn_cls_score

        rpn_bbox_pred = tf.reshape(rpn_original_bbox_pred, [-1, 4])

        # Store the prediction
        prediction_dict['rpn_bbox_pred'] = rpn_bbox_pred

        # Get region proposals
        proposal_prediction = self._proposal(
            rpn_cls_prob, rpn_bbox_pred, all_anchors, image_shape)

        prediction_dict['proposals'] = proposal_prediction['proposals']
        prediction_dict['scores'] = proposal_prediction['scores']

        if gt_boxes is not None:
            # When training we use a separate module to calculate the target
            # values we want to output.
            (rpn_cls_target, rpn_bbox_target,
             rpn_max_overlap) = self._anchor_target(
                all_anchors, gt_boxes, image_shape
            )

            prediction_dict['rpn_cls_target'] = rpn_cls_target
            prediction_dict['rpn_bbox_target'] = rpn_bbox_target

        # Variables summaries.
        variable_summaries(prediction_dict['scores'], 'rpn_scores', 'reduced')
        variable_summaries(rpn_cls_prob, 'rpn_cls_prob', 'reduced')
        variable_summaries(rpn_bbox_pred, 'rpn_bbox_pred', 'reduced')

        return prediction_dict

    def loss(self, prediction_dict):
        """
        Returns cost for Region Proposal Network based on:

        Args:
            rpn_cls_score: Score for being an object or not for each anchor
                in the image. Shape: (num_anchors, 2)
            rpn_cls_target: Ground truth labeling for each anchor. Should be
                * 1: for positive labels
                * 0: for negative labels
                * -1: for labels we should ignore.
                Shape: (num_anchors, )
            rpn_bbox_target: Bounding box output delta target for rpn.
                Shape: (num_anchors, 4)
            rpn_bbox_pred: Bounding box output delta prediction for rpn.
                Shape: (num_anchors, 4)
        Returns:
            Multiloss between cls probability and bbox target.
        """

        rpn_cls_score = prediction_dict['rpn_cls_score']
        rpn_cls_target = prediction_dict['rpn_cls_target']

        rpn_bbox_target = prediction_dict['rpn_bbox_target']
        rpn_bbox_pred = prediction_dict['rpn_bbox_pred']

        with tf.variable_scope('RPNLoss'):
            # Flatten already flat Tensor for usage as boolean mask filter.
            rpn_cls_target = tf.cast(tf.reshape(
                rpn_cls_target, [-1]), tf.int32, name='rpn_cls_target')
            # Transform to boolean tensor mask for not ignored.
            labels_not_ignored = tf.not_equal(
                rpn_cls_target, -1, name='labels_not_ignored')

            # Now we only have the labels we are going to compare with the
            # cls probability.
            labels = tf.boolean_mask(rpn_cls_target, labels_not_ignored)
            cls_score = tf.boolean_mask(rpn_cls_score, labels_not_ignored)

            # We need to transform `labels` to `cls_score` shape.
            # convert [1, 0] to [[0, 1], [1, 0]] for ce with logits.
            cls_target = tf.one_hot(labels, depth=2)

            # Equivalent to log loss
            ce_per_anchor = tf.nn.softmax_cross_entropy_with_logits(
                labels=cls_target, logits=cls_score
            )
            prediction_dict['cross_entropy_per_anchor'] = ce_per_anchor

            # Finally, we need to calculate the regression loss over
            # `rpn_bbox_target` and `rpn_bbox_pred`.
            # We use SmoothL1Loss.
            rpn_bbox_target = tf.reshape(rpn_bbox_target, [-1, 4])
            rpn_bbox_pred = tf.reshape(rpn_bbox_pred, [-1, 4])

            # We only care for positive labels (we ignore backgrounds since
            # we don't have any bounding box information for it).
            positive_labels = tf.equal(rpn_cls_target, 1)
            rpn_bbox_target = tf.boolean_mask(rpn_bbox_target, positive_labels)
            rpn_bbox_pred = tf.boolean_mask(rpn_bbox_pred, positive_labels)

            # We apply smooth l1 loss as described by the Fast R-CNN paper.
            reg_loss_per_anchor = utils.smooth_l1_loss(
                rpn_bbox_pred, rpn_bbox_target, sigma=self._l1_sigma
            )

            prediction_dict['reg_loss_per_anchor'] = reg_loss_per_anchor

            # Loss summaries.
            tf.summary.scalar('batch_size', tf.shape(labels)[0], ['rpn'])
            foreground_cls_loss = tf.boolean_mask(
                ce_per_anchor, tf.equal(labels, 1))
            background_cls_loss = tf.boolean_mask(
                ce_per_anchor, tf.equal(labels, 0))
            tf.summary.scalar(
                'foreground_cls_loss',
                tf.reduce_mean(foreground_cls_loss), ['rpn'])
            tf.summary.histogram(
                'foreground_cls_loss', foreground_cls_loss, ['rpn'])
            tf.summary.scalar(
                'background_cls_loss',
                tf.reduce_mean(background_cls_loss), ['rpn'])
            tf.summary.histogram(
                'background_cls_loss', background_cls_loss, ['rpn'])
            tf.summary.scalar(
                'foreground_samples', tf.shape(rpn_bbox_target)[0], ['rpn'])

            return {
                'rpn_cls_loss': tf.reduce_mean(ce_per_anchor),
                'rpn_reg_loss': tf.reduce_mean(reg_loss_per_anchor),
            }
