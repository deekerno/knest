# UCF Senior Design 2017-18
# Group 38

import models.faster_rcnn.utils as utils
import tensorflow as tf
import tflearn
import yaml

class RPNProposal(object):
    def __init__(self, rpn_cls_prob, rpn_bbox_pred, all_anchors, image_shape, num_anchors, config, name="RPNProposal"):
        self.config = utils.read_config(config)
        self.prediction_dict = self._create_dict(rpn_cls_prob, rpn_bbox_pred, all_anchors, image_shape)

    def _create_dict(self, rpn_cls_prob, rpn_bbox_pred, all_anchors, image_shape):
        # Scores are extracted from the second scalar of the cls probability.
        # cls_probability is a softmax of (background, foreground).
        all_scores = rpn_cls_prob[:, 1]
        # Force flatten the scores (it should be already be flatten).
        all_scores = tf.reshape(all_scores, [-1])

        if self._filter_outside_anchors:
            with tf.name_scope('filter_outside_anchors'):
                (x_min_anchor, y_min_anchor,
                 x_max_anchor, y_max_anchor) = tf.unstack(all_anchors, axis=1)

                anchor_filter = tf.logical_and(
                    tf.logical_and(
                        tf.greater_equal(x_min_anchor, 0),
                        tf.greater_equal(y_min_anchor, 0)
                    ),
                    tf.logical_and(
                        tf.less(x_max_anchor, image_shape[1]),
                        tf.less(y_max_anchor, image_shape[0])
                    )
                )
                anchor_filter = tf.reshape(anchor_filter, [-1])
                all_anchors = tf.boolean_mask(
                    all_anchors, anchor_filter, name='filter_anchors')
                rpn_bbox_pred = tf.boolean_mask(rpn_bbox_pred, anchor_filter)
                all_scores = tf.boolean_mask(all_scores, anchor_filter)

        # Decode boxes
        all_proposals = decode(all_anchors, rpn_bbox_pred)

        # Filter proposals with less than threshold probability.
        min_prob_filter = tf.greater_equal(
            all_scores, self._min_prob_threshold
        )

        # Filter proposals with negative or zero area.
        (x_min, y_min, x_max, y_max) = tf.unstack(all_proposals, axis=1)
        zero_area_filter = tf.greater(
            tf.maximum(x_max - x_min, 0.0) * tf.maximum(y_max - y_min, 0.0),
            0.0
        )
        proposal_filter = tf.logical_and(zero_area_filter, min_prob_filter)

        # Filter proposals and scores.
        all_proposals_total = tf.shape(all_scores)[0]
        unsorted_scores = tf.boolean_mask(
            all_scores, proposal_filter,
            name='filtered_scores'
        )
        unsorted_proposals = tf.boolean_mask(
            all_proposals, proposal_filter,
            name='filtered_proposals'
        )
        if self._debug:
            proposals_unclipped = tf.identity(unsorted_proposals)

        if not self._clip_after_nms:
            # Clip proposals to the image.
            unsorted_proposals = utils.clip_boxes(unsorted_proposals, image_shape)

        filtered_proposals_total = tf.shape(unsorted_scores)[0]

        tf.summary.scalar(
            'valid_proposals_ratio',
            (
                tf.cast(filtered_proposals_total, tf.float32) /
                tf.cast(all_proposals_total, tf.float32)
            ), ['rpn'])

        tf.summary.scalar(
            'invalid_proposals',
            all_proposals_total - filtered_proposals_total, ['rpn'])

        # Get top `pre_nms_top_n` indices by sorting the proposals by score.
        k = tf.minimum(self._pre_nms_top_n, tf.shape(unsorted_scores)[0])
        top_k = tf.nn.top_k(unsorted_scores, k=k)

        sorted_top_proposals = tf.gather(unsorted_proposals, top_k.indices)
        sorted_top_scores = top_k.values

        if self._apply_nms:
            with tf.name_scope('nms'):
                # We reorder the proposals into TensorFlows bounding box order
                # for `tf.image.non_max_supression` compatibility.
                proposals_tf_order = utils.change_order(sorted_top_proposals)
                # We cut the pre_nms filter in pure TF version and go straight
                # into NMS.
                selected_indices = tf.image.non_max_suppression(
                    proposals_tf_order, tf.reshape(
                        sorted_top_scores, [-1]
                    ),
                    self._post_nms_top_n, iou_threshold=self._nms_threshold
                )

                # Selected_indices is a smaller tensor, we need to extract the
                # proposals and scores using it.
                nms_proposals_tf_order = tf.gather(
                    proposals_tf_order, selected_indices,
                    name='gather_nms_proposals'
                )

                # We switch back again to the regular bbox encoding.
                proposals = utils.change_order(nms_proposals_tf_order)
                scores = tf.gather(
                    sorted_top_scores, selected_indices,
                    name='gather_nms_proposals_scores'
                )
        else:
            proposals = sorted_top_proposals
            scores = sorted_top_scores

        if self._clip_after_nms:
            # Clip proposals to the image after NMS.
            proposals = utils.clip_boxes(proposals, image_shape)

        pred = {
            'proposals': proposals,
            'scores': scores,
        }

        if self._debug:
            pred.update({
                'sorted_top_scores': sorted_top_scores,
                'sorted_top_proposals': sorted_top_proposals,
                'unsorted_proposals': unsorted_proposals,
                'unsorted_scores': unsorted_scores,
                'all_proposals': all_proposals,
                'all_scores': all_scores,
                # proposals_unclipped has the unsorted_scores scores
                'proposals_unclipped': proposals_unclipped,
            })

        return pred
