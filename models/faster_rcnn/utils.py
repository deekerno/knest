# UCF Senior Design 2017-18
# Group 38

import numpy as np
import tensorflow as tf
import yaml

VAR_LOG_LEVELS = {
    'full': ['variable_summaries_full'],
    'reduced': ['variable_summaries_reduced', 'variable_summaries_full'],
}


def read_config(config):
    with open(config, 'r') as ymlfile:
        cfg = yaml.load(ymlfile)
        return cfg


def generate_anchors_reference(base_size, aspect_ratios, scales):
    """
        Generate an anchor that will be use as a reference for all others.
            base_size:       (Integer) base size of the reference anchor
            aspect_ratios:   (Tensor) ratios to use to generate different anchors
            scales:          (Tensor) scaling ratios to be applied to area.
    """

    # Create a rectangular grid from the source arrays.
    scales_grid, aspect_ratios_grid = np.meshgrid(scales, aspect_ratios)

    # Flatten the scale and aspect ratio grids to 1D arrays.
    base_scales = scales_grid.reshape(-1)
    base_aspect_ratios = aspect_ratios_grid.reshape(-1)

    # Create a list of different heights and widths.
    aspect_ratio_sqrts = np.sqrt(base_aspect_ratios)
    heights = base_scales * aspect_ratio_sqrts * base_size
    widths = base_scales / aspect_ratio_sqrts * base_size

    # Center point has the same X, Y value.
    center_xy = 0

    # Create anchor reference.
    anchors = np.column_stack([
        center_xy - (widths - 1) / 2,
        center_xy - (heights - 1) / 2,
        center_xy + (widths - 1) / 2,
        center_xy + (heights - 1) / 2,
    ])

    # Create a list of positive heights and widths in order to cover
    # the image in anchors in a mix of different sizes. These operations
    # subtract the second column from the fourth and the first from the third.
    real_heights = (anchors[:, 3] - anchors[:, 1]).astype(np.int)
    real_widths = (anchors[:, 2] - anchors[:, 0]).astype(np.int)

    # Cannot have an anchor of zero height/width.
    if (real_widths == 0).any() or (real_heights == 0).any():
        raise ValueError(
            'base_size {} is too small for aspect_ratios and scales.'.format(
                base_size
            )
        )

    return anchors


def get_width_upright(bboxes):
    """
        Get the width, height, and top right corner coordinates of
        the bounding boxes.
            bboxes:         (Tensor) bounding boxes
    """

    with tf.name_scope('BoundingBoxTransform/get_width_upright'):

        # Convert tensor to a new type.
        bboxes = tf.cast(bboxes, tf.float32)

        # Split tensor into sub-tensors.
        x1, y1, x2, y2 = tf.split(bboxes, 4, axis=1)

        # Calculate box width and height.
        width = x2 - x1 + 1.
        height = y2 - y1 + 1.

        # Calculate up right point of bbox (urx = up right x)
        urx = x1 + .5 * width
        ury = y1 + .5 * height

        return width, height, urx, ury


def encode(bboxes, gt_boxes):
    """
        Get the width, height, and top right corner coordinates of
        the bounding boxes.
            bboxes:         (Tensor) bounding boxes
            gt_boes:        (Tensor) ground-truth bounding boxes
    """

    with tf.name_scope('BoundingBoxTransform/encode'):

        # Get the width, height, and top right corners of bounding boxes.
        (bboxes_width, bboxes_height,
         bboxes_urx, bboxes_ury) = get_width_upright(bboxes)

        # Get the width, height, and top right corners of ground truth boxes.
        (gt_boxes_width, gt_boxes_height,
         gt_boxes_urx, gt_boxes_ury) = get_width_upright(gt_boxes)

        # Calculate the deltas-x,y between ground truth and proposed boxes.
        targets_dx = (gt_boxes_urx - bboxes_urx) / bboxes_width
        targets_dy = (gt_boxes_ury - bboxes_ury) / bboxes_height

        # Calculate the deltas-w,h between ground truth and proposed boxes.
        targets_dw = tf.log(gt_boxes_width / bboxes_width)
        targets_dh = tf.log(gt_boxes_height / bboxes_height)

        # Collapse the target sub-tensors into a final target tensor in which
        # the first element of each sub-tensor is collapsed into the first
        # element of the final target tensor and so on.
        targets = tf.concat(
            [targets_dx, targets_dy, targets_dw, targets_dh], axis=1)

        return targets


def clip_boxes(bboxes, imshape):
    """
        Clips bounding boxes to image boundaries based on image shape.
            bboxes:         (Tensor) bounding box with point order x1, y1, x2, y2
            imshape:        (Tensor) with shape (2, ), the first value is
                                     height and the next is width.
    """

    with tf.name_scope('BoundingBoxTransform/clip_bboxes'):

        # Convert tensor to new type.
        bboxes = tf.cast(bboxes, dtype=tf.float32)
        imshape = tf.cast(imshape, dtype=tf.float32)

        # Split the tensor into sub-tensors.
        x1, y1, x2, y2 = tf.split(bboxes, 4, axis=1)

        width = imshape[1]
        height = imshape[0]

        # Make sure the coordinates of the box are greater than
        # zero, but less than the size of the entire image
        x1 = tf.maximum(tf.minimum(x1, width - 1.0), 0.0)
        x2 = tf.maximum(tf.minimum(x2, width - 1.0), 0.0)

        y1 = tf.maximum(tf.minimum(y1, height - 1.0), 0.0)
        y2 = tf.maximum(tf.minimum(y2, height - 1.0), 0.0)

        # Turn the tensor into a flat 1D list of each box's coordinates.
        bboxes = tf.concat([x1, y1, x2, y2], axis=1)

        return bboxes


def change_order(bboxes):
    """
    Change bounding box encoding order. TensorFlow works with the (y_min, x_min,
    y_max, x_max) order while we work with the (x_min, y_min, x_max, y_min).
        bboxes: A Tensor of shape (total_bboxes, 4)
    """
    with tf.name_scope('BoundingBoxTransform/change_order'):
        first_min, second_min, first_max, second_max = tf.unstack(
            bboxes, axis=1)

        bboxes = tf.stack(
            [second_min, first_min, second_max, first_max], axis=1)

        return bboxes


def smooth_l1_loss(bbox_prediction, bbox_target, sigma=3.0):
    """
        Return Smooth L1 Loss for bounding box prediction.
            bbox_prediction: shape (1, H, W, num_anchors * 4)
            bbox_target:     shape (1, H, W, num_anchors * 4)

        Smooth L1 loss is defined as:
            0.5 * x^2                  if |x| < d
            abs(x) - 0.5               if |x| >= d
        where d = 1 and x = prediction - target
    """
    sigma_sq = sigma ** 2

    diff = bbox_prediction - bbox_target

    abs_diff = tf.abs(diff)
    abs_diff_lt_sigma_sq = tf.less(abs_diff, 1.0 / sigma_sq)

    bbox_loss = tf.reduce_sum(
        tf.where(
            abs_diff_lt_sigma_sq,
            0.5 * sigma_sq * tf.square(abs_diff),
            abs_diff - 0.5 / sigma_sq
        ), [1]
    )

    return bbox_loss


def create_all_anchors(anchor_ref, stride, image_size):
    """
        Creates a tensor that cotains all the anchors needed for the network.
            anchor_ref:         (Tensor) the anchor reference from which others are based
            stride:             (Integer) the stride of the network
            image_size:         (Integer) size of the input image
    """

    # Calculate parameters of the grid.
    grid_width = int(image_size / stride)
    grid_height = int(image_size / stride)

    # Make arrays of pixel coordinates that will function as anchor centers.
    shift_x = np.arange(grid_width) * stride
    shift_y = np.arange(grid_height) * stride

    # Create a rectangular grid from the two arrays.
    shift_x, shift_y = np.meshgrid(shift_x, shift_y)

    # Reshape the arrays into their own grids for easier iteration.
    shift_x = shift_x.reshape([-1])
    shift_y = shift_y.reshape([-1])

    # Create the final shifts array for use in representing objects in image.
    shifts = np.stack([shift_x, shift_y, shift_x, shift_y], axis=0)
    shifts = shifts.T

    num_anchors = anchor_ref.shape[0]
    num_anchor_points = shifts.shape[0]

    # Assign anchors to respective coordinates all over the image.
    all_anchors = (
        anchor_ref.reshape((1, num_anchors, 4)) +
        np.transpose(
            shifts.reshape((1, num_anchor_points, 4)),
            axes=(1, 0, 2)
        )
    )

    all_anchors = np.reshape(all_anchors, (num_anchors * num_anchor_points, 4))

    return all_anchors


def variable_summaries(var, name, collection_key):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization).
    Args:
        - var: Tensor for variable from which we want to log.
        - name: Variable name.
        - collection_key: Collection to save the summary to, can be any key of
          `VAR_LOG_LEVELS`.
    """
    if collection_key not in VAR_LOG_LEVELS.keys():
        raise ValueError('"{}" not in `VAR_LOG_LEVELS`'.format(collection_key))
    collections = VAR_LOG_LEVELS[collection_key]

    with tf.name_scope(name):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean, collections)
        num_params = tf.reduce_prod(tf.shape(var))
        tf.summary.scalar('num_params', num_params, collections)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev, collections)
        tf.summary.scalar('max', tf.reduce_max(var), collections)
        tf.summary.scalar('min', tf.reduce_min(var), collections)
        tf.summary.histogram('histogram', var, collections)
        tf.summary.scalar('sparsity', tf.nn.zero_fraction(var), collections)
