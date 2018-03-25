from PIL import Image
from tf_object_detection.models.research.object_detection.utils import label_map_util
from tf_object_detection.models.research.object_detection.utils import visualization_utils as vis_util
import cv2
import numpy as np
import os
import tensorflow as tf

MODEL_NAME = 'tf_object_detection/models/research/object_detection/bird_inference_graph'
# path to frozen detection graph. This is the actual model that is used
# for the object detection.
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'
# list of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join(
    'tf_object_detection/models/research/object_detection/training', 'bird_object_detection.pbtxt')
# classes: bird and face
NUM_CLASSES = 2

detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(
    label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)


def inference(image):
    # copy image parameter
    img = np.copy(image)

    with detection_graph.as_default():
        with tf.Session() as sess:
            # get handles to input and output tensors
            ops = tf.get_default_graph().get_operations()
            tensor_names = {
                output.name for op in ops for output in op.outputs}
            tensor_dict = {}

        for key in [
            'num_detections', 'detection_boxes', 'detection_scores',
            'detection_classes', 'detection_masks'
        ]:
            name = key + ':0'
            if name in tensor_names:
                tensor_dict[key] = tf.get_default_graph(
                ).get_tensor_by_name(name)

        image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

        # run inference
        output_dict = sess.run(tensor_dict, feed_dict={
                               image_tensor: np.expand_dims(img, 0)})

        # convert all types as float32 numpy arrays
        output_dict['num_detections'] = int(output_dict['num_detections'][0])
        output_dict['detection_classes'] = output_dict[
            'detection_classes'][0].astype(np.uint8)
        output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
        output_dict['detection_scores'] = output_dict['detection_scores'][0]

        vis_util.visualize_boxes_and_labels_on_image_array(img, output_dict[
            'detection_boxes'], output_dict['detection_classes'], output_dict[
            'detection_scores'], category_index, instance_masks=output_dict.get(
            'detection_masks'), use_normalized_coordinates=True, line_thickness=8)

        # get detection boxes as [N, 4]
        boxes = np.squeeze(output_dict['detection_boxes'])
        # get the number of boxes
        rows = boxes.shape[0]
        # get the width and height of image
        image_width = np.shape(img)[0]
        image_height = np.shape(img)[1]

        # iterate through bounding boxes
        for i in range(0, rows):
            # if the coordinates are all zero, stop processing
            if boxes[i, 0] == 0 and boxes[i, 1] == 0 and boxes[i, 2] == 0 and boxes[i, 3] == 0:
                break

            # retrieve normalized coordinates and convert them to coordinates
            # that make sense
            ymin = boxes[i, 0] * image_height
            xmin = boxes[i, 1] * image_width
            ymax = boxes[i, 2] * image_height
            xmax = boxes[i, 3] * image_width
            boxes[i, 0] = ymin.astype('int64')
            boxes[i, 1] = xmin.astype('int64')
            boxes[i, 2] = ymax.astype('int64')
            boxes[i, 3] = xmax.astype('int64')

        # return the image with bounding boxes displayed
        # and the coordinates for all the boxes
        return img, boxes


if __name__ == '__main__':

    for filename in os.listdir('/users/ayylmao/desktop/goo_test/'):
        if not filename.startswith('.'):
            img = cv2.imread(os.path.join('/users/ayylmao/desktop/goo_test/', filename))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            image, boxes = inference(img)

            img = Image.fromarray(image)
            img.show()
