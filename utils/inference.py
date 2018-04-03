# UCF Senior Design 2017-18
# Group 38

import numpy as np
import tensorflow as tf
import utils.global_var as gv
import utils.label_map_utils as label_map_utils
import utils.visualization as vis

# Path to frozen detection graph.
PATH_TO_CKPT = 'architectures/detection/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = 'architectures/detection/bird_object_detection.pbtxt'

# Classes: bird, bird_face
NUM_CLASSES = 2


def inference(filename, image):
    """
    Infers the location of birds and their faces in a given image and returns
    a copy of that image with bounding boxes drawn ontop and its count of faces
        filename: (String) name of the image file
        image: (ndarray) colored array representation of image
    """
    # copy image parameter
    img = np.copy(image)

    # run inference
    output_dict = gv.sess.run(gv.tensor_dict, feed_dict={
        gv.image_tensor: np.expand_dims(img, 0)})

    # convert all types as float32 numpy arrays
    output_dict['detection_classes'] = output_dict[
        'detection_classes'][0].astype(np.uint8)
    output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
    output_dict['detection_scores'] = output_dict['detection_scores'][0]

    # draw bounding boxes ontop of image copy
    vis.visualize_boxes_and_labels_on_image_array(img, output_dict[
        'detection_boxes'], output_dict['detection_classes'], output_dict[
        'detection_scores'], gv.category_index,
        use_normalized_coordinates=True, line_thickness=16)

    # get detection boxes as [N, 4]
    boxes = np.squeeze(output_dict['detection_boxes'])
    # get the width and height of image
    width = np.shape(img)[1]
    height = np.shape(img)[0]

    # list of tuples that hold bounding box coordinates
    # for birds and faces
    birds = []
    faces = []

    # determine if image contains birds
    for i, score in enumerate(output_dict['detection_scores']):
        # if score is greater than 50%, a label is given
        if score > .5:
            # determine class label of box, whether bird or face
            label = output_dict['detection_classes'][i]
            # divide box into four coordinates
            xmin = round(boxes[i, 1] * width, 0)
            ymin = round(boxes[i, 0] * height, 0)
            xmax = round(boxes[i, 3] * width, 0)
            ymax = round(boxes[i, 2] * height, 0)

            # add box coordinates to appropriate label list
            if label == 1:
                birds.append((xmin, ymin, xmax, ymax))
            if label == 2:
                faces.append((xmin, ymin, xmax, ymax))

    # added all bird and face coordinates to respective dictionaries
    # all linked to the image filename
    gv.boxes[filename] = {'birds': birds, 'faces': faces}

    # return the image with bounding boxes displayed
    return img


def instantiate():
    """
    Initializes detection-dependent variables that will only
    be calculated once throughout main driver
    """
    gv.graph = tf.Graph()
    with gv.graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

    gv.label_map = label_map_utils.load_labelmap(PATH_TO_LABELS)
    gv.categories = label_map_utils.convert_label_map_to_categories(
        gv.label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
    gv.category_index = label_map_utils.create_category_index(gv.categories)

    with gv.graph.as_default():
        with tf.Session() as gv.sess:
            # get handles to input and output tensors
            gv.ops = tf.get_default_graph().get_operations()
            gv.tensor_names = {
                output.name for op in gv.ops for output in op.outputs}
            gv.tensor_dict = {}

        for key in [
            'num_detections', 'detection_boxes', 'detection_scores',
            'detection_classes', 'detection_masks'
        ]:
            name = key + ':0'
            if name in gv.tensor_names:
                gv.tensor_dict[key] = tf.get_default_graph(
                ).get_tensor_by_name(name)

        gv.image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')
