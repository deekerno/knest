# UCF Senior Design 2017-18
# Group 38

from PIL import Image
import math
import numpy as np
import piexif

SCALING_FACTOR = 3


def man(boxes, image_array, landscape=True, scaling_factor=SCALING_FACTOR):
    """
        Crop and manipulate the image for final output.
            image_array:    (Array) array representation of the image
            boxes:          (Dict) bounding box around subject;
                            form: (ymin, xmin, ymax, xmax)
            scaling_factor  (Integer) the amount by which to scale the
                            bounding box of the object; this is done as a way
                            to include some of the environment in the final image
    """
    image = Image.fromarray(image_array)
    width, height = image.size
    img_center_x = width / 2

    dist_face_center_x = math.inf
    central_face_x, central_face_y = 0, 0

    for i in boxes['faces']:

        # Get the bounding box coordinates of the face.
        fb_xmin, fb_ymin, fb_xmax, fb_ymax = i

        # Calculate the center of the face bounding box.
        fb_center_x, fb_center_y = (
            fb_xmax + fb_xmin) / 2, (fb_ymax + fb_ymin) / 2

        # Calculate the distance of the x-component of the
        # the face box centerfrom the center of the image.
        delta_center_x = math.fabs(fb_center_x - img_center_x)

        # Get the closest face box center coordinates over all face boxes.
        if delta_center_x < dist_face_center_x:
            dist_face_center_x = delta_center_x
            central_face_x, central_face_y = fb_center_x, fb_center_y

    # Initialize bounding box extrema.
    sm_xmin, sm_ymin, lar_xmax, lar_ymax = math.inf, math.inf, 0, 0

    # Calculate the factor by which we multiply the width and height of box.
    factor = math.sqrt(scaling_factor)

    # Get the extrema of the bounding boxes returned from the detection graph.
    for i in boxes['birds']:
        sm_xmin = min(sm_xmin, i[0])
        sm_ymin = min(sm_ymin, i[1])
        lar_xmax = max(lar_xmax, i[2])
        lar_ymax = max(lar_ymax, i[3])

    # Calculate the width and height of the final crop area.
    bb_width, bb_height = lar_xmax - sm_xmin, lar_ymax - sm_ymin
    new_width, new_height = round(
        bb_width * factor, 0), round(bb_height * factor, 0)

    if landscape:
        if new_width * 1.5 < new_height or math.fabs(new_width - new_height) < new_height * .5:
            new_width = new_height * 1.5

    if landscape:
        if new_width * 1.5 < new_height or math.fabs(new_width - new_height) < new_height * .5:
            new_width = new_height * 1.5

    # Calculate the amounts by which to adjust the face_box coordinates.
    width_diff, height_diff = new_width / 2, new_height / 2

    # Set the new dimensions for the final crop box.
    final_xmin, final_xmax = central_face_x - \
        width_diff, central_face_x + width_diff
    final_ymin, final_ymax = central_face_y - \
        height_diff, central_face_y + height_diff

    # Edge case handling.
    if final_xmin < 0: final_xmin = 0
    if final_xmax > width: final_xmax = width
    if final_ymin < 0: final_ymin = 0
    if final_ymax > height: final_ymax = height

    # Crop and attempt to save image.
    cropped_area = image.crop((final_xmin, final_ymin, final_xmax, final_ymax))

    try:
        final_image = np.asarray(cropped_area)
        return final_image, True
    except IOError:
        print("File could not be written properly.")
        return False


def exif(filename, image_array):
    """
        Transfer the EXIF metadata from the original photograph to the
        cropped version, along with some changes to a few values.
            filename:    (String) filename from original photo
            new_array:   (Array) array_representation of cropped image
    """
    try:
        # Get the EXIF metadata from the original image.
        exif_dict = piexif.load(filename)
    except piexif._exceptions.InvalidImageDataError:
        return None

    # Some photos mysteriously do not contain EXIF data, so
    # check if it actually exists and then return properly.
    if len(exif_dict["Exif"].items()) == 0:
        return None

    else:
        # Set the image height and width of new EXIF to the new
        # crop dimensions. Values come from the Piexif documentation.
        exif_dict["Exif"][40963], exif_dict["Exif"][40962], _ = image_array.shape

        # Convert EXIF dictionary to a bytes object for writing with PIL.
        exif_bytes = piexif.dump(exif_dict)

    return exif_bytes
