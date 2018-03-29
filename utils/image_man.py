# UCF Senior Design 2017-18
# Group 38

from PIL import Image
import math
import numpy as np

SCALING_FACTOR = 3

def man(filename, image_array, bird_box, face_box, scaling_factor=SCALING_FACTOR):
    """
        Crop and manipulate the iamge for final output.
            image:          (Array) array representation of the image
            box:            (Tuple) bounding box around subject;
                            form: (ymin, xmin, ymax, xmax)
            scaling_factor  (Integer) the amount by which to scale the 
                            bounding box of the object; this is done as a way
                            to include some of the environment in the final image
    """
    image = Image.fromarray(image_array)
    width, height = image.size()

    # Calculate the factor by which we multiply the width and height of box.
    factor = math.sqrt(scaling_factor)

    # Get the corners of the bounding boxes returned from the detection graph.
    bb_xmin, bb_ymin, bb_xmax, bb_ymax = bird_box
    fb_xmin, fb_ymin, fb_xmax, fb_ymax = face_box

    # Calculate the center of the face bounding box.
    fb_center_x, fb_center_y = fb_xmax - fbxmin, fb_ymax - fb_ymin

    # Calculate the width and height of the final crop area.
    bb_width, bb_height = bb_xmax - bb_xmin, bb_ymax - bb_ymin
    new_width, new_height = round(bb_width * factor, 0),
                            round(bb_height * factor, 0)

    # Calculate the amounts by which to adjust the face_box coordinates.
    width_diff, height_diff = new_width / 2, new_height / 2

    # Set the new dimenstions for the final crop box.
    final_xmin, final_xmax = fb_center_x - width_diff, fb_center_x + width_diff
    final_ymin, final_ymax = fb_center_y - height_diff, fb_center_y + height_diff

    # Edge case handling.
    if final_xmin < 0: final_xmin = 0
    if final_xmax > width: final_xmax = width
    if final_ymin < 0: final_ymin = 0
    if final_ymax > width: final_ymax = height

    # Crop and attempt to save image.
    cropped_area = image.crop(final_xmin, final_ymin, final_xmax, final_ymax)
    
    try:
        final_image = np.asarray(cropped_area)
        return final_image, True
    except IOError:
        print("File could not be written properly.")
        return False
