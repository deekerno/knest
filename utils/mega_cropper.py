# UCF Senior Design 2017-18
# Group 38

import os
import xml.etree.ElementTree as ET
from PIL import Image

#BOX_FOLDER = '/users/ad/projects/knest/birdcages'
#DATASET_FOLDER = '/users/ad/projects/knest/birdstheword'
CROPPED_FOLDER = '/users/ad/projects/knest/crops'
NONBIRD_CROP_FOLDER = '/users/ad/projects/knest/nonbird_crops'
NONBIRDS = '/users/ad/projects/knest/nonbirds'
NONBIRD_CROP_AREA = (2392, 1528, 2792, 1928)


def list_gen_xml(folder):
    gen_list = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith('.xml')]
    return sorted(gen_list)


def list_gen_image(folder):
    gen_list = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith('.JPG')]
    return sorted(gen_list)


nonbird_images = list_gen_image(NONBIRDS)
#boxes = list_gen_xml(BOX_FOLDER)
crop_areas = []


"""def bounding_boxes(xml_file):
    # lists to hold xmin, xmax, ymin, ymax coordinates
    xmin = []
    xmax = []
    ymin = []
    ymax = []
    # parse xml for bounding box coordinates
    tree = ET.parse(xml_file)
    root = tree.getroot()
    # for every bounding box in the image, add the max and min 
    # coordinates for the x,y points into four separate lists
    member = root.find('object')
    print(xml_file)
    print(member)
    xmin = (int(member[4][0].text))
    xmax = (int(member[4][2].text))
    ymin = (int(member[4][1].text))
    ymax = (int(member[4][3].text))
    return (xmin, ymin, xmax, ymax)"""


"""for i in range(0, len(boxes)):
    crop_areas.append(bounding_boxes(boxes[i]))

print(crop_areas)

for i in range(0, len(crop_areas)):
    basename = os.path.basename(boxes[i])
    filename = os.path.splitext(basename)[0] + '.JPG'
    image = Image.open(os.path.join(DATASET_FOLDER, filename))
    copy = image.copy()
    cropped_area = copy.crop(crop_areas[i])
    cropped_area.save(os.path.join(CROPPED_FOLDER, filename))"""

for img in nonbird_images:
    image = Image.open(img)
    copy = image.copy()
    cropped_area = copy.crop(NONBIRD_CROP_AREA)
    cropped_area.save(os.path.join(NONBIRD_CROP_FOLDER, img))
