# UCF Senior Design 2017-18
# Group 38

"""
Module to hold all global variables used in `main.py`
"""
# completion indicators of process checkpoints
blur_step = 0
bird_step = 0
birdbb_step = 0

# classification model variables
load = 0
model = None

# detection model variables
graph = None
label_map = None
categories = None
category_index = None

sess = None
ops = None
tensor_names = None
tensor_dict = None
image_tensor = None

# path information variables
dir_paths = []
path_index = 0
des_path = ''
startpath = ''
num_files = 0

# reusuable variables for use in real-time display
first_pass = 0
index = 0
files = []

# holds image data of entire process
images = {}

# holds box coordinates for each image
boxes = {}

# hold folder_select screen class instance
fs = None

# image comparison variable
comp = True
std = ''
std_hash = None
count = 0

# image manipulation variables
crop = True
landscape = True

canvas = None


def reset():
    """
    Reset all global variables for use in future passes
    """
    global blur_step, bird_step, birdbb_step
    global dir_path, des_path, num_files
    global first_pass, index, files
    global comp, std, std_hash, count
    global images, boxes, canvas

    blur_step = bird_step = birdbb_step = 0

    des_path = ''
    num_files = 0

    first_pass = index = 0
    files = []

    std = ''
    std_hash = None
    count = 0

    # empty dictionaries
    images.clear()
    boxes.clear()

    # clear numpy texture
    if canvas is not None:
        canvas.clear()
    canvas = None
