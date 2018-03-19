# UCF Senior Design 2017-18
# Group 38

"""
Module to hold all global variables used in `main.py`
"""
# completion indicators of process checkpoints
blur_step = 0
bird_step = 0

# classification model variables
load = 0
model = None

# path information variables
dir_path = ''
des_path = ''
num_files = 0

# reusuable variables for use in real-time display
first_pass = 0
index = 0
files = []

# holds image data of entire process
images = {}

# image comparison variable
comp = 0
std = ''
std_hash = None
count = 0


def reset():
    """
    Reset all global variables for use in future passes
    """
    global blur_step, bird_step
    global dir_path, des_path, num_files
    global first_pass, index, files
    global images
    global comp, std, std_hash, count

    blur_step = bird_step = 0

    dir_path = des_path = ''
    num_files = 0

    first_pass = index = 0
    files = []

    std = ''
    std_hash = None
    count = 0

    # empty images dictionary
    images.clear()
