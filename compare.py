import cv2
from PIL import Image
import os

# similarity threshold, set at 55%
SIM_INDEX = .55

"""
Creates a histogram for every image in the given directory
    dir_path: (String) path to the directory of images
"""
def make_hist(dir_path):
    # dictionaries to hold relevant info
    images = {}
    histograms = {}

    # loop through all images in directory
    for filename in os.listdir(dir_path):
        # ignore hidden files
        if not filename.startswith('.'):
            # open the file in the directory
            file = os.path.join(dir_path, filename)
            # convert to cv2 array, then convert colors
            img = cv2.imread(file)
            images[filename] = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # create a histogram for each image
            hist = cv2.calcHist([img], [0, 1], None, [180, 256], ranges=None)
            # normalize and flatten histogram
            hist = cv2.normalize(hist, hist).flatten()
            histograms[filename] = hist

    return images, histograms



"""
Compares two images' histograms, using opencv's intersection method.
Returns a boolean.
    hist1: first histogram, array of sorts
    hist2: second histogram, array of sorts
"""
def compare(hist1, hist2):

    # compare each histogram to itself, as to set a threshold for comparison
    h1 = cv2.compareHist(hist1, hist1, cv2.HISTCMP_INTERSECT)
    h2 = cv2.compareHist(hist2, hist2, cv2.HISTCMP_INTERSECT)

    # compare each histogram to each other; if values are within 55% of h1 and h2, respectively
    # return 'True' for similar, otherwise 'False'
    if cv2.compareHist(hist1, hist2, cv2.HISTCMP_INTERSECT) >= h1 * SIM_INDEX:
        if cv2.compareHist(hist2, hist1, cv2.HISTCMP_INTERSECT) >= h2 * SIM_INDEX:
            return True
        else:
            return False

    return False  

"""
Test function: compares a directory of images to a single image. All similar images will be added to a separate folder.
    img: (String) the name of the image being compared to. MUST be included in the folder with rest of images
    images: a dictionary that holds array forms of a number of images
    histograms: said images' histograms
    des_path: path to a folder that will hold all similar images to 'img'
"""
def temp_test(img, images, histograms, des_path):

    for (filename, hist) in histograms.items():
        # if the image is similar to 'img', add to destination folder
        if compare(histograms[img], hist):
            im = Image.fromarray(images[filename])
            im.save(os.path.join(des_path, filename))

