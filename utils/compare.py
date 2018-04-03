# UCF Senior Design 2017-18
# Group 38

from PIL import Image
import cv2
import imagehash
import math
import numpy as np

DIFF_THRES = 20
LIMIT = 2
RESIZE = 1000


def calc_hash(img):
    """
    Calculate the wavelet hash of the image
        img: (ndarray) image file
    """
    # resize image if height > 1000
    img = resize(img)
    return imagehash.whash(Image.fromarray(img))


def compare(hash1, hash2):
    """
    Calculate the difference between two images
        hash1: (array) first wavelet hash
        hash2: (array) second wavelet hash
    """
    return hash1 - hash2


def limit(img, std_hash, count):
    """
    Determine whether image should be removed from image dictionary in main.py
        img: (ndarray) image file
        std_hash: (array) wavelet hash of comparison standard
        count: (int) global count of images similar to comparison standard
    """
    # calculate hash for given image
    cmp_hash = calc_hash(img)

    # compare to standard
    diff = compare(std_hash, cmp_hash)

    # image is similar to standard
    if diff <= DIFF_THRES:
        # if there are 3 similar images already, remove image
        if count >= LIMIT:
            return 'remove'

    # non-similar image found
    else:
        # update comparison standard
        return 'update_std'

    # else continue reading images with same standard
    return 'continue'


def resize(img):
    """
    Resize an image
        img: (ndarray) RGB color image
    """
    # get dimensions of image
    width = np.shape(img)[1]
    height = np.shape(img)[0]

    # if height of image is greater than 1000, resize it to 1000
    if width > RESIZE:
        # keep resize proportional
        scale = RESIZE / width
        resized_img = cv2.resize(
            img, (RESIZE, math.floor(height / scale)), cv2.INTER_AREA)
        # return resized image
        return resized_img

    # if height of image is less than 1000, return image unresized
    return img


def set_standard(images, filename):
    """
    Set new comparison standard and update information
        images: (dictionary) dictionary containing all the image data
        filename: (String) name of the image file
    """
    return filename, calc_hash(images[filename]), 0
