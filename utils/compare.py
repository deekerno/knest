# UCF Senior Design 2017-18
# Group 38

from PIL import Image
import imagehash

DIFF_THRES = 20
LIMIT = 2


def calc_hash(img):
    """
    Calculate the wavelet hash of the image
        img: (ndarray) image file
    """
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


def set_standard(images, filename):
    """
    Set new comparison standard and update information
        images: (dictionary) dictionary containing all the image data
        filename: (String) name of the image file
    """
    return filename, calc_hash(images[filename]), 0
