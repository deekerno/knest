# UCF Senior Design 2017-18
# Group 38

import cv2

THRESHOLD = 100.0


def variance(image_path):
    """
        Calculate the variance of Laplacian distribution, which is a
        measure of the sharpness of an image. Blurry pictures have less
        regions containing rapid intensity changes, meaning that there will
        be less of a spread of responses (low variance). Sharper images will
        have a higher spread of responses (high variance).
            image_path:     (String) path to the image being tested
    """
    image = cv2.imread(image_path)
    greyscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(greyscale, cv2.CV_64F).var()


def check_sharpness(image, threshold=THRESHOLD):
    """
        Determine whether the sharpness of an image exceeds the threshold
        for variance. Those surpassing the threshold will return True.
            image:          (String) path to the image being tested
            threshhold      (Float)  minimum variance for acceptance
    """
    sharpness = variance(image)

    if sharpness > threshold:
        return True
    else:
        return False
