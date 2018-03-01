# UCF Senior Design 2017-18
# Group 38

import cv2
import scipy.fftpack as fp
import numpy as np
import pywt

LAP_THRESHOLD = 100.0
FFT_THRESHOLD = 150.0
K_SIZE = 5


def variance(image):
    """
        Calculate the variance of Laplacian distribution, which is a
        measure of the sharpness of an image. Blurry pictures have less
        regions containing rapid intensity changes, meaning that there will
        be less of a spread of responses (low variance). Sharper images will
        have a higher spread of responses (high variance).
            image_path:     (String) path to the image being tested
    """
    #greyscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(image, cv2.CV_64F).var()


def teng(image, k_size=K_SIZE):
    #greyscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gauss_x = cv2.Sobel(image, cv2.CV_64F, 1, 0)
    gauss_y = cv2.Sobel(image, cv2.CV_64F, 1, 0)
    return np.mean((gauss_x * gauss_x) + (gauss_y * gauss_y))


def fft(image, threshold=FFT_THRESHOLD):
    #greyscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    fft = fp.rfft(fp.rfft(image, axis=0), axis=1)
    result = np.mean(fft)
    return result, result > threshold


def lapm(image):
    #greyscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    kernel = np.array([-1, 2, 1])
    lap_x = np.abs(cv2.filter2D(image, -1, kernel))
    lap_y = np.abs(cv2.filter2D(image, -1, kernel.T))
    return np.mean(lap_x + lap_y)


def sum_wave(image):
    coeffs = pywt.dwt2(image, 'haar')
    cA, (cH, cV, cD) = coeffs
    total = np.sum(cA) + np.sum(cH) + np.sum(cV) + np.sum(cD)
    return total


def check_sharpness(image_path, threshold=LAP_THRESHOLD):
    """
        Determine whether the sharpness of an image exceeds the threshold
        for variance. Those surpassing the threshold will return True.
            image:          (String) path to the image being tested
            threshhold      (Float)  minimum variance for acceptance
    """
    sharpness = variance(image_path)

    return sharpness, sharpness > threshold
