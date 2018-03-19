# UCF Senior Design 2017-18
# Group 38

import cv2
import numpy as np
import pywt

FIXED_SIZE = 1024
EDGE_THRESH = 35
MIN_ZERO = 0


def partition(emap, rows, cols):
    """
    Partition edge maps into windows
        emap: (array) edge map
        rows: (int) number of rows for each partition
        cols: (int) number of columns for each partition
    """
    h, w = emap.shape

    return (emap.reshape(h // rows, rows, -1, cols)
            .swapaxes(1, 2)
            .reshape(-1, rows, cols))


def haar_wavelet_transform(img):
    """
    Calculate haar wavelet transform for a given image
        img: (ndarray) grayscale image file
    """
    coeffs = pywt.dwt2(img, 'haar')

    # from pywt documentation: cA, (cH, cV, cD)
    # cA: LL
    # cH: LH
    # cV: HL
    # cD: HH
    LL, (LH, HL, HH) = coeffs

    return LL, LH, HL, HH


def calc_emap(LH, HL, HH):
    """
    Construct edge map
        LH: (array) vertical detail
        HL: (array) horizontal detail
        HH: (array) diagonal detail
    """
    # square root of LH^2 + HL^2 + HH^2
    return np.sqrt((np.power(LH, 2) + np.power(HL, 2) + np.power(HH, 2)))


def calc_emax(emap, window_size):
    """
    Calculate local maxima of each edgemap partition
        emap: (array) edge map
        window_size: (int) dimensions of each partition
    """
    # split edge map into partitions sized (window_sized X window_sized)
    result = partition(emap, window_size, window_size)
    max_points = []

    # from 0 to the total number of partitions
    for i in range(0, len(result)):
        # find the max of each partition and
        # add to total list of local maxes
        max_points.append(np.max(result[i]))

    # return list of all local maxes
    return max_points


def calc_values(emax1, emax2, emax3):
    """
    Calculate values necessary for blur detection
        emax1: (array) list of local maxima in first edgemap
        emax2: (array) list of local maxima in second edgemap
        emax3: (array) list of local maxima in third edgemap
    """
    # total number of edge points
    Nedge = 0
    # total number of Dirac- or Astep-Structure edge points
    Nda = 0
    # total number of Roof- or Gstep-Structure edge points
    Nrg = 0
    # total number of Roof- or Gstep-Structure edge points
    # that have lost their sharpness
    Nbrg = 0

    # all emax have the same length
    for i in range(0, len(emax1)):
        # Rule 1
        if emax1[i] > EDGE_THRESH or emax2[i] > EDGE_THRESH or emax3[i] > EDGE_THRESH:
            Nedge += 1

            # Rule 2
            if emax1[i] > emax2[i] and emax2[i] > emax3[i]:
                Nda += 1

            # Rule 3 & 4
            elif emax1[i] < emax2[i]:
                # Roof- or
                if emax2[i] < emax3[i]:
                    Nrg += 1

                # Roof structure
                elif emax2[i] > emax3[i]:
                    Nrg += 1

                # Rule 5
                if emax1[i] < EDGE_THRESH:
                    Nbrg += 1

    return Nedge, Nda, Nrg, Nbrg


def calc_intensities(img):
    """
    Calculate haar wavelet transforms, edge maps and maxima
        img: (ndarray) grayscale image file
    """
    # i = 1
    LL_1, LH_1, HL_1, HH_1 = haar_wavelet_transform(img)
    emap1 = calc_emap(LH_1, HL_1, HH_1)
    emax1 = calc_emax(emap1, 8)

    # i = 2
    LL_2, LH_2, HL_2, HH_2 = haar_wavelet_transform(LL_1)
    emap2 = calc_emap(LH_2, HL_2, HH_2)
    emax2 = calc_emax(emap2, 4)

    # i = 3
    LL_3, LH_3, HL_3, HH_3 = haar_wavelet_transform(LL_2)
    emap3 = calc_emap(LH_3, HL_3, HH_3)
    emax3 = calc_emax(emap3, 2)

    return emax1, emax2, emax3


def determine_resize(img):
    """
    Determine dimensions for resize. Default value is 1024; however,
    if image dimensions are smaller, resize accordingly.
    Smallest resize will be 256 or FIXED_SIZE divided by 4
        img: (ndarray) grayscale image file
    """
    size = FIXED_SIZE

    if len(img[0]) < size:
        if len(img[0]) < size // 2:
            return size // 4
        else:
            return size // 2

    return size


def blur_result(size, per, blur_extent):
    """
    Classify whether or not an image is blurry. Blur metrics vary based
    on resize dimensions
        size: (int) resized dimension
        per: (float) ratio of Dirac- and Astep-Structure to all edges
        blur_extent: (float) blur confident coefficient;
                     how many Roof- and Gstep-Structure edges are
    """
    if size == FIXED_SIZE:
        if (per <= MIN_ZERO and blur_extent > .83) or blur_extent >= .9:
            # blurry
            return 0
        else:
            # not blurry
            return 1
    else:
        if (per <= MIN_ZERO and blur_extent > .75) or blur_extent >= .9:
            # blurry
            return 0
        else:
            # not blurry
            return 1

    # preventive measure
    # default to not blurry result
    return 1


def detect_blur(img_path):
    """
    Method where final image and blur classification is returned
        img_path: (String) absolute path to image file
    """
    # convert image to numpy array
    image = cv2.imread(img_path)
    # convert to RGB
    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # determine dimensions for resize
    size = determine_resize(img)
    img = cv2.resize(img, (size, size))

    # calculate local maxima of each edge map (3 in total)
    emax1, emax2, emax3 = calc_intensities(img)

    # calculate Nedge, Nda, Nrg and Nbrg
    Nedge, Nda, Nrg, Nbrg = calc_values(emax1, emax2, emax3)

    # ratio of Dirac- and Astep-Structure to all edges
    # if divisor is 0, set per to 0
    per = Nedge == 0 and 0 or Nda / Nedge
    # blur confident coefficient; how many Roof- and Gstep-Structure edges are
    # blurred; if divisor is 0, set blur_extent to 0
    blur_extent = Nrg == 0 and 0 or Nbrg / Nrg

    # classify whether or not image is blurry
    result = blur_result(size, per, blur_extent)

    return image, result
