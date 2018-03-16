import cv2
import numpy as np
import pywt

FIXED_SIZE = 1024
EDGE_THRESH = 35
MIN_ZERO = 0

# partition edge maps into windows
def partition(emap, rows, cols):
    h, w = emap.shape

    return (emap.reshape(h // rows, rows, -1, cols)
            .swapaxes(1, 2)
            .reshape(-1, rows, cols))


# haar wavelet transform calculation
def haar_wavelet_transform(img):
    coeffs = pywt.dwt2(img, 'haar')

    # from pywt documentation: cA, (cH, cV, cD)
    # cA: LL
    # cH: LH
    # cV: HL
    # cD: HH
    LL, (LH, HL, HH) = coeffs

    return LL, LH, HL, HH


# calc emap, iterating over LL
def calc_emap(LH, HL, HH):
    # square root of LH^2 + HL^2 + HH^2
    return np.sqrt((np.power(LH, 2) + np.power(HL, 2) + np.power(HH, 2)))
    

# find local maxima of each edgemap partition
def calc_emax(emap, window_size):
    # split edge map into windows sized (window_size X window_size)
    # find the local maxima of each window
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


# default resize value is 1024; however, if image dimensions are smaller,
# resize accordingly. Smallest resize will be 256 or FIXED_SIZE divided by 4
def determine_resize(img):
    size = FIXED_SIZE

    if len(img[0]) < size:
        if len(img[0]) < size // 2:
            return size // 4
        else:
            return size // 2

    return size


# classify whether image is blurry or not blurry
# blur metrics vary based on resize value
def blur_result(size, per, blur_extent):
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
    image = cv2.imread(img_path)
    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    size = determine_resize(img)
    img = cv2.resize(img, (size, size))

    emax1, emax2, emax3 = calc_intensities(img)

    # calculate Nedge, Nda, Nrg and Nbrg
    Nedge, Nda, Nrg, Nbrg = calc_values(emax1, emax2, emax3)

    # ratio of Dirac- and Astep-Structure to all edges
    # if divisor is 0, set per to 0
    per = Nedge == 0 and 0 or Nda / Nedge
    # blur confident coefficient; how many Roof- and Gstep-Structure edges are
    # blurred; if divisor is 0, set blur_extent to 0
    blur_extent = Nrg == 0 and 0 or Nbrg / Nrg

    result = blur_result(size, per, blur_extent)

    return image, result
