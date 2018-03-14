import cv2
import numpy as np
from PIL import Image
import pywt

FIXED_SIZE = 1024
EDGE_THRESH = 35
MIN_ZERO = 0


def img_handler(img_path):
    try:
        # file is an image
        img = Image.open(img_path)
        img.close()
        return True

    except IOError:
        # file is not an image
        return False

# partition edge maps into windows
def partition(emap, rows, cols):
    h, w = emap.shape

    return (emap.reshape(h // rows, rows, -1, cols)
            .swapaxes(1, 2)
            .reshape(-1, rows, cols))

# haar wavelet transform calculation
def haar_wavelet_transform(img):
    coeffs = pywt.dwt2(img, 'haar')

    # cA, (cH, cV, cD)
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


def detect_blur(img_path):
    image = cv2.imread(img_path)
    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (FIXED_SIZE, FIXED_SIZE))

    emax1, emax2, emax3 = calc_intensities(img)

    # calculate Nedge, Nda, Nrg and Nbrg
    Nedge, Nda, Nrg, Nbrg = calc_values(emax1, emax2, emax3)

    # ratio of Dirac- and Astep-Structure to all edges
    per = Nda / Nedge
    # blur confident coefficient; how many Roof- and Gstep-Structure edges are blurred
    blur_extent = Nbrg / Nrg

    if (per <= MIN_ZERO and blur_extent > .83) or blur_extent >= .9:
        # blurry
        result = 0
    else:
        # not blurry
        result = 1

    return image, result
