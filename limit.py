from PIL import Image
import utils.compare as compare
import os

DIFF_THRES = 20
LIMIT = 2


def limit(images, des_path):
    # initialized variable
    std = ''
    count = 0
    files = images.keys()

    # go through dictionary of images
    for filename in files:
        if std == '':
            std, std_hash, count = set_standard(images, filename, des_path)

        else:
            # calculate hash for working image
            cmp_hash = compare.calc_hash(images[filename])
            # compare to standard
            diff = compare.compare(std_hash, cmp_hash)

            # image is similar to standard
            if diff <= DIFF_THRES:
                # if there are 3 similar images already, remove the rest from
                # the image dictionary
                if count >= LIMIT:
                    images.pop(filename)

                # keep track of number of similar images
                count += 1

            # a non-similar image has been found, make it the new standard
            # for comparison
            else:
                std, std_hash, count = set_standard(images, filename, des_path)

    return images


def set_standard(images, filename, des_path):
    # automatically save standard to processed folder
    # img = Image.fromarray(images[filename])
    # img.save(os.path.join(des_path, filename))

    return filename, compare.calc_hash(images[filename]), 0
