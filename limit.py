import utils.compare as compare

DIFF_THRES = 20
LIMIT = 2


def limit(img, std_hash, count):
    # calculate hash for given image
    cmp_hash = compare.calc_hash(img)

    # compare to standard
    diff = compare.compare(std_hash, cmp_hash)

    # if image is similar to standard
    if diff <= DIFF_THRES:
        # if there are 3 similar images already, return False
        if count >= LIMIT:
            return 'remove'

    else:
        return 'update_std'

    return 'continue'


def set_standard(images, filename):
    return filename, compare.calc_hash(images[filename]), 0
