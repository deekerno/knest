from PIL import Image
import imagehash

DIFF_THRES = 20
LIMIT = 2


def calc_hash(img):
    return imagehash.whash(Image.fromarray(img))


def compare(hash1, hash2):
    return hash1 - hash2


def limit(img, std_hash, count):
    # calculate hash for given image
    cmp_hash = calc_hash(img)

    # compare to standard
    diff = compare(std_hash, cmp_hash)

    # if image is similar to standard
    if diff <= DIFF_THRES:
        # if there are 3 similar images already, return False
        if count >= LIMIT:
            return 'remove'

    else:
        return 'update_std'

    return 'continue'


def set_standard(images, filename):
    return filename, calc_hash(images[filename]), 0
