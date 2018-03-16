from PIL import Image
import imagehash


def calc_hash(img):
    return imagehash.whash(Image.fromarray(img))


def compare(hash1, hash2):
    return hash1 - hash2
