# UCF Senior Design 2017-18
# Group 38

from PIL import ExifTags
from PIL import Image
from PIL import ImageOps

class Pill(object):
    """
    When an image object is instantiated, a Pillow (PIL) image 
    object (henceforth referred to as 'pill') is instantly created.
    From here, one can use the defined methods for cropping and color
    transformations.
    """
    def __init__(self, arg):
        self.pill = Image.open(arg)
        self.path = self.pill.filename
        self.file_obj = open(self.path)
        self.exif_tags = self.pill._getexif()
        self.gray_con = None

    def crop_to_subject(self, box):
        """
        Box should be a 4-tuple (left, upper, right, lower) containing
        the pixel location coordinates of the area to which the image
        will be cropped. Note that (0,0) is the upper left corner.
        """
        self.pill = self.pill.crop(box)

    def gray_contrast(self):
        """
        Experimental method that first converts the image to grayscale
        and then maximizes image contrast. The darkest pixel in the
        image becomes black and the lightest pixel becomes white. This
        should help with eye detection.
        """
        self.gray_con = ImageOps.autocontrast(ImageOps.grayscale(self.pill))