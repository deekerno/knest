# UCF Senior Design 2017-18
# Group 38

import cv2
import unittest
import utils.image_man as im

GOOD_EXIF = 'tests/images/test_good_exif.JPG'
EMPTY_EXIF = 'tests/images/test_empty_exif.JPG'
NONEXIF = 'tests/images/test_nonexif_image_format.png'


class ImageManipulationTestCase(unittest.TestCase):
    """Tests for `image_man.py`."""

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_empty_exif(self):
        """An image with empty EXIF data should lead to a NoneType return"""
        image = cv2.imread(EMPTY_EXIF)
        exif_data = im.exif(EMPTY_EXIF, image)
        self.assertIsNone(exif_data)

    def test_proper_exif_loading(self):
        """An image with EXIF data should lead to a return of a bytes object"""
        image = cv2.imread(GOOD_EXIF)
        exif_data = im.exif(GOOD_EXIF, image)
        self.assertIsInstance(exif_data, bytes)

    def test_nonexif_image_format(self):
        """An image of a format that does not typicallysupport EXIF should lead to a NoneType return"""
        image = cv2.imread(NONEXIF)
        exif_data = im.exif(NONEXIF, image)
        self.assertIsNone(exif_data)

    #def test_proper_image_crop(self):
    #    pass


if __name__ == '__main__':
    unittest.main()
