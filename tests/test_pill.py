import unittest
from PIL import Image
from pill import Pill
import os

TEST_PHOTO = 'tests/test_photo.jpg'

class ImageTestCase(unittest.TestCase):
    """Tests for `pill.py`."""
    
    def setUp(self):
        self.img = Pill(TEST_PHOTO)

    def tearDown(self):
        self.img.pill.close()
    
    def test_pil_image_object_creation(self):
        """Is a test photo successfully opened as an image file?"""
        self.assertTrue(isinstance(self.img.pill, Image.Image))
    
    def test_existence_of_exif_data(self):
        """Is EXIF metadata being pulled out correctly?"""
        self.assertTrue(isinstance(self.img.exif_tags, dict))

    def test_cropping(self):
        box = (0, 0, 100, 100)
        self.img.crop_to_subject(box)
        self.assertEqual(self.img.pill.size, (100, 100))

if __name__ == '__main__':
    unittest.main()