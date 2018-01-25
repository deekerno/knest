# UCF Senior Design 2017-18
# Group 38

import unittest
import blur

TEST_PHOTO = 'tests/test_photo.jpg'


class BlurTestCase(unittest.TestCase):
    """Tests for `pill.py`."""

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_check_sharpness(self):
        """Does a sharp image successfully pass the sharpness check?"""
        self.assertTrue(blur.check_sharpness(TEST_PHOTO))
