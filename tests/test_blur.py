# UCF Senior Design 2017-18
# Group 38

import unittest
import utils.blur as blur

TEST_PHOTO = 'tests/test_blurry_photo.JPG'


class BlurTestCase(unittest.TestCase):
    """Tests for `blur.py`."""

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_detect_blur(self):
        """Does an objectively blurry image fail blur detection?"""
        img, result = blur.detect_blur(TEST_PHOTO)
        self.assertFalse(result)


if __name__ == '__main__':
    unittest.main()
