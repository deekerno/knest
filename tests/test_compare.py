# UCF Senior Design 2017-18
# Group 38

import cv2
import unittest
import utils.compare as compare

TEST_PHOTO = 'tests/test_photo.jpg'
TEST_PHOTO2 = 'tests/test_photo2.jpg'
DIFF = 34


class CompareTestCase(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_calc_hash(self):
        img = cv2.imread(TEST_PHOTO)
        hsh = compare.calc_hash(img)
        self.assertEqual(str(hsh), 'f8f8e1e72103719c')

    def test_compare_to_self(self):
        img1 = cv2.imread(TEST_PHOTO)
        img2 = cv2.imread(TEST_PHOTO)
        hash1 = compare.calc_hash(img1)
        hash2 = compare.calc_hash(img2)
        self.assertEqual(compare.compare(hash1, hash2), 0)

    def test_compare_diff(self):
        img1 = cv2.imread(TEST_PHOTO)
        img2 = cv2.imread(TEST_PHOTO2)
        hash1 = compare.calc_hash(img1)
        hash2 = compare.calc_hash(img2)
        self.assertEqual(compare.compare(hash1, hash2), 36)


if __name__ == '__main__':
    unittest.main()
