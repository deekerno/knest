import unittest
import compare

TEST_PHOTO = 'tests/test_photo.jpg'
TEST_PHOTO2 = 'tests/test_photo2.jpg'
DIFF = 34


class CompareTestCase(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_calc_hash(self):
        hsh = compare.calc_hash(TEST_PHOTO)
        self.assertEqual(str(hsh), 'f8f8e0e72107719c')

    def test_compare_to_self(self):
        hash1 = compare.calc_hash(TEST_PHOTO)
        hash2 = compare.calc_hash(TEST_PHOTO)
        self.assertEqual(compare.compare(hash1, hash2), 0)

    def test_compare_diff(self):
        hash1 = compare.calc_hash(TEST_PHOTO)
        hash2 = compare.calc_hash(TEST_PHOTO2)
        self.assertEqual(compare.compare(hash1, hash2), 34)

if __name__ == '__main__':
    unittest.main()
