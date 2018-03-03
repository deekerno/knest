# UCF Senior Design 2017-18
# Group 38

import unittest
from models.bobo.model import Model
import json

TEST_HYPERPARAMETERS = 'tests/test_hp.json'


class ModelTestCase(unittest.TestCase):
    """Tests for `pill.py`."""

    def setUp(self):
        with open(TEST_HYPERPARAMETERS) as json_file:
            self.hp_content = json.load(json_file)
        self.test = Model("Test", hp_content=self.hp_content)

    def tearDown(self):
        # Tensorflow should handle the closing of the object
        pass

    def test_hyperparameter_read(self):
        """Are the hyperparameters successfully read into a dictionary?"""
        self.assertTrue(isinstance(self.hp_content, dict))

    def test_network_instantiation(self):
        """Is the network instantiated successfully?"""
        self.assertTrue(isinstance(self.test, Model))


if __name__ == '__main__':
    unittest.main()
