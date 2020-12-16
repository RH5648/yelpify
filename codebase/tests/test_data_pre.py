"""
NAME
    test_data_pre
DESCRIPTION
    This module test model data_preparation
        and generate features as needed.
FUNCTIONS
    test_round(self)
        assert value with human calculated value

    test_preparation(self)
        make sure data preparation has no exceptions

"""
import os
import unittest

import codebase
from yelpify.data_preparation import round_of_rating, prepare_data

data_path = os.path.join(codebase.__path__[0], 'data')


class test_preparation(unittest.TestCase):

    def test_round(self):
        """
        Testing round
        """
        a = 3.4
        b = round_of_rating(a)
        c = 3.5
        # print(b)
        self.assertEqual(b, c)

    def test_preparation(self):
        """
        Testing the data preparation
        smoke test make sure no exceptions
        """
        prepare_data(raw=False, round_ratings=False)


if __name__ == "__main__":
    unittest.main()
