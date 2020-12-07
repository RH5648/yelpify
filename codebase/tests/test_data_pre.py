import os
import unittest

import pandas as pd

import codebase
from yelpify.data_preparation import round_of_rating, prepare_data

data_path = os.path.join(codebase.__path__[0], 'data')


class test_Perceptron(unittest.TestCase):

    def test_round(self):
        a = 3.4
        b = round_of_rating(a)
        c = 3.5
        # print(b)
        self.assertEqual(b, c)

    def test_predict(self):
        """
        Testing the perceptron

        """
        prepare_data(raw=False, round_ratings=False)


if __name__ == "__main__":
    unittest.main()