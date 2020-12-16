"""
NAME
    test_model_hybrid
DESCRIPTION
    This module test model data_preparation
        and generate features as needed.
FUNCTIONS
    test_train_model(self)
        make sure no exceptions during model training

    test_evaluate_model(self)
        make sure model devaluation has no exceptions

"""
import os
import unittest

import codebase
from yelpify.data_preparation import prepare_data_features
from yelpify.model_hybrid import train_model, evaluate_model

data_path = os.path.join(codebase.__path__[0], 'data')


class test_model(unittest.TestCase):

    def test_train_model(self):
        """
        Testing process of training model
        smoke test, make sure no exceptions
        training data may take a while
        """
        df = prepare_data_features(raw=False)
        train_model(df, user_id_col='user_id', item_id_col='business_id',
                    item_name_col='name_business', evaluate=False)

    def test_evaluate_model(self):
        """
        Testing process of training model
        smoke test, make sure no exceptions
        """
        df = prepare_data_features(raw=False)
        evaluate_model(df, user_id_col='user_id',
                       item_id_col='business_id', stratify=None)


if __name__ == "__main__":
    unittest.main()
