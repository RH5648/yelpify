import os
import unittest

import codebase
from yelpify.data_preparation import prepare_data_features
from yelpify.model_hybrid import train_model, evaluate_model

data_path = os.path.join(codebase.__path__[0], 'data')


class test_model(unittest.TestCase):

    def test_train_model(self):
        print("test_train_model")
        df = prepare_data_features(raw=False)
        train_model(df, user_id_col='user_id', item_id_col='business_id',
                    item_name_col='name_business', evaluate=False)

    def test_evaluate_model(self):
        print("test_evaluate_model")
        df = prepare_data_features(raw=False)
        evaluate_model(df, user_id_col='user_id',
                       item_id_col='business_id', stratify=None)


if __name__ == "__main__":
    unittest.main()