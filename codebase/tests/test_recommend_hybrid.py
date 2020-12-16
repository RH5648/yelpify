"""
NAME
    test_recommend_hybrid
DESCRIPTION
    This module test hybrid recommendation filtering
FUNCTIONS
    test_hybrid_item(self)
        test known item for recommendation system
        make sure no exceptions

    test_hybrid_new_item(self)
        test new item for recommendation system
        make sure no exceptions
"""
import os
import unittest
import numpy as np
import codebase
from yelpify.model_hybrid import train_model
from yelpify.recommend_hybrid import recommend_hybrid_item, \
    recommend_hybrid_new_item
from yelpify.data_preparation import prepare_data_features

USER_ID = "avXKk5RYsDWeRgkHv1wfGQ"
ITEM_ID = "VMPSdoBgJuyS9t_x_caTig"
data_path = os.path.join(codebase.__path__[0], 'data')
df = prepare_data_features(raw=False)

model_full, df_interactions, user_dict, item_dict, \
    user_feature_map, business_feature_map = train_model(
        df=df,
        user_id_col='user_id',
        item_id_col='business_id',
        item_name_col='name_business',
        evaluate=True)


class TestModel(unittest.TestCase):

    def test_hybrid_item(self):
        """
         Testing recommendation system for item in database
         smoke test, make sure no exceptions
        """
        rec_list_item = recommend_hybrid_item(
            df=df,
            model=model_full,
            interactions=df_interactions,
            item_id=ITEM_ID,
            user_dict=user_dict,
            item_dict=item_dict,
            topn=10,
            show=True)
        self.assertEqual(len(rec_list_item), 10)

    def test_hybrid_new_item(self):
        """
         Testing recommendation system for item not in database
         smoke test, make sure no exceptions
        """
        rec_list_new_item = recommend_hybrid_new_item(
            df=df,
            model=model_full,
            interactions=df_interactions,
            new_item_features=np.random.binomial(1, 0.05, size=(1, 89)),
            user_dict=user_dict,
            item_dict=item_dict,
            topn=10,
            show=True)
        self.assertEqual(len(rec_list_new_item), 10)


if __name__ == "__main__":
    unittest.main()
