"""
NAME
    test_recommend_known
DESCRIPTION
    This module test collaborative  recommendation filtering
FUNCTIONS
    test_recommend_known_user(self)
        test known user for recommendation system
        make sure no exceptions

    test_recommend_known_item(self)
        test known item for recommendation system
        make sure no exceptions
"""
import os
import unittest


import codebase
from yelpify.data_preparation import prepare_data
from yelpify.model_cf import train_model
from yelpify.recommend_known import recommend_known_user, recommend_known_item

data_path = os.path.join(codebase.__path__[0], 'data')
USER_ID = "avXKk5RYsDWeRgkHv1wfGQ"
ITEM_ID = "VMPSdoBgJuyS9t_x_caTig"


class TestModel(unittest.TestCase):

    def test_recommend_known_user(self):
        """
        Testing recommendation system for user in database
        smoke test, make sure no exceptions
        """
        df = prepare_data(raw=False)
        model_full, df_interactions, user_dict, item_dict = train_model(
            df=df,
            user_id_col='user_id',
            item_id_col='business_id',
            item_name_col='name_business',
            evaluate=True)
        rec_list_user = recommend_known_user(
            model=model_full,
            interactions=df_interactions,
            user_id=USER_ID,
            user_dict=user_dict,
            item_dict=item_dict,
            topn=10,
            show=True)
        self.assertEqual(len(rec_list_user), 10)

    def test_recommend_known_item(self):
        """
        Testing recommendation system for item in database
        smoke test, make sure no exceptions
        """
        df = prepare_data(raw=False)
        model_full, df_interactions, user_dict, item_dict = train_model(
            df=df,
            user_id_col='user_id',
            item_id_col='business_id',
            item_name_col='name_business',
            evaluate=True)
        rec_list_item = recommend_known_item(
            model=model_full,
            interactions=df_interactions,
            item_id=ITEM_ID,
            user_dict=user_dict,
            item_dict=item_dict,
            topn=10,
            show=True)
        self.assertEqual(len(rec_list_item), 10)


if __name__ == "__main__":
    unittest.main()
