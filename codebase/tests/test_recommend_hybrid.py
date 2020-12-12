import os
import unittest


import codebase
from yelpify.model_hybrid import train_model
from yelpify.recommend_hybrid import recommend_hybrid_item,\
    recommend_hybrid_user
from yelpify.data_preparation import prepare_data_features

USER_ID = "avXKk5RYsDWeRgkHv1wfGQ"
ITEM_ID = "VMPSdoBgJuyS9t_x_caTig"
data_path = os.path.join(codebase.__path__[0], 'data')
df = prepare_data_features(raw=False)

model_full, df_interactions, user_dict, item_dict = train_model(
    df=df,
    user_id_col='user_id',
    item_id_col='business_id',
    item_name_col='name_business',
    evaluate=True)


class TestModel(unittest.TestCase):

    def test_hybrid_item(self):
        rec_list_user = recommend_hybrid_user(
            model=model_full,
            interactions=df_interactions,
            user_id=USER_ID,
            user_dict=user_dict,
            item_dict=item_dict,
            topn=10,
            show=True)
        self.assertEqual(len(rec_list_user), 10)

    def test_evaluate_model(self):
        rec_list_item = recommend_hybrid_item(
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
