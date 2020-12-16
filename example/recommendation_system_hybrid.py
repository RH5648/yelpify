import sys
sys.path.append('./')

from yelpify.model_hybrid import train_model
from yelpify.recommend_hybrid import recommend_hybrid_item
from yelpify.recommend_hybrid import recommend_hybrid_user
from yelpify.recommend_hybrid import recommend_hybrid_new_user
from yelpify.recommend_hybrid import recommend_hybrid_new_item
from yelpify.data_preparation import prepare_data_features
import numpy as np

USER_ID = "avXKk5RYsDWeRgkHv1wfGQ"
ITEM_ID = "VMPSdoBgJuyS9t_x_caTig"

df = prepare_data_features(raw=False)

model_full, df_interactions, user_dict, item_dict, \
    user_feature_map, business_feature_map = train_model(
        df=df,
        user_id_col='user_id',
        item_id_col='business_id',
        item_name_col='name_business',
        evaluate=True)

# make prediction for known users
rec_list_user = recommend_hybrid_user(
    df=df,
    model=model_full,
    interactions=df_interactions,
    user_id=USER_ID,
    user_dict=user_dict,
    item_dict=item_dict,
    topn=10,
    show=True)

# make recommendation for known businesses
rec_list_item = recommend_hybrid_item(
    df=df,
    model=model_full,
    interactions=df_interactions,
    item_id=ITEM_ID,
    user_dict=user_dict,
    item_dict=item_dict,
    topn=10,
    show=True)

# make recommendation for new users
rec_list_new_user = recommend_hybrid_new_user(
    df=df,
    model=model_full,
    interactions=df_interactions,
    new_user_features=5*np.random.rand(1, 1),
    user_dict=user_dict,
    item_dict=item_dict,
    topn=10,
    show=True)

# make recommendation for new businesses
rec_list_new_item = recommend_hybrid_new_item(
    df=df,
    model=model_full,
    interactions=df_interactions,
    new_item_features=np.random.binomial(1, 0.05, size=(1, 89)),
    user_dict=user_dict,
    item_dict=item_dict,
    topn=10,
    show=True)
