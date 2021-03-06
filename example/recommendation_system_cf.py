"""
NAME
    recommendation_system_cf
DESCRIPTION
    A demo of our recommendation system if this user or item is in the database
"""
import sys
from yelpify.model_cf import train_model
from yelpify.recommend_known import recommend_known_item, recommend_known_user
from yelpify.data_preparation import prepare_data
sys.path.append('./')

USER_ID = "avXKk5RYsDWeRgkHv1wfGQ"
ITEM_ID = "VMPSdoBgJuyS9t_x_caTig"

df = prepare_data(raw=False)

model_full, df_interactions, user_dict, item_dict = train_model(
    df=df,
    user_id_col='user_id',
    item_id_col='business_id',
    item_name_col='name_business',
    evaluate=True)

# make prediction for known users
rec_list_user = recommend_known_user(
    model=model_full,
    interactions=df_interactions,
    user_id=USER_ID,
    user_dict=user_dict,
    item_dict=item_dict,
    new_only=False,
    topn=10,
    threshold=3,
    show=True)

# make recommendation for known businesses
rec_list_item = recommend_known_item(
    model=model_full,
    interactions=df_interactions,
    item_id=ITEM_ID,
    user_dict=user_dict,
    item_dict=item_dict,
    topn=10,
    show=True)
