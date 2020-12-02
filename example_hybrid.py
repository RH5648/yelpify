from yelpify.model_hybrid import train_model
from yelpify.recommend_hybrid import recommend_hybrid_item, recommend_hybrid_user
from yelpify.data_preparation import prepare_data

USER_ID = "qRWzBX1q07ZuPgaTXB_4JA"
ITEM_ID = "VMPSdoBgJuyS9t_x_caTig"

df = prepare_data(raw=False)

model_full, df_interactions, user_dict, item_dict = train_model(
    df=df,
    user_id_col='user_id',
    item_id_col='business_id',
    item_name_col='name_business',
    evaluate=True)

# make prediction for known users
rec_list_user = recommend_hybrid_user(
    model = model_full,
    interactions = df_interactions,
    user_id = USER_ID,
    user_dict = user_dict,
    item_dict = item_dict,
    topn = 10,
    show = True)

# make recommendation for known businesses
rec_list_item = recommend_hybrid_item(
    model = model_full,
    interactions = df_interactions,
    item_id = ITEM_ID,
    user_dict = user_dict,
    item_dict = item_dict,
    topn = 10,
    show = True)
