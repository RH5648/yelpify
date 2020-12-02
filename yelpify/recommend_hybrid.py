import pandas as pd
import numpy as np

# function to make prediction for known items
def recommend_hybrid_item(model, interactions, item_id, user_dict, item_dict, topn, show=True):
    """
    Funnction to produce a list of top N interested users for a given item
    Required Input -
        - model = Trained matrix factorization model
        - interactions = dataset used for training the model
        - item_id = item ID for which we need to generate recommended users
        - user_dict =  Dictionary type input containing user_id as key and interaction_index as value
        - item_dict = Dictionary type input containing item_id as key and item_name as value
        - topn = Number of users needed as an output
    Expected Output -
        - user_list = List of recommended users 
    """
    print('Recommending users for item {}...'.format(item_id))
    n_users, n_items = interactions.shape
    x = np.array(interactions.columns)
    scores = pd.Series(model.predict(np.arange(n_users), np.repeat(x.searchsorted(item_id),n_users)))
    user_list = list(interactions.index[scores.sort_values(ascending=False).head(topn).index])
    if show == True:
        print("Recommended Users:")
        counter = 1
        for i in user_list:
            print(str(counter) + '- ' + i)
            counter+=1
    return user_list 

# function to make prediction for known users
def recommend_hybrid_user(model, interactions, user_id, user_dict, item_dict, topn, new_only = False, threshold = 3, show = True):
    """
    Function to produce user recommendations
    Required Input - 
        - model = Trained matrix factorization model
        - interactions = dataset used for training the model
        - user_id = user ID for which we need to generate recommendation
        - user_dict = Dictionary type input containing user_id as key and interaction_index as value
        - item_dict = Dictionary type input containing item_id as key and item_name as value
        - threshold = value above which the rating is favorable in new interaction matrix
        - topn = Number of output recommendation needed
    Expected Output - 
        - Prints list of items the given user has already visited
        - Prints list of N recommended items  which user hopefully will be interested in
    """
    print('Recommending items for user {}...'.format(user_id))
    n_users, n_items = interactions.shape
    user_x = user_dict[user_id]
    scores = pd.Series(model.predict(user_x,np.arange(n_items)))
    scores.index = interactions.columns
    scores = list(pd.Series(scores.sort_values(ascending=False).index))
    known_items = list(pd.Series(interactions.loc[user_id,:] \
                                [interactions.loc[user_id,:] > threshold].index).sort_values(ascending=False))
    if new_only:
        scores = [x for x in scores if x not in known_items]
    item_list = scores[:topn]
    known_items = list(pd.Series(known_items).apply(lambda x: item_dict[x]))
    recommended_items = list(pd.Series(item_list).apply(lambda x: item_dict[x]))
    if show == True:
        print("Known Likes:")
        counter = 1
        for i in known_items:
            print(str(counter) + '- ' + i)
            counter+=1

        print("Recommended Items:")
        counter = 1
        for i in recommended_items:
            print(str(counter) + '- ' + i)
            counter+=1
    return item_list