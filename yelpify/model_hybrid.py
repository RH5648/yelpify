"""
NAME
    model_hybrid
DESCRIPTION
    This module provides access to functions that train and evaluate
        models using hybrid filtering.
FUNCTIONS
    get_users_features_tuple(user)
    get_items_features_tuple(item, categories)
    train_model(df, user_id_col, item_id_col, item_name_col, evaluate)
        Return the trained model, dataset with user-item interactions,
            user dictionary and item dictionary.
    evaluate_model(df, user_id_col, item_id_col)
        Return the auc-roc score of the training and testing sets.
"""

import pandas as pd
from lightfm import LightFM
from lightfm.evaluation import auc_score
from sklearn.model_selection import train_test_split
from lightfm.data import Dataset


def get_users_features_tuple(user):
    """ Get user's feature and return a tuple
    Arg:
        user: the user line in the dataframe
    Returns:
        Tuple(user id string, dict{user feature: feature value})
    """

    return (user[0], {'average_stars': float(user[5])})


def get_items_features_tuple(item, categories):
    """ Get the item's feature and return a tuple
    Arg:
        item: one item line in the dataframe
        categories: item features list
    Returns:
        Tuple(item id string, dict{item feature: feature value})
    """

    item.tolist()
    item_id = item[1]
    features_ind = item[10:]
    features = {}
    for i in range(len(features_ind)):
        features[categories[i]] = float(features_ind[i])
    return (item_id, features)


def train_model(
               df, user_id_col='user_id', item_id_col='business_id',
               item_name_col='name_business', evaluate=True):
    """ Train the model using collaborative filtering.
    Args:
        df: the input dataframe.
        user_id_col: user id column.
        item_id_col: item id column.
        item_name_col: item name column.
        evaluate: if evaluate the model performance.
    Returns:
        model_full: the trained model.
        df_interactions: dataframe with user-item interactions.
        user_dict: user dictionary containing user_id as key and
            interaction_index as value.
        item_dict: item dictionary containing item_id as key and
            item_name as value.
        user_feature_map: the feature map of users
        business_feature_map: the feature map of items
    """
    if evaluate:
        print('Evaluating model...')
        evaluate_model(df, user_id_col='user_id', item_id_col='business_id')
    print('Training model...')

    # build recommendations for known users and known businesses
    # with collaborative filtering method
    ds_full = Dataset()
    # we call fit to supply userid, item id and user/item features
    user_cols = ['user_id', 'average_stars']
    categories = [c for c in df.columns if c[0].isupper()]
    item_cols = ['business_id', 'state']

    for i in df.columns[10:]:
        item_cols.append(str(i))

    user_features = user_cols[1:]
    item_features = item_cols[2:]

    ds_full.fit(
        df[user_id_col].unique(),  # all the users
        df[item_id_col].unique(),  # all the items
        user_features=user_features,  # additional user features
        item_features=item_features
         )

    df_users = df.drop_duplicates(user_id_col)
    # df_users = df[df.duplicated(user_id_col) == False]
    users_features = []
    for i in range(len(df_users)):
        users_features.append(get_users_features_tuple(df_users.values[i]))
    users_features = ds_full.build_user_features(
        users_features, normalize=False)

    items = df.drop_duplicates(item_id_col)
    # items = df[df.duplicated(item_id_col) == False]
    items_features = []
    for i in range(len(items)):
        items_features.append(get_items_features_tuple(
            items.values[i], categories))
    items_features = ds_full.build_item_features(
        items_features, normalize=False)

    (interactions, weights) = ds_full.build_interactions(
        [(x[0], x[1], x[2]) for x in df.values])
    # model
    model_full = LightFM(
        no_components=100, learning_rate=0.05, loss='warp', max_sampled=50)
    model_full.fit(
        interactions, user_features=users_features,
        item_features=items_features, sample_weight=weights,
        epochs=10, num_threads=10)
    # mapping
    user_id_map, user_feature_map, business_id_map, business_feature_map = \
        ds_full.mapping()

    # data preparation
    df_interactions = pd.DataFrame(weights.todense())
    df_interactions.index = list(user_id_map.keys())
    df_interactions.columns = list(business_id_map.keys())
    user_dict = user_id_map
    item_dict = df.set_index(item_id_col)[item_name_col].to_dict()
    return model_full, df_interactions, user_dict, \
        item_dict, user_feature_map, business_feature_map


def evaluate_model(
                  df, user_id_col='user_id',
                  item_id_col='business_id', stratify=None):
    """ Model evaluation.
    Args:
        df: the input dataframe.
        user_id_col: user id column.
        item_id_col: item id column.
        stratify: if use stratification.
    No return value
    """
    # create test and train datasets
    train, test = train_test_split(df, test_size=0.2, stratify=stratify)
    ds = Dataset()
    # we call fit to supply userid, item id and user/item features
    user_cols = ['user_id', 'average_stars']
    categories = [c for c in df.columns if c[0].isupper()]
    item_cols = ['business_id', 'state']

    for i in df.columns[10:]:
        item_cols.append(str(i))

    user_features = user_cols[1:]
    item_features = item_cols[2:]

    ds.fit(
        df[user_id_col].unique(),  # all the users
        df[item_id_col].unique(),  # all the items
        user_features=user_features,  # additional user features
        item_features=item_features
         )

    train_users = train.drop_duplicates('user_id')
    # train_users = train[train.duplicated('user_id') == False]
    train_user_features = []
    for i in range(len(train_users)):
        train_user_features.append(get_users_features_tuple(
            train_users.values[i]))
    train_user_features = ds.build_user_features(
        train_user_features, normalize=False)

    test_users = test.drop_duplicates('user_id')
    # test_users = test[test.duplicated('user_id') == False]
    test_user1_features = []
    for i in range(len(test_users)):
        test_user1_features.append(get_users_features_tuple(
            test_users.values[i]))
    test_user_features = ds.build_user_features(
        test_user1_features, normalize=False)

    train_items = train.drop_duplicates('business_id')
    # train_items = train[train.duplicated('business_id') == False]
    train_item1_features = []
    for i in range(len(train_items)):
        train_item1_features.append(get_items_features_tuple(
            train_items.values[i], categories))
    train_item_features = ds.build_item_features(
        train_item1_features, normalize=False)

    test_items = test.drop_duplicates('business_id')
    # test_items = test[test.duplicated('business_id') == False]
    test_item_features = []
    for i in range(len(test_items)):
        test_item_features.append(get_items_features_tuple(
            test_items.values[i], categories))
    test_item_features = ds.build_item_features(
        test_item_features, normalize=False)

    # plugging in the interactions and their weights
    (train_interactions, train_weights) = ds.build_interactions(
        [(x[0], x[1], x[2]) for x in train.values])
    (test_interactions, test_weights) = ds.build_interactions(
        [(x[0], x[1], x[2]) for x in test.values])

    # model
    model = LightFM(
        no_components=100, learning_rate=0.05, loss='warp', max_sampled=50)
    model.fit(
        train_interactions, user_features=train_user_features,
        item_features=train_item_features, sample_weight=train_weights,
        epochs=10, num_threads=10)

    # auc-roc
    train_auc = auc_score(
        model, train_interactions, user_features=train_user_features,
        item_features=train_item_features, num_threads=20).mean()
    print('Training set AUC: %s' % train_auc)
    test_auc = auc_score(
        model, test_interactions, user_features=test_user_features,
        item_features=test_item_features, num_threads=20).mean()
    print('Testing set AUC: %s' % test_auc)
