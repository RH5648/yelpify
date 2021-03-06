"""
NAME
    model_cf

DESCRIPTION
    This module provides access to functions that train and
        evaluate models using collaborative filtering.

FUNCTIONS
    train_model(df, user_id_col, item_id_col, item_name_col, evaluate)
        Return the trained model, dataset with user-item
        interactions, user dictionary and item dictionary.

    evaluate_model(df, user_id_col, item_id_col)
        Return the auc-roc score of the training and testing sets.
"""

import pandas as pd
from lightfm import LightFM
from lightfm.evaluation import auc_score
from sklearn.model_selection import train_test_split
from lightfm.data import Dataset


def train_model(df, user_id_col='user_id', item_id_col='business_id',
                item_name_col='name_business', evaluate=True):
    """Train the model using collaborative filtering.

    Args:
        df: the input dataframe.
        user_id_col: user id column.
        item_id_col: item id column.
        item_name_col: item name column.
        evaluate: if evaluate the model performance.

    Returns:
        model_full: the trained model.
        df_interactions: dataframe with user-item interactions.
        user_dict: user dictionary containing user_id as
            key and interaction_index as value.
        item_dict: item dictionary containing item_id
            as key and item_name as value.

    """
    if evaluate:
        print('Evaluating model...')
        evaluate_model(df, user_id_col='user_id', item_id_col='business_id')

    print('Training model...')
    # build recommendations for known users and known businesses
    # with collaborative filtering method
    ds_full = Dataset()
    # we call fit to supply userid, item id and user/item features
    ds_full.fit(
            df[user_id_col].unique(),   # all the users
            df[item_id_col].unique(),   # all the items
    )
    (interactions, weights) = ds_full.build_interactions([(x[0], x[1], x[2])
                                                         for x in df.values]
                                                         )
    # model
    model_full = LightFM(no_components=100, learning_rate=0.05,
                         loss='warp', max_sampled=50)
    model_full.fit(interactions, sample_weight=weights,
                   epochs=10, num_threads=10)
    # mapping
    user_id_map, _, business_id_map, _ = ds_full.mapping()

    # data preparation
    df_interactions = pd.DataFrame(weights.todense())
    df_interactions.index = list(user_id_map.keys())
    df_interactions.columns = list(business_id_map.keys())
    user_dict = user_id_map
    item_dict = df.set_index(item_id_col)[item_name_col].to_dict()
    return model_full, df_interactions, user_dict, item_dict


def evaluate_model(df, user_id_col='user_id',
                   item_id_col='business_id', stratify=None):
    """ Model evaluation.

    Args:
        df: the input dataframe.
        user_id_col: user id column.
        item_id_col: item id column.
        stratify: if use stratification.

    Returns:
        train_auc: training set auc score.
        test_auc: testing set auc score.

    """
    # model evaluation
    # create test and train datasets
    print('model evaluation')
    train, test = train_test_split(df, test_size=0.2, stratify=stratify)
    ds = Dataset()

    # we call fit to supply userid, item id and user/item features
    ds.fit(
            df[user_id_col].unique(),   # all the users
            df[item_id_col].unique(),   # all the items
    )

    # plugging in the interactions
    (train_interactions, train_weights) = ds.build_interactions(
        [(x[0], x[1], x[2]) for x in train.values]
        )
    (test_interactions, _) = ds.build_interactions(
        [(x[0], x[1], x[2]) for x in test.values]
        )
    # model
    model = LightFM(no_components=100, learning_rate=0.05,
                    loss='warp', max_sampled=50)
    model.fit(train_interactions, sample_weight=train_weights,
              epochs=10, num_threads=10)

    # auc-roc
    train_auc = auc_score(model, train_interactions, num_threads=20).mean()
    print('Training set AUC: %s' % train_auc)
    test_auc = auc_score(model, test_interactions, num_threads=20).mean()
    print('Testing set AUC: %s' % test_auc)
