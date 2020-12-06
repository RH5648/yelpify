"""
NAME
    data_preparation

DESCRIPTION
    This module provides access to functions that prepare data and generate features as needed.

FUNCTIONS
    get_input(url)
        Return the dataframe as downloaded from the url.

    prepare_data()
        Return the joined and cleaned dataset.

    round_of_rating(number)
        Return the number as rounded to the closest half integer
"""

import pandas as pd
import numpy as np

def get_input(url):
    """ Utility function to get file from url and read into a dataframe.

    Args:
        url: the dropbox link from which to download the raw or cleaned data.

    Returns:
        The dataframe.

    """
    if '.csv' in url:
        return pd.read_csv(url)
    elif '.json' in url:
        return pd.read_json(url, lines=True)
    elif '.parquet' in url:
        return pd.read_parquet(url)
    else:
        raise NotImplementedError("File type not supported")

def prepare_data(raw=False, round_ratings=False):
    """ Download and read the dataset.

    Args:
        raw: whether to download raw data or to download cleaned data.
        round_ratings: whether to perform round of ratings.

    Returns:
        the cleaned dataframe.

    """
    print('Downloading input data...')
    if raw:
        # read data
        review = get_input('https://www.dropbox.com/s/mtln9b6udoydn2h/yelp_academic_dataset_review_sample.csv?dl=1')
        user = get_input('https://www.dropbox.com/s/pngrptljotqm4ds/yelp_academic_dataset_user.json?dl=1')
        business = get_input('https://www.dropbox.com/s/w0wy854u5swrhmc/yelp_academic_dataset_business.json?dl=1')

        # join datasets
        review_user = pd.merge(review, user, on = "user_id", how = "left", suffixes=("","_user"))
        review_user_business = pd.merge(review_user, business, on = "business_id", how = "left", suffixes=("","_business"))
        review_user_business = review_user_business[['user_id', 'business_id', 'stars', 'text',
                            'name', 'average_stars',
                            'name_business', 'stars_business', 'categories', 'state', 'city']]
    else:
        review_user_business = get_input('https://www.dropbox.com/s/0c9zandfdsn4ujj/data_clean.parquet?dl=1')
    if round_ratings:
        # bucketize numeric features to reduce dimensions
        review_user_business['average_stars'] = review_user_business['average_stars'].apply(lambda x: round_of_rating(x))
        review_user_business['stars_business'] = review_user_business['stars_business'].apply(lambda x: round_of_rating(x))
    return review_user_business

# feature engineering
def round_of_rating(number):
    """Round a number to the closest half integer.

    Args:
        number: input of a number

    Returns: the closest half integer to this number.

    Example:
    >>> round_of_rating(1.3)
    1.5
    >>> round_of_rating(2.6)
    2.5
    >>> round_of_rating(3.0)
    3.0
    >>> round_of_rating(4.1)
    4.0

    """
    return round(number * 2) / 2