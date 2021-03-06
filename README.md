# Yelpify
[![Build Status](https://travis-ci.com/RH5648/yelpify.svg?branch=main)](https://travis-ci.com/RH5648/yelpify)

Group Members: June Yang, Quoc Dung (Daniel) Cao, Runqiu Hu, Zichen Liu

## Introduction
We are building a meal/user recommedation system based on Yelp academic databases.  
The databases we used are review, user, and business.  
Review database has the contents of review, who worte this review (UserID), and which business is this review for (business ID).  
User database covers the demographic information of the user.  
Business database includes rating of businerss and geographic information.  

## Datasets
The datasets we used is from yelp academic database:  
https://www.yelp.com/dataset   
We use three of them, they are review, user, and business.  
Review database has the contents of review, who wrote this review (user ID), which business review is for (business ID).  
Users database has the demographic information of the user.  
Business database has the rating of business and geographic information.  

## Use cases
We have two use cases:
For Yelp user, they use this system to get recommendations based on their previous reviews or demographics data.
For business owner, they use this system to find out what kind of user like their food and which feature attracts clients most. 

## Video overview of code
Recommendations for known user/item (collaborative filtering)

https://drive.google.com/file/d/1C29OzIHcP1R8YaI03jkBoA14c71E5Twn/view?usp=sharing

Recommendations for know or new user/item (hybrid filtering)

https://www.youtube.com/watch?v=83fH7h-o9zk&feature=youtu.be

## Install and usage of recommendation system
For installation, you just need to download our code. 
You are given two examples, recommendation_system_cf.py and recommendation_system_hybrid.py.
The recommendation_system_cf.py calls the collaborative filter, you can use recommendation_system_cf.py to recommend for known users and items. 
The recommendation_system_hybrid.py calls the hybrid filter, you can use recommendation_system_hybrid.py to recommend for known and new users / items.

Use 
```
python recommendation_system_cf.py 
```
or
```
python recommendation_system_hybrid.py 
```
to run collaborative model or hybrid model.
#### If this user or item is in the database, you can use both recommendation_system_cf and recommendation_system_hybrid.   
For Yelp user, just replace the current USER_ID with your USER_ID and run recommendation_system_cf. You will get top 10 recommended restaurants/businesses.   
For business own, replace the current ITEM_ID with your ITEM_ID and run recommendation_system_cf. You will get 10 USER_IDs representing 10 users that may favorite your business the most.   
#### If the user or item is not in the database, you can use recommendation_system_hybrid.   
You don't have to input USER_ID or ITEM_ID.
For user, you just need to input a user feature list at new_user_feature. Because we only consider one user feature, which is average star, you only need to enter a list containing one number, such as [x], the value range of x is [0,5]. You will get top 10 recommendation restaurants/businesses.   
For business own, the new_item_feature requires a list contains 89 numbers, these number can be 0 or 1. You will get 10 USER_IDs representing 10 users that may favorite your business the most.   

## Environment requirements:
```
brotlipy==0.7.0
certifi==2020.11.8
cffi==1.14.4
chardet==3.0.4
cryptography==3.2.1
idna==2.10
joblib==0.17.0
lightfm==1.15
numpy==1.19.4
pandas==1.1.4
pyarrow==0.15.1
pycparser==2.20
pyOpenSSL==20.0.0
PySocks==1.7.1
python-dateutil==2.8.1
pytz==2020.4
requests==2.25.0
scikit-learn==0.23.2
scipy==1.5.3
six==1.15.0
threadpoolctl==2.1.0
urllib3==1.25.11
vim==0.0.1
win-inet-pton==1.1.0
wincertstore==0.2
```
