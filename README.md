# Yelpify
[![Build Status](https://travis-ci.com/RH5648/yelpify.svg?branch=main)](https://travis-ci.com/RH5648/yelpify)

Group Members: June Yang, Quoc Dung (Daniel) Cao, Runqiu Hu, Zichen Liu

## Introductrion
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

## Usage of recommendation system
If this user or item is in the database, you can use recommendation_system_cf.   
For Yelp user, just replace the current USER_ID with your USER_ID and run recommendation_system_cf. You will get top 10 recommendation restruants/businesses.   
For business own, replace the current ITEM_ID with your ITEM_ID and run recommendation_system_cf. You will get 10 USER_IDs represent 10 users that may appreciate your business the most.   
If you not sure whether you are in the database or not, you can use recommendation_system_hybird.   
For Yelp user, just replace the current USER_ID with your BUSINESS_ID and run recommendation_system_hybrid. You will get top 10 recommendation restruants/businesses.   
For business own, replace the current ITEM_ID with your BUSINESS_ID and run recommendation_system_hybrid. You will get 10 USER_IDs represent 10 users that may appreciate your business the most.   
