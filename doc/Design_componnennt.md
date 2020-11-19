# Software Design
Runqiu Hu, Zichen Liu, Quoc Dung Cao, June Yang


### Background
Recommender systems are beneficial to both service providers and users. People have too many options to use in this age of information explosion. With overload options, people are not sure what they really need. Recommendation system reduces the decision-making time of users. It can not only accurately predict the user’s behavior, but also expand the user’s field of vision, helping users find things that they may be interested in but not so easy to find. The same recommendation system also benefits merchants, through more precise targeting of target users and gaining exposure. Using the yelp data set, we try to provide accurate meal recommendations for users on yelp, as well as businesses with user groups who appreciate their services more.

# Functional Specification
## Use cases
### For users:
Given a known user from Yelp dataset, the recommendation tool could return some most recommended dining places based on collaborative filtering. 
Given a new user, the recommendation tool could return some most recommended dining places based on content-based filtering, using existing user features.

### For businesses:
Given a known business from Yelp dataset, the recommendation tool could return some users that are most likely to visit this business based on collaborative filtering.
Given a new business, the recommendation tool could return some users that are most likely to visit this business based on content-based filtering, using existing business features. 

## Data Sources
Yelp academic dataset: https://www.yelp.com/dataset
The recommendation tool uses three datasets. Here we a description:
The first dataset, review.json contains information about business ID, review ID, user ID and the actual review text. The dataset contains 8,021,122 records. To properly prepare it for modeling, we take a sample dataset that firstly, take the first 1,000,000 rows of the dataset. Secondly, we filter out businesses that received more than five reviews, and users that have given more than five reviews to get dense interactions between users and businesses.
The second dataset business.json contains features about the business, for example, the location, name and if it’s open or not. 
The third dataset user.json contains information about the users such as review count, and average stars given. 
We join these three datasets together by business ID and user ID. Quality of the dataset is high: we have removed (very few) null values that were contained by the data. In the end, we are working with a sample dataset of 208,977 rows. 

# Component Specification
## For users:
### sub-component 1: 
Name: is_new_user  
What is does: determine if a user is new or existing  
Input: user_id(str)  
Output: true/false (bool)  

### sub-component 2:
Name: predict_new_users  
What is does: find most recommended businesses for new users  
Input: user_id(str), k(integer)  
Output: list of business_id (list of str)  

### sub-component 3:
Name: predict_existing_users  
What is does: find most recommended businesses for existing users  
Input: user_id(str), k(integer)  
Output: list of business_id (list of str)  

### Interactions
			IF is_new_user(user_id):
				return predict_new_users(user_id, k)
			ELSE:
				return predict_existing_users(user_id, k)


## For business:
### sub-component 1: 
Name: is_new_business  
What is does: determine if a business is new or existing  
Input: business_id(str)  
Output: true/false (bool)  

### sub-component 2:
Name: predict_new_businesses  
What is does: find most recommended users for new businesses  
Input: business_id(str), k(integer)  
Output: list of user_id (list of str)  

### sub-component 3:
Name: predict_existing_businesses  
What is does: find most recommended users for existing businesses  
Input: business_id(str), k(integer)  
Output: list of user_id (list of str)  

### Interactions
			IF is_new_business(business_id):
				return predict_new_businesses(business_id, k)
			ELSE:
				return predict_existing_businesses(business_id, k)
