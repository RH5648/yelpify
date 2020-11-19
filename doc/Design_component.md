# Software Design
Runqiu Hu, Zichen Liu, Quoc Dung Cao, June Yang

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
