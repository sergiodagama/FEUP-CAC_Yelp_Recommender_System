import os

from recommender_system import RecommenderSystem


current_directory = os.getcwd()
print("Current working directory:", current_directory)

# creating an instance of the RecommenderSystem
rs = RecommenderSystem('svd')
rs.build_recommender_system()

# TEST 1: get top 5 recommendations for 5 different users and check their validity
number_of_recommendations = 5

user1 = 'bJ5FtCtZX3ZZacz2_2PJjA'
user2 = 'nnImk681KaRqUVHlSfZjGQ'
user3 = '84HvpQDxcHWmbMDfs8IEYw'

users_to_test = [user1, user2, user3]

# using SVD
print("---------- Using SVD ----------")
rs.make_recommendations_to_multiple_users(users_to_test, number_of_recommendations)

# using KNN
""" print("---------- Using KNN ----------")
rs.set_model_type('knn')
rs.build_recommender_system()
rs.make_recommendations_to_multiple_users(users_to_test, number_of_recommendations) """
