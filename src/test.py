import os

from recommender_system import RecommenderSystem


current_directory = os.getcwd()
print("Current working directory:", current_directory)

# creating an instance of the RecommenderSystem
rs = RecommenderSystem('svd')
rs.build_recommender_system()

# TEST 1: get top 5 recommendations for 5 different users and check their validity
number_of_recommendations = 5

# user 1 (id: qVc8ODYU5SZjKXVBgXdI7w, name: Walker)
user1 = 'qVc8ODYU5SZjKXVBgXdI7w'

# user 2  (id: j14WgRoU_-2ZE1aw1dXrJg, name: Daniel)
user2 = 'j14WgRoU_-2ZE1aw1dXrJg'

# user 3 (id: , name: )
# user 4 (id: , name: )
# user 5 (id: , name: )

users_to_test = [user1]

# using SVD
print("---------- Using SVD ----------")
rs.make_recommendations_to_multiple_users(users_to_test, number_of_recommendations)

# using kNN
print("---------- Using KNN ----------")
rs.set_model_type('knn')
rs.build_recommender_system()
rs.make_recommendations_to_multiple_users(users_to_test, number_of_recommendations)
