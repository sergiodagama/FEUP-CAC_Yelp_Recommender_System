import os

from recommender_system import RecommenderSystem


current_directory = os.getcwd()
print("Current working directory:", current_directory)

# creating an instance of the RecommenderSystem
rs = RecommenderSystem('svd')
rs.build_recommender_system()

# TEST 1: get top 3 recommendations for 2 different users and check their validity
number_of_recommendations = 5

user1 = 'bJ5FtCtZX3ZZacz2_2PJjA'
user2 = 'nnImk681KaRqUVHlSfZjGQ'

users_to_test = [user1, user2]

# using SVD
print("---------- Using SVD ----------")
rs.make_recommendations_to_multiple_users(users_to_test, number_of_recommendations)

# using KNN
print("---------- Using KNN ----------")
rs.set_model_type('knn')
rs.build_recommender_system()
rs.make_recommendations_to_multiple_users(users_to_test, number_of_recommendations)

# TEST 2: check the average similarity between the results returned using SVD and KNN, for 100 random users
print("---------- Average similarity between SVD and KNN ----------")

# rs.get_avg_similarity_between_models(100, 20)

# TEST 3: check the recommendations inside the communities
print("---------- Recommendations inside the communities ----------")

rs.reviews = rs.read_file('../data/user1_community_reviews.pickle')
rs.preprocess_data()
rs.set_model_type('svd')
rs.build_recommender_system()

rs.make_recommendations_to_multiple_users([user1], number_of_recommendations)

