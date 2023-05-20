import os

from recommender_system import RecommenderSystem

current_directory = os.getcwd()
print("Current working directory:", current_directory)

rs = RecommenderSystem('svd')

rs.build_recommender_system()
rs.make_recommendations('qVc8ODYU5SZjKXVBgXdI7w')

