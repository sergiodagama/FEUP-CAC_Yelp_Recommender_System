import os

from recommender_system import RecommenderSystem


current_directory = os.getcwd()
print("Current working directory:", current_directory)

# creating an instance of the RecommenderSystem
rs = RecommenderSystem('svd')
rs.build_recommender_system()

""" # TEST 1: get top 3 recommendations for 2 different users and check their validity
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

rs.make_recommendations_to_multiple_users([user1], number_of_recommendations) """

user1 = 'bJ5FtCtZX3ZZacz2_2PJjA'

print("Number of categories: ", rs.count_n_different_categories())

top_ten_relevants = rs.get_top_x_relevant_categories(user1, 10)

print("Top relevants for user Bill: ", top_ten_relevants)

# check if businesses are relevant to a given user (from SVD results)
print(rs.check_if_a_certain_business_is_relevant_to_user(user1, ["Day Spas", "Beauty & Spas"], top_ten_relevants))
print(rs.check_if_a_certain_business_is_relevant_to_user(user1, ["American (Traditional)", "Restaurants", "Pizza", "Sandwiches", "Meat Shops", "Food", "Wraps", "Delis", "Specialty Food", "Salad"], top_ten_relevants))
print(rs.check_if_a_certain_business_is_relevant_to_user(user1, ["Venues & Event Spaces", "Event Planning & Services", "Hotels", "Hotels & Travel"], top_ten_relevants))
print(rs.check_if_a_certain_business_is_relevant_to_user(user1, ["Cocktail Bars", "Food Delivery Services", "Nightlife", "Breakfast & Brunch", "Food", "Bars", "Event Planning & Services", "Caterers", "Restaurants", "Indian"], top_ten_relevants))
print(rs.check_if_a_certain_business_is_relevant_to_user(user1, ["Steakhouses", "Seafood", "American (Traditional)", "Restaurants"], top_ten_relevants))

# check if businesses are relevant to a given user (from KNN results)
print(rs.check_if_a_certain_business_is_relevant_to_user(user1, ["Restaurants", "Pizza"], top_ten_relevants))
print(rs.check_if_a_certain_business_is_relevant_to_user(user1, ["Shopping", "Cosmetics & Beauty Supply", "Convenience Stores", "Beauty & Spas", "Food", "Drugstores"], top_ten_relevants))
print(rs.check_if_a_certain_business_is_relevant_to_user(user1, ["Restaurants", "Party & Event Planning", "Event Planning & Services", "Sandwiches", "Cafes", "Spanish", "Breakfast & Brunch"], top_ten_relevants))
print(rs.check_if_a_certain_business_is_relevant_to_user(user1, ["Wedding Planning", "Transportation", "Party Bus Rentals", "Wineries", "Party & Event Planning", "Local Flavor", "Food", "Wine Tours", "Arts & Entertainment", "Tours", "Limos", "Event Planning & Services", "Hotels & Travel"], top_ten_relevants))
print(rs.check_if_a_certain_business_is_relevant_to_user(user1, ["Specialty Food", "Salad", "American (New)", "Sandwiches", "Food", "Desserts", "Breakfast & Brunch", "Restaurants", "Cheese Shops"], top_ten_relevants))

# check if businesses are relevant to a given user (from SVD results in GMC community)
print(rs.check_if_a_certain_business_is_relevant_to_user(user1, ["American (Traditional)", "Restaurants", "Pizza", "Sandwiches", "Meat Shops", "Food", "Wraps", "Delis", "Specialty Food", "Salad"], top_ten_relevants))
print(rs.check_if_a_certain_business_is_relevant_to_user(user1, ["Shopping", "Flowers & Gifts", "Gift Shops", "Women's Clothing", "Fashion", "Jewelry"], top_ten_relevants))
print(rs.check_if_a_certain_business_is_relevant_to_user(user1, ["Massage", "Active Life", "Health & Medical", "Fitness & Instruction", "Chiropractors", "Beauty & Spas", "Dance Studios", "Acupuncture", "Yoga"], top_ten_relevants))
print(rs.check_if_a_certain_business_is_relevant_to_user(user1, ["Arts & Entertainment", "Active Life", "Skating Rinks", "Local Flavor", "Parks", "Music Venues", "Nightlife"], top_ten_relevants))
print(rs.check_if_a_certain_business_is_relevant_to_user(user1, ["Restaurants", "Sandwiches"], top_ten_relevants))

# Testing the recommender system with svd and knn with a sample of 100 users
# rs.get_avg_similarity_between_models(100, 20)

# create a sample of users to test 100, svd
rs = RecommenderSystem('svd')
rs.build_recommender_system()

users_to_test = rs.users.sample(100)['user_id']
number_of_recommendations = 10

precision_per_user = []

for user in users_to_test:
    recommendations = rs.make_recommendations(user, number_of_recommendations)
    top_ten_relevants = rs.get_top_x_relevant_categories(user, 5)
    print("top ten relevants: ", top_ten_relevants)

    results = []
    for recommendation in recommendations:
        print("recommendation: ", recommendation)
        business_categories = rs.get_businesses_categories(recommendation)
        is_relevant = rs.check_if_a_certain_business_is_relevant_to_user(user, business_categories, top_ten_relevants)
        results.append(is_relevant)

    precision = results.count(True) / len(results)
    print("precision: ", precision)
    precision_per_user.append(precision)

print("Average precision for svd: ", sum(precision_per_user) / len(precision_per_user))

# create a sample of users to test 100, knn
rs = RecommenderSystem('knn')
rs.build_recommender_system()

users_to_test = rs.users.sample(100)['user_id']
number_of_recommendations = 10

precision_per_user = []

for user in users_to_test:
    recommendations = rs.make_recommendations(user, number_of_recommendations)
    top_ten_relevants = rs.get_top_x_relevant_categories(user, 5)
    print("top ten relevants: ", top_ten_relevants)

    results = []
    for recommendation in recommendations:
        print("recommendation: ", recommendation)
        business_categories = rs.get_businesses_categories(recommendation)
        is_relevant = rs.check_if_a_certain_business_is_relevant_to_user(user, business_categories, top_ten_relevants)
        results.append(is_relevant)

    precision = results.count(True) / len(results)
    print("precision: ", precision)
    precision_per_user.append(precision)

print("Average precision for knn: ", sum(precision_per_user) / len(precision_per_user))
