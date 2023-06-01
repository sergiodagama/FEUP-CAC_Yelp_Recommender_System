from collections import Counter
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt


class RecommenderSystem:
    """
    RecommenderSystem class for building and making recommendations using different algorithms.

    Parameters:
        - model_type (str): The type of model to use for recommendations.
                            Supported options: 'svd', 'knn'
    """

    def __init__(self, model_type):
        """
        Initialize the RecommenderSystem object.

        Parameters:
            - model_type (str): The type of model to use for recommendations.
                                Supported options: 'svd', 'knn'
        """
        self.model_type = model_type
        self.load_data()
        self.preprocess_data()

    def set_model_type(self, model_type):
        """
        Set the model type for the RecommenderSystem object.

        Parameters:
            - model_type (str): The type of model to use for recommendations.
                                Supported options: 'svd', 'knn'
        """
        self.model_type = model_type

    def read_file(self, filename):
        """
        Read and load a pickled file.

        Parameters:
            filename (str): The name of the file to be read.

        Returns:
            The loaded object from the pickled file.
        """
        with open(filename, 'rb') as f:
            return pd.read_pickle(f)

    def save_file(self, filename, var):
        """
        Save an object to a pickled file.

        Parameters:
            filename (str): The name of the file to be saved.
            var: The object to be pickled and saved.

        Returns:
            None
        """
        with open(filename, 'wb') as f:
            pd.to_pickle(var, f)

    def load_data(self):
        """
        Load the user, business, and review data from pickle files.
        """
        # load user data
        self.users = self.read_file('../data/sample_users.pickle')

        # load business data
        self.businesses = self.read_file('../data/sample_business.pickle')

        # load review data
        self.reviews = self.read_file('../data/sample_reviews.pickle')

    def preprocess_data(self):
        """
            Preprocess the data by pivoting the review data to create a user-item matrix.
        """
        review_data = pd.DataFrame(self.reviews)

        self.user_item_ratings = review_data.pivot_table(index='user_id', columns='business_id', values='stars', aggfunc='mean', fill_value=0)

        # split the data into training and testing sets
        self.train_data, self.test_data = train_test_split(self.user_item_ratings, test_size=0.000000001)

    def build_recommender_system(self):
        """
        Build the recommender system by preprocessing the data, splitting into training and testing sets,
        and training the chosen model based on the specified algorithm.
        """
        if self.model_type == 'svd':
            self.model = TruncatedSVD(n_components=10)
            self.model.fit(self.train_data)
        elif self.model_type == 'knn':
            self.model = NearestNeighbors(n_neighbors=5, algorithm='brute', metric='cosine')
            self.model.fit(self.train_data.values)
        else:
            raise ValueError('Invalid model type.')

    def make_recommendations(self, user_id, number_of_recommendations):
        """
        Make recommendations for a given user based on the trained model.

        Parameters:
            - user_id (str): The ID of the user for whom recommendations are to be made.

        Returns:
            - recommended_items (list): A list of the top 5 recommended items for the given user.
        """

        # preprocess data
        user_ratings = self.train_data.loc[user_id].values.reshape(1, -1)

        # create a DataFrame with the reshaped data and feature names
        reshaped_data = pd.DataFrame(user_ratings, columns=self.train_data.columns)

        # make recommendations for a user
        if self.model_type == 'svd':
            predicted_ratings = self.model.transform(reshaped_data)
            top_n_indices = predicted_ratings.argsort()[0, ::-1][:number_of_recommendations]
            recommended_items = self.train_data.columns[top_n_indices]
        elif self.model_type == 'knn':
            _, indices = self.model.kneighbors(reshaped_data)
            recommended_items = self.train_data.columns[indices.flatten()][:number_of_recommendations]
        else:
            raise ValueError('Invalid model type.')

        return recommended_items

    def get_business_info(self, business_id):
        """
        Get the information for a given business.

        Parameters:
            - business_id (str): The ID of the business for which information is to be retrieved.

        Returns:
            - business_info (dict): A dictionary containing the information for the given business.
        """
        business_info = self.businesses[self.businesses['business_id'] == business_id].to_dict('records')[0]
        return business_info

    def get_user_info(self, user_id):
        """
        Get the information for a given user.

        Parameters:
            - user_id (str): The ID of the user for which information is to be retrieved.

        Returns:
            - user_info (dict): A dictionary containing the information for the given user.
        """
        user_info = self.users[self.users['user_id'] == user_id].to_dict('records')[0]
        return user_info

    def make_recommendations_to_multiple_users(self, users_list, number_of_recommendations, verbose=True):
        """
        Make recommendations for a list of users.

        Parameters:
            - users_list (list): A list of user IDs for whom recommendations are to be made.
            - number_of_recommendations (int): The number of recommendations to make for each user.

        Returns:
            - None
        """
        for user in users_list:
            user_name = self.get_user_info(user)['name']
            if verbose:
                print("Top {} recommendations for user {}:".format(number_of_recommendations, user_name))
                print()

            recommendations = self.make_recommendations(user, number_of_recommendations)
            for recommendation in recommendations:
                business_info = self.get_business_info(recommendation)
                if verbose:
                    print(business_info['name'])
                    print(business_info['categories'])
                    print()
            if verbose:
                print()

    def count_n_different_categories(self):
        """
        Count the number of different categories in the dataset.

        Parameters:
            - None

        Returns:
            - len(categories) (int): The number of different categories in the dataset.
        """
        categories = set()

        for _, business in self.businesses.iterrows():
            if business['categories'] is not None:
                categories.update(business['categories'])

        return len(categories)

    def get_avg_similarity_between_models(self, number_of_users, number_of_recommendations):
        user_ids = self.users.sample(number_of_users)['user_id']
        knn_results = []
        svd_results = []

        print(user_ids)

        for i in user_ids:
            print("User ID:", i)

            self.set_model_type('svd')
            self.build_recommender_system()
            svd_results.append(self.make_recommendations(i, number_of_recommendations))

            self.set_model_type('knn')
            self.build_recommender_system()
            knn_results.append(self.make_recommendations(i, number_of_recommendations))


        similarity = []

        for i in range(len(knn_results)):
            similarity.append(len(set(knn_results[i]) & set(svd_results[i])) / float(len(set(knn_results[i]) | set(svd_results[i]))))

        print("Average similarity:", sum(similarity) / float(len(similarity)))

    def get_businesses_categories(self, business_id):
        """
        Get the categories of a given business.

        Parameters:
            - business_id (str): The ID of the business for which categories are to be retrieved.

        Returns:
            - categories (list): A list of categories for the given business.
        """
        business_info = self.get_business_info(business_id)
        return business_info['categories']

    def get_top_x_relevant_categories(self, user_id, x):
        """
        Get the top x relevant categories for a given user.
        
        Parameters:
            - user_id (str): The ID of the user for which categories are to be retrieved.
            - x (int): The number of categories to retrieve.  
        
        Returns:
            - top_x_categories (list): A list of the top x relevant categories for the given user.
        """
        all_pos_reviews_categories = [] # positive means with 4 or 5 stars

        for _, review in self.reviews.iterrows():
            if review['user_id'] == user_id and review['stars'] >= 4:
                business_id = review['business_id']
                business_info = self.get_business_info(business_id)
                if business_info['categories'] is not None:
                    all_pos_reviews_categories.append(business_info['categories'])

        # make a histogram with the all_pos_reviews_categories counts
        # Count the frequency of each string
        histogram = {}
        for item in all_pos_reviews_categories:
            if item in histogram:
                histogram[item] += 1
            else:
                histogram[item] = 1

        # Extract the unique strings and their frequencies
        labels = list(histogram.keys())
        frequencies = list(histogram.values())

        # Create the histogram
        """ plt.bar(labels, frequencies)
        plt.xticks(rotation=45)

        # Add labels and title
        plt.xlabel('Categories the user rated 4 or 5 stars')
        plt.ylabel('Frequency')
        plt.title('All Categories of businesses reviewed by user with 4 or 5 Histogram')

        # Display the histogram
        plt.show() """

        """sorted_histogram = sorted(histogram.items(), key=lambda x: x[1], reverse=True)

        labels = [item[0] for item in sorted_histogram[:10]]
        frequencies = [item[1] for item in sorted_histogram[:10]]

        plt.bar(labels, frequencies)
        plt.xticks(rotation=20)

        plt.xlabel('Categories the user rated 4 or 5 stars')
        plt.ylabel('Frequency')
        plt.title('Top ten Categories of businesses reviewed by user with 4 or 5 Histogram')

        # Display the histogram
        plt.show() """

        # get top x categories in all_pos_reviews_categories
        return [y[0] for y in sorted(Counter(all_pos_reviews_categories).items(), key=lambda y: y[1], reverse=True)[:x]]

    def check_if_a_certain_business_is_relevant_to_user(self, user_id, business_categories, top_categories):
        # check if the business_categories are in the top 10 categories
        for category in top_categories:
            for business_category in business_categories:
                if business_category in category:
                    return True
        return False
