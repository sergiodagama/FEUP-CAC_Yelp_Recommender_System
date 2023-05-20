import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import NearestNeighbors


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


    def read_file(self, filename):
        """
        Read and load a pickled file.

        Parameters:
            filename (str): The name of the file to be read.

        Returns:
            The loaded object from the pickled file.
        """
        with open(filename, 'rb') as f:
            return pickle.load(f)


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
            pickle.dump(var, f)


    def load_data(self):
        """
        Load the user, business, and review data from pickle files.
        """
        # load user data
        self.users = self.read_file('sample_users.pickle')

        # load business data
        self.businesses = self.read_file('sample_businesses.pickle')

        # load review data
        self.businesses = self.read_file('sample_reviews.pickle')


    def build_recommender_system(self):
        """
        Build the recommender system by preprocessing the data, splitting into training and testing sets,
        and training the chosen model based on the specified algorithm.
        """
        # preprocess data
        review_data = pd.DataFrame(self.reviews)
        user_item_ratings = review_data.pivot(index='user_id', columns='business_id', values='rating').fillna(0)

        # split the data into training and testing sets
        self.train_data, self.test_data = train_test_split(user_item_ratings, test_size=0.2)

        # train the model based on the chosen algorithm
        if self.model_type == 'svd':
            self.model = TruncatedSVD(n_components=10)
            self.model.fit(self.train_data)
        elif self.model_type == 'knn':
            self.model = NearestNeighbors(n_neighbors=5, algorithm='brute', metric='cosine')
            self.model.fit(self.train_data.values)
        else:
            raise ValueError('Invalid model type.')
        

    def make_recommendations(self, user_id):
        """
        Make recommendations for a given user based on the trained model.

        Parameters:
            - user_id (str): The ID of the user for whom recommendations are to be made.

        Returns:
            - recommended_items (list): A list of the top 5 recommended items for the given user.
        """
        # preprocess data
        user_ratings = self.train_data.loc[user_id].values.reshape(1, -1)

        # make recommendations for a user
        if self.model_type == 'svd':
            predicted_ratings = self.model.transform(user_ratings)
            top_n_indices = predicted_ratings.argsort()[0, ::-1][:5]
            recommended_items = self.train_data.columns[top_n_indices]
        elif self.model_type == 'knn':
            _, indices = self.model.kneighbors(user_ratings)
            recommended_items = self.train_data.columns[indices.flatten()]
        else:
            raise ValueError('Invalid model type.')

        # Print the recommended items
        for item_id in recommended_items:
            print(item_id)

        return recommended_items



import os

current_directory = os.getcwd()
print("Current working directory:", current_directory)

rs = RecommenderSystem('svd')

rs.build_recommender_system()
rs.make_recommendations('qVc8ODYU5SZjKXVBgXdI7w')
