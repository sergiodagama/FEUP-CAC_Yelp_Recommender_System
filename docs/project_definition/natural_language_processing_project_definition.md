# Recommender System

## Context

The Yelp dataset is a collection of data that contains information on local businesses, reviews, and user information from the Yelp.com website. The dataset has over 7 million reviews and it contains information about businesses, users, check-ins, tips and photos, as well.

## Relevant data to use

The main table that has something of interest is the following:
- Reviews

## Tools

- Scikit-learn (Python): for most of the methods comparison (most of the research done with it then)
- Numpy (Python)
- Jupyter notebooks
- Gensim (Python): To use word embeddings approach
- NLTK (Python): To use Bag-of-words approach

## Methodology

- Data understanding, and analysis of the dataset in the recommender system context
- Preprocess data of the Yelp dataset, for example removing unwanted attributes, different methods might require different preprocessing
- Split data into training and testing, stratified so that it is guaranteed that on all the models the split is equal (controlled variables)
- Train a model using each one of the methods, using the train data
- Test the model, using the test data
- Evaluate the model/method performance using a set of predefined metrics for all the recommender system version

## Goals

- The goal of the Natural Language processing is to predict the nature of the review using one or a variety of features.
We will try to use a bag-of-words approach and a word embedding one where the word2Vec model will be trained directly on
 the dataset. Finally, we will use pandas and numpy to better illustrate the process and relevant results.