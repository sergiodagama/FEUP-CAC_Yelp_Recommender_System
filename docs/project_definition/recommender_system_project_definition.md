# Recommender System

## Context

The Yelp dataset is a collection of data that contains information on local businesses, reviews, and user information from the Yelp.com website. The dataset has over 7 million reviews and it contains information about businesses, users, check-ins, tips and photos, as well.

## Relevant data to use

Almost all the tables have some interesting attributes, but mainly these ones:
- Businesses
- Reviews
- Users
- Check-in

## Tools

- Scikit-learn (Python): for most of the methods comparison (most of the research done with it then)
- TensorFlow (Python): to build the hybrid model, with deep learning, in an attempt to showcase and compare the state-of-the-art techniques
- Django (Python): for the showcase application backend
- React (TypeScript): for the showcase application frontend

## Methodology

- Data understanding, and analysis of the dataset in the recommender system context
- Preprocess data of the Yelp dataset, for example removing unwanted attributes, different methods might require different preprocessing
- Split data into training and testing, guaranteeing that on all the models the split is equal (controlled variables)
- Train a model using each one of the methods, using the train data
- Test the model, using the test data
- Evaluate the model/method performance using a set of predefined metrics for all the recommender system version
- Finally, create an application with a dedicated frontend and backend, in order to showcase the recommender system, allowing to choose between the versions

## Goals

- The goal of the recommender system is to recommend the most relevant businesses to each user. In order to find out the best way to do that, in this project, we are developing different versions of recommender systems (user-based collaborative filtering, item-based collaborative filtering, content-based filtering, matrix factorization and a hybrid method). These multiple versions will allow the comparison of the performance of the different methods, so that we find out the best method to build a recommender system of businesses, in our problem context.