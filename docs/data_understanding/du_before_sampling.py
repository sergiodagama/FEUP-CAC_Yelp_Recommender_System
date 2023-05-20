import json
from progress.spinner import Spinner
import time

import matplotlib.pyplot as plt

##################### DATA LOADING

# get the start time
startTime = time.time()

def readJSONObjectsToList(filePath):
    objList = []
    spinner = Spinner('Loading ')
    with open(filePath, 'r', encoding="utf8") as f:
        for jsonObj in f:
            rowDict = json.loads(jsonObj)
            objList.append(rowDict)
            spinner.next()
    return objList

# import dataset and check dataset size
""" businesses = readJSONObjectsToList('./../../dataset/yelp_academic_dataset_business.json')
print('Number of businesses: ', len(businesses))

checkins = readJSONObjectsToList('./../../dataset/yelp_academic_dataset_checkin.json')
print('Number of checkins: ', len(checkins))

reviews = readJSONObjectsToList('./../../dataset/yelp_academic_dataset_review.json')
print('Number of reviews: ', len(reviews))

tips = readJSONObjectsToList('./../../dataset/yelp_academic_dataset_tip.json')
print('Number of tips: ', len(tips)) """

users = readJSONObjectsToList('./../../dataset/yelp_academic_dataset_user.json')
print('Number of users: ', len(users))

# get the loading time
elapsedTime = time.time() - startTime
print('Loading time:', elapsedTime/60, 'mins')

##################### DATA UNDERSTANDING

# check missing data
# TODO: open each of the json objs check if all properties exit and if they are not undefined, or empty

# User
reviewCounts = []
for user in users:
    reviewCounts.append(user['review_count'])

plt.title("Users' count of reviews histogram")
plt.hist(reviewCounts, bins=20)
plt.show()

averageStars = []
for user in users:
    averageStars.append(user['average_stars'])

plt.title("Users' average stars histogram")
plt.hist(averageStars)
plt.show()

fans = []
for user in users:
    fans.append(user['fans'])

plt.title("Users' fans histogram")
plt.hist(fans)
plt.show()

useful = []
for user in users:
    useful.append(user['useful'])

plt.title("Users' useful votes sent histogram")
plt.hist(useful)

funny = []
for user in users:
    funny.append(user['funny'])

plt.title("Users' funny votes sent histogram")
plt.hist(funny)
plt.show()

cool = []
for user in users:
    cool.append(user['cool'])

plt.title("Users' cool votes sent histogram")
plt.hist(useful)
plt.show()

# TODO: add all compliments by user and plot total compliments histogram

# TODO: plot the period of the yelping since

# Reviews