import pandas as pd
import time

import matplotlib.pyplot as plt

# get the start time
startTime = time.time()

with open('../../data/sample_users.pickle', 'rb') as f:
    users = pd.read_pickle(f)

with open('../../data/sample_business.pickle', 'rb') as f:
    business = pd.read_pickle(f)

with open('../../data/sample_reviews.pickle', 'rb') as f:
    reviews = pd.read_pickle(f)

# get the loading time
elapsedTime = time.time() - startTime
print('Loading time:', elapsedTime/60, 'mins')


# User
# traverse the users and get the review counts
reviewCounts = []
for c in users['review_count']:
    reviewCounts.append(c)

plt.title("Users' count of reviews histogram")
# plt.hist(reviewCounts, bins=range(0, 60, 10))
plt.hist(reviewCounts, bins=100)
plt.show()

averageStars = []
for stars in users['average_stars']:
    averageStars.append(stars)

plt.title("Users' average stars histogram")
plt.hist(averageStars)
plt.show()
