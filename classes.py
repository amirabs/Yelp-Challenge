import matplotlib.pyplot as plt
import math
from numpy import *
from common import *

class Cluster:
    def __init__(self, businesses):
        self.businesses = businesses
    def __str__(self):
        name="["
        for b in self.businesses:
            name+=str(b)+",";
        return name+"]"

class Business:
    def __init__(self, business_id, review_count,
            categories = [], longitude = 0,
            latitude = 0, price_range = -1, zip_code = -1, cluster_id=-1):
        self.review_count = review_count
        self.categories = categories
        self.business_id = business_id
        self.longitude = longitude
        self.latitude = latitude
        self.open_date = -1
        self.last_review = -1
        self.closing_date = -1
        self.price_range = price_range
        self.cat_features = None
        self.zip_code = zip_code
        self.moving_avg_ratings = []
        self.reviews_of_days = []
        self.cluster_id = cluster_id

    def __str__(self):
        return "("+str(self.business_id)+","+str(self.review_count)+", opened on: "+str(self.open_date)+", closed on:"+str(self.closing_date)+", price: "+str(self.price_range)+", reviews: " + str(self.review_count) + ")"

    def plot_moving_avg(self):
        ratings_xy = nested_list_to_xy_mat(self.reviews_of_days)
        ratings_x = ratings_xy[:, 0]
        ratings_y = ratings_xy[:, 1]

        plt.figure(figsize=(30, 9))
        ax = plt.subplot(111)
        ax.spines["top"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.get_xaxis().tick_bottom()
        ax.get_yaxis().tick_left()
        plt.xlabel("Day", fontsize=30)
        plt.ylabel("Rating", fontsize=30)
        plt.plot(ratings_x, ratings_y, 'o', color=tableau20[0], markeredgecolor=None, markersize=8.0)
        plt.plot(self.moving_avg_ratings, lw=5, color=tableau20[3])
        plt.xticks(fontsize=24)
        plt.yticks(fontsize=24)
        plt.xlim(self.open_date + 60, len(self.moving_avg_ratings) - 60)
        plt.ylim(0,5.5)
        plt.savefig("moving_avg.pdf")
        plt.show()

    def diff_features(self, other):
        diff_features = []
        geo_dist = math.sqrt(math.pow(self.longitude - other.longitude, 2) + math.pow(self.latitude - other.latitude, 2))
        diff_features.append(geo_dist)
        diff_features.append(math.fabs(self.price_range - other.price_range))
        diff_features.append(math.fabs(self.open_date - other.open_date))
        #return diff_features + map(lambda xs: -1 if xs[0] == xs[1] else 1, zip(self.cat_features, other.cat_features))
        return diff_features + self.cat_features + other.cat_features
        # return diff_features + map(lambda xs: xs[0] + xs[1], zip(self.cat_features, other.cat_features))
