import matplotlib.pyplot as plt
import math
from numpy import *

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
        num_days = len(self.moving_avg_ratings)
        plt.figure()
        plt.plot(range(num_days), self.moving_avg_ratings)
        plt.axis([0, num_days, 0 , 5]);
        plt.show()

    def diff_features(self, other):
        diff_features = []
        geo_dist = math.pow(self.longitude - other.longitude, 2) + math.pow(self.latitude - other.latitude, 2)
        diff_features.append(geo_dist)
        return diff_features + map(lambda xs: -1 if xs[0] == xs[1] else 1, zip(self.cat_features, other.cat_features))
