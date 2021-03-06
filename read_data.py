import json
import string
import numpy as np
from numpy import *
from datetime import datetime
from sets import Set
import matplotlib.pyplot as plt

from classes import *

# Global params
base_date = datetime.strptime("2000-01-01",'%Y-%m-%d')
closing_threshold = 16
interval = 30 # number of days in moving average 

# Sigma^2 used for the gaussian filter
sigma_sq = math.pow(interval / 1.0, 2)
mid_interval = interval / 2.0
gaussian_filter = map(lambda x: 1.0 / math.sqrt(2 * sigma_sq * math.pi) * math.exp(-math.pow(x - mid_interval, 2) / (2 * sigma_sq)), range(interval))
gaussian_filter_norm = gaussian_filter / sum(gaussian_filter)

def load_reviews(path, businesses):
    reviews = dict()

    print "Reading review data"
    last_review_in_dataset = None
    with open(path + '/review.json') as json_file:
        for line in json_file:
            review = json.loads(line)
            business_id = review['business_id']

            if business_id in reviews:
                reviews[business_id].append(review)
            else:
                reviews[business_id] = [review]

            date = (datetime.strptime(review['date'],'%Y-%m-%d') - base_date).days
            if (not last_review_in_dataset) or (date > last_review_in_dataset):
                last_review_in_dataset = date

    print "Computing moving averages"
    moving_avgs = dict()
    for business in businesses:
        if business_id in reviews:
            business_id = business.business_id
            reviews_for_business = reviews[business_id]
            reviews_of_days = [[] for _ in range(last_review_in_dataset + 1)]
            moving_avg = []
            first_review = None
            last_review = None
            for review in reviews_for_business:
                reviews_of_days[date].append(review['stars'])
                date = (datetime.strptime(review['date'],'%Y-%m-%d') - base_date).days
                if (not first_review) or (date < first_review):
                    first_review = date
                if (not last_review) or (date > last_review):
                    last_review = date

            reviews_of_days_avg = map(lambda x: float(sum(x)) / len(x) if x != [] else 0, reviews_of_days)

            # This code interpolates the ratings
            #
            # for i in range(first_review, last_review):
            #    if reviews_of_days_avg[i] == 0.0:
            #        reviews_of_days_avg[i] = reviews_of_days_avg[i - 1]
            # moving_avg = np.convolve(reviews_of_days_avg, gaussian_filter_norm)

            # Old moving average code:
            #
            for d in range(len(reviews_of_days)):
               past_days = reviews_of_days[max(0, d - interval):d]
               ratings_in_interval = reduce(lambda x, y: x + y, past_days, [])
               if len(ratings_in_interval) > 0:
                   moving_avg.append(average(ratings_in_interval))
               else:
                   moving_avg.append(0)

            # Update the business
            business.open_date = first_review
            business.last_review = last_review
            if (last_review < last_review_in_dataset - closing_threshold):
                business.closing_date = last_review
            business.moving_avg_ratings = moving_avg
            business.reviews_of_days = reviews_of_days

    print "Last review in dataset: " + str(last_review_in_dataset)

def load_businesses(path):
    businesses = []

    print "Reading business data"
    with open(path + '/business.json') as json_file:
        for line in json_file:
            business = json.loads(line)
            business_id = business['business_id']
            business_categories = business['categories']
            business_attributes = business['attributes']
            business_address = business['full_address']

            zip_code_str = string.split(business_address, " ")[-1]
            zip_code = -1
            if zip_code_str.isdigit():
                zip_code = int(zip_code_str)

            business_price_range = -1
            if 'Price Range' in business_attributes:
                business_price_range = int(business_attributes['Price Range'])

            #if "Restaurants" in business_categories:
            #    businesses.append(Business(business_id, business['review_count'], business_categories, business['longitude'], business['latitude'], business_price_range, zip_code))
            businesses.append(Business(business_id, business['review_count'], business_categories, business['longitude'], business['latitude'], business_price_range, zip_code))

    print "Number of businesses: " + str(len(businesses))
    return businesses

def generate_cat_features(businesses):
    print "Generating category features"

    all_categories = Set()
    for business in businesses:
        for category in business.categories:
            all_categories.add(category)

    list_of_categories = sorted(list(all_categories))
    for business in businesses:
        cat_features = []
        for category in list_of_categories:
            if category in business.categories:
                cat_features.append(1)
            else:
                cat_features.append(0)
        business.cat_features = cat_features

    print "Categories: " + str(list_of_categories)
    print "Number of features: " + str(len(list_of_categories))

def construct_feature_diff_matrix(businesses, rich_features):
    print "Constructing feature diff matrix"

    diff_matrix = []
    for i in range(len(businesses)):
        for j in range(i):
            diff_matrix.append(businesses[i].diff_features(businesses[j], rich_features))

    return array(diff_matrix)

def print_list_of_businesses(businesses):
    for business in businesses:
        print business

if __name__ == "__main__":
    businesses = load_businesses(".")
    sorted_businesses = sorted(businesses, key = lambda b: b.review_count)
    generate_cat_features(sorted_businesses)
    load_reviews(".", sorted_businesses[-10:])
