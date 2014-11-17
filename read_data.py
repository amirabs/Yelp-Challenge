import json
import string
from datetime import datetime
from sets import Set

from classes import *

# Global params
base_date = datetime.strptime("2000-01-01",'%Y-%m-%d')
closing_threshold = 365

def load_reviews():
    reviews = dict()

    print "Reading review data"
    last_review_in_dataset = None
    with open('yelp_academic_dataset_review.json') as json_file:
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

    print "Last review in dataset: " + str(last_review_in_dataset)
    return reviews, last_review_in_dataset

def load_businesses(reviews, last_review_in_dataset):
    businesses = []

    print "Reading business data"
    with open('yelp_academic_dataset_business.json') as json_file:
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

            open_date = -1
            closing_date = -1
            if business_id in reviews:
                reviews_of_business = reviews[business_id]

                first_review = None
                last_review = None
                for review in reviews_of_business:
                    date = (datetime.strptime(review['date'],'%Y-%m-%d') - base_date).days
                    if (not first_review) or (date < first_review):
                        first_review = date
                    if (not last_review) or (date > last_review):
                        last_review = date

                open_date = first_review

                if (last_review < last_review_in_dataset - closing_threshold):
                    closing_date = last_review

            businesses.append(Business(business_id, business['review_count'], business_categories, open_date, closing_date, business['longitude'], business['latitude'], business_price_range, zip_code))

    return businesses

def generate_cat_features(businesses):
    print "Generating category features"

    all_categories = Set()
    for business in businesses:
        for category in business.categories:
            all_categories.add(category)

    list_of_categories = list(all_categories)
    for business in businesses:
        cat_features = []
        for category in list_of_categories:
            if category in business.categories:
                cat_features.append(1)
            else:
                cat_features.append(0)
        business.cat_features = cat_features

def print_list_of_businesses(businesses):
    for business in businesses:
        print business

if __name__ == "__main__":
    #reviews, last_review_in_dataset = load_reviews()
    reviews, last_review_in_dataset = [], 0
    businesses = load_businesses(reviews, last_review_in_dataset)
    generate_cat_features(businesses)
