import json
from datetime import datetime
from sets import Set

class Business:
    def __init__(self, id_string,review_count, open_date=0, closing_date=0, longitude=0, latitude=0):
        self.review_count=review_count
        self.id_string = id_string
        self.longitude = longitude
        self.latitude = latitude
        self.open_date = open_date
        self.closing_date = closing_date

    def __str__(self):
        return "("+str(self.id_string)+","+str(self.review_count)+", opened on: "+str(self.open_date)+", closed on:"+str(self.closing_date)+")"

def read_date():
    closing_threshold = 365
    base_date = datetime.strptime("2000-01-01",'%Y-%m-%d')
    businesses = []
    reviews = dict()
    all_categories = Set()

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

    print last_review_in_dataset
    print "Reading business data"
    with open('yelp_academic_dataset_business.json') as json_file:
        for line in json_file:
            business = json.loads(line)
            business_id = business['business_id']
            business_categories = business['categories']

            for category in business_categories:
                all_categories.add(category)

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

            businesses.append(Business(business_id, business['review_count'], open_date, closing_date, business['longitude'], business['latitude']))

    print all_categories
    return businesses

def print_list_of_businesses(businesses):
    for business in businesses:
        print business

if __name__ == "__main__":
    read_date()
