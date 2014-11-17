import matplotlib.pyplot as plt

class Cluster:
    def __init__(self, businesses):
        self.businesses = businesses

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
        self.cluster_id = cluster_id

    def __str__(self):
        return "("+str(self.business_id)+","+str(self.review_count)+", opened on: "+str(self.open_date)+", closed on:"+str(self.closing_date)+", price: "+str(self.price_range)+", reviews: " + str(self.review_count) + ")"

    def plot_moving_avg(self):
        num_days = len(self.moving_avg_ratings)
        plt.plot(range(num_days), self.moving_avg_ratings)
        plt.axis([self.open_date, num_days, 0 , 5]);
        plt.show()
