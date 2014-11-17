class Cluster:
    def __init__(self, businesses):
        self.businesses = businesses

class Business:
    def __init__(self, business_id, review_count, moving_avg_ratings = [],
            categories = [], open_date = -1, closing_date = -1, longitude = 0,
            latitude = 0, price_range = -1, zip_code = -1,cluster_id=-1):
        self.review_count = review_count
        self.categories = categories
        self.business_id = business_id
        self.longitude = longitude
        self.latitude = latitude
        self.open_date = open_date
        self.closing_date = closing_date
        self.price_range = price_range
        self.cat_features = None
        self.zip_code = zip_code
        self.moving_avg_ratings = moving_avg_ratings
        self.cluster_id=cluster_id

    def __str__(self):
        return "("+str(self.business_id)+","+str(self.review_count)+", opened on: "+str(self.open_date)+", closed on:"+str(self.closing_date)+",price"+str(self.price_range)+")"
