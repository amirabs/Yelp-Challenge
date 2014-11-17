class Cluster:
    def __init__(self, businesses):
        self.businesses = businesses

class Business:
    def __init__(self, id_string, review_count, moving_avg_ratings = [],
            categories = [], open_date = -1, closing_date = -1, longitude = 0,
            latitude = 0, price_range = -1, zip_code = -1):
        self.review_count = review_count
        self.categories = categories
        self.id_string = id_string
        self.longitude = longitude
        self.latitude = latitude
        self.open_date = open_date
        self.closing_date = closing_date
        self.price_range = price_range
        self.cat_features = None
        self.zip_code = zip_code
        self.moving_avg_ratings = moving_avg_ratings

    def __str__(self):
        return "("+str(self.id_string)+","+str(self.review_count)+", opened on: "+str(self.open_date)+", closed on:"+str(self.closing_date)+",price"+str(self.price_range)+")"
