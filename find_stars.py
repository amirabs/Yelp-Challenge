import json
import matplotlib.pyplot as plt
import operator
import numpy as np
from numpy import *
from datetime import datetime


PATH="/Users/amir/Desktop/yelp/dataset/"


class Business:
	def __init__(self, id_string,review_count, open_date=0, closing_date=0, longitude=0, latitude=0):
		self.review_count=review_count
		self.id_string = id_string
		self.longitude = longitude
		self.latitude = latitude
		self.open_date = open_date
		self.closing_date = closing_date
	def __str__(self):
		return  "("+str(self.id_string)+","+str(self.review_count)+")"

businesses = dict();

def count_review_per_business():
	id_count=0
	with open(PATH+'business.json') as json_file:
		for line in json_file:
			business = json.loads(line)
			business_id=business["business_id"]
			review_count=business["review_count"]
			businesses[business_id] = Business(business_id, review_count)
		businesses_list = businesses.values()
		businesses_list.sort(key=operator.attrgetter('review_count'));
		for b in businesses_list:
			print b

def day_move_ave(businessid):
	ave = []
	recent_stars=np.array([[]])
	move_ave=[]
	interval = 60 # number of days in moving average 
	ys = []
	with open(PATH+'review.json') as json_file:
		i=1
		sum=0.0
		start_date = datetime.strptime("2000-01-01",'%Y-%m-%d')
		end_date = datetime.strptime("2014-11-11",'%Y-%m-%d')
		num_days = (end_date - start_date).days
		reviews_of_days = [[] for _ in range(num_days)]
		first_date = None

		for line in json_file:
			review = json.loads(line)
			if(review['business_id']==businessid):
				star=review['stars']
				date=str(review['date'])
			

				date = datetime.strptime(date,'%Y-%m-%d')
				if first_date==None:
					first_date = date 
				diff_date_day=(date - start_date).days
				reviews_of_days[diff_date_day].append(star)
		for d in range(len(reviews_of_days)):
			# recent_stars=insert(recent_stars,[0],reviews_of_days[d])
			move_ave.append(average(reduce(lambda x,y: x+y, reviews_of_days[max(0,d-interval):d], [])))

		plt.plot(range(len(reviews_of_days)), move_ave)
		plt.axis([(first_date - start_date).days, num_days, 0 , 5]);
		# plt.show()	

def move_ave(businessid):
	ave = []
	stars =[]
	move_ave =[]
	recent_stars=np.array([])
	interval = 50
	ys = []
	with open(PATH+'review.json') as json_file:
		i=1
		sum=0.0
		for line in json_file:
			review = json.loads(line)
			if(review['business_id']==businessid):
				star=review['stars']
				sum+=star
				recent_stars=insert(recent_staers,[0],star)
				move_ave.append(average(recent_stars[0:interval]))         	
				ave.append(sum/i)
				stars.append(star)
				ys.append(i)
				i=i+1
		plt.plot(ys, move_ave)
		plt.axis([0, i, 0 , 5]);
		# plt.show()

def main():
	day_move_ave("zt1TpTuJ6y9n551sw9TaEg")

if __name__ == "__main__":
	count_review_per_business()
	main()
