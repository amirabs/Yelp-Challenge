import json
import matplotlib.pyplot as plt
import operator
import numpy as np
from numpy import *
from datetime import datetime

# Imports for sklearn
import time
from sklearn import cluster, datasets
from sklearn.metrics import euclidean_distances
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler



PATH="/Users/amir/Desktop/yelp/dataset/"


class Business:
	def __init__(self, business_id, longitude=0, latitude=0, review_count=0, open_date=0, closing_date=0, ):
		self.review_count=review_count
		self.business_id = business_id
		self.longitude = longitude
		self.latitude = latitude
		self.open_date = open_date
		self.closing_date = closing_date
	def __str__(self):
		return str(self.business_id)+ ": ("+str(self.longitude)+","+str(self.latitude)+")"

businesses = dict();


def cluster_business(businesses):
	np.random.seed(0)

	# Generate datasets. We choose the size big enough to see the scalability
	# of the algorithms, but not too big to avoid too long running times
	n_samples = 1500
	noisy_circles = datasets.make_circles(n_samples=n_samples, factor=.5,
										  noise=.05)
	noisy_moons = datasets.make_moons(n_samples=n_samples, noise=.05)
	blobs = datasets.make_blobs(n_samples=n_samples, random_state=8)
	no_structure = np.random.rand(n_samples, 2), None

	colors = np.array([x for x in 'bgrcmykbgrcmykbgrcmykbgrcmyk'])
	colors = np.hstack([colors] * 20)

	plt.figure(figsize=(17, 9.5))
	plt.subplots_adjust(left=.001, right=.999, bottom=.001, top=.96, wspace=.05,
						hspace=.01)

	plot_num = 1
	X = np.ndarray(shape=(0,2))
	count =0;
	for b in businesses:
		X=vstack([X,[b.longitude,b.latitude]])
		if(count>1000):
			break
		count+=1
	print type(X)
	print X

	dbscan = cluster.DBSCAN(eps=.2)

	for name, algorithm in [
							('DBSCAN', dbscan)
						   ]:
		# predict cluster memberships
		t0 = time.time()
		algorithm.fit(X)
		t1 = time.time()
		if hasattr(algorithm, 'labels_'):
			y_pred = algorithm.labels_.astype(np.int)
		else:
			y_pred = algorithm.predict(X)

		# plot
		# plt.subplot(4, 7, plot_num)
		# if i_dataset == 0:
		# 	plt.title(name, size=18)
		plt.scatter(X[:, 0], X[:, 1], color=colors[y_pred].tolist(), s=10)

		if hasattr(algorithm, 'cluster_centers_'):
			centers = algorithm.cluster_centers_
			center_colors = colors[:len(centers)]
			plt.scatter(centers[:, 0], centers[:, 1], s=100, c=center_colors)
		# plt.xlim(-2, 2)
		# plt.ylim(-2, 2)
		plt.xticks(())
		plt.yticks(())
		plt.text(.99, .01, ('%.2fs' % (t1 - t0)).lstrip('0'),
				 transform=plt.gca().transAxes, size=15,
				 horizontalalignment='right')
		plot_num += 1

	plt.show()

def init_business():
	id_count=0
	with open(PATH+'business.json') as json_file:
		for line in json_file:
			business = json.loads(line)
			business_id=business["business_id"]
			longitude=business["longitude"]
			latitude=business["latitude"]
			businesses[business_id] = Business(business_id, longitude, latitude)
		businesses_list = businesses.values()
		businesses_list.sort(key=operator.attrgetter('business_id'));
		# for b in businesses_list:
		# 	print b
		return businesses_list

def main():
	businesses_list = init_business()
	cluster_business(businesses_list)

if __name__ == "__main__":
	main()
