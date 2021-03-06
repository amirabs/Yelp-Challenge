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

# Classes Import
from classes import *
from read_data import *

PATH="dataset/"

businesses= dict()
clusters = []

def cluster_business(businesses):
	NClusters=50
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

	plt.figure(1)
	plt.subplots_adjust(left=.001, right=.999, bottom=.001, top=.96, wspace=.05,
						hspace=.01)

	plot_num = 1
	X = np.ndarray(shape=(0,2))
	count =0;
	for b in businesses:
		X=vstack([X,[b.longitude,b.latitude]])
		# if(count>1000):
		# 	break
		count+=1
	# print type(X)
	# print X

	k_means = cluster.MiniBatchKMeans(n_clusters=NClusters)
	dbscan = cluster.DBSCAN(eps=.03)
	affinity_propagation = cluster.AffinityPropagation(damping=.9, preference=-200)
	spectral = cluster.SpectralClustering(n_clusters=2,eigen_solver='arpack',affinity="nearest_neighbors")


	for name, algorithm in [
							('MiniBatchKMeans', k_means),
							# ('DBSCAN', dbscan),
							# ('SpectralClustering', spectral)
							# ('AffinityPropagation', affinity_propagation),
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
		ax=plt.subplot(1, 2, plot_num)
		plt.title(name, size=16)
		plt.scatter(X[:, 0], X[:, 1], color=colors[y_pred].tolist(), s=10)

		if hasattr(algorithm, 'cluster_centers_'):
			centers = algorithm.cluster_centers_
			center_colors = colors[:len(centers)]
			plt.scatter(centers[:, 0], centers[:, 1], s=100, c=center_colors)
		ax.spines["top"].set_visible(False)
		ax.spines["bottom"].set_visible(False)
		ax.spines["right"].set_visible(False)
		ax.spines["left"].set_visible(False)
		plt.xticks(())
		plt.yticks(())
		plot_num += 1
	plt.show()


	clusters=[]
	for index in range(NClusters):
		clusters.append(Cluster([]))
	for index in range(len(businesses)):
		businesses[index].cluster_id=y_pred[index]
		clusters[y_pred[index]].businesses.append(businesses[index])
	return clusters

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
	# businesses_list = init_business()
	businesses_list=load_businesses("./dataset")
	businesses_list.sort(key=operator.attrgetter('business_id'));
	clusters=cluster_business(businesses_list)
	

if __name__ == "__main__":
	c=main()
	print c

