import matplotlib.pyplot as plt
import numpy as np
from numpy import *
from pylab import *

# Classes Import
from classes import *
from read_data import *
from clustering import *

def correlation_bus(x,y):
    start = max(x.open_date+60,y.open_date+60)
    end = min(x.last_review-60,y.last_review-60)

    if(end <= start):
        return 0

    x_se = x.moving_avg_ratings[start:end]
    y_se = y.moving_avg_ratings[start:end]
    x_avg = np.average(x_se)
    y_avg = np.average(y_se)

    a = x_se - x_avg
    b = y_se - y_avg

    length = end - start
    E_XY = np.dot(a, b) / length
    E_XX = np.dot(a, a) / length
    E_YY = np.dot(b, b) / length
    return E_XY / math.sqrt(E_XX * E_YY)

def correlation_mat(businesses):
	n=len(businesses)
	corr=np.zeros((n,n))
	max_i = 0
	max_j = 0
        max_corr = None
	for i in range(n):
		for j in range(i + 1):
			temp_corr=correlation_bus(businesses[i],businesses[j])
			corr[i][j]=temp_corr
			corr[j][i]=temp_corr
			if (max_corr == None or max_corr < temp_corr) and i != j:
				max_i, max_j = i, j
				max_corr = temp_corr

	plt.figure(2)
	plt.pcolor(corr)
	plt.colorbar()
	plt.show()

	print max_i, max_j
	businesses[max_i].plot_moving_avg()
	businesses[max_j].plot_moving_avg()
	businesses[10].plot_moving_avg()
	return corr
	# print corr

def cluster_num(businesses):
	businesses.sort(key=operator.attrgetter('review_count'));
	print businesses[-1].cluster_id
	print businesses[-1].review_count
	return businesses[-1].cluster_id

def pair_cor():
	review_count_thres=500

	businesses_list=load_businesses("./dataset")
	businesses_list.sort(key=operator.attrgetter('business_id'));
	clusters=cluster_business(businesses_list)
	#for c in clusters:
	#	if(len(c.businesses)>3000):
	#		clus=c
	clus = Cluster(businesses_list)
	cluster_businesses = filter(lambda b: b.review_count > review_count_thres, clus.businesses)
	load_reviews("./dataset",cluster_businesses)
	corr=correlation_mat(cluster_businesses)
	to_file(corr,len(cluster_businesses))
	generate_cat_features(cluster_businesses)
	features = construct_feature_diff_matrix(cluster_businesses)
	print features.shape
	np.savetxt("feature_mat.txt",features)

def to_file(corr,n):
	corr_1d=np.zeros((n)*(n-1)/2)
	print n
	ind=0
	for i in range(n):
		for j in range(i):
			corr_1d[ind]=corr[i][j]
			ind+=1
	print corr
	print corr_1d
	np.save("corr_1d",corr_1d)
	np.savetxt("corr_1d.txt",corr_1d)



if __name__ == "__main__":
	pair_cor();

