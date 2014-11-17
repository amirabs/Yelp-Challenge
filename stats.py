import matplotlib.pyplot as plt
import numpy as np
from numpy import *
from pylab import *

# Classes Import
from classes import *
from read_data import *
from clustering import *

def correlation_bus(x,y):
	start=max(x.open_date,y.open_date)+0.0
	end=min(x.last_review,y.last_review)+0.0
	if(end==start):
		return 0
	x_ave=np.average(x.moving_avg_ratings[x.open_date:x.last_review+1])
	y_ave=np.average(y.moving_avg_ratings[y.open_date:y.last_review+1])

	a=map(lambda el: 0 if el ==0 else el-x_ave, x.moving_avg_ratings)
	b=map(lambda el: 0 if el ==0 else el-y_ave, y.moving_avg_ratings)

	return np.dot(a,b)/(end-start)


def variance_vec(businesses):
	n=len(businesses)
	var_vec=np.zeros(n)
	for i in range(n):
		vari=correlation_bus(businesses[i],businesses[i])
		var_vec[i]=vari
	# plt.figure(3)	
	# plt.plot(var_vec)
	# plt.show()
	# plt.figure(4)	
	# plt.plot(map(lambda x: x.review_count, businesses))
	# plt.show()
	print var_vec
	return var_vec

def correlation_mat(businesses,var_vec):
	n=len(businesses)
	corr=np.zeros((n,n))
	maxim=-100000
	max_i=0
	max_j=0
	for i in range(n):
		for j in range(i+1):
			temp_corr=correlation_bus(businesses[i],businesses[j])
			normalizer = sqrt(var_vec[i]*var_vec[j])
			corr[i][j]=temp_corr/normalizer
			corr[j][i]=temp_corr/normalizer
			if(maxim<corr[i][j] and i != j):
				max_j=j
				max_i=i
				maxim=corr[i][j]
	plot_corr=np.array(corr)
	plt.figure(2)
	plt.pcolor(plot_corr)
	plt.colorbar()
	plt.show()
	businesses[max_i].plot_moving_avg(3)
	businesses[max_j].plot_moving_avg(4)
	print corr

def cluster_num(businesses):
	businesses.sort(key=operator.attrgetter('review_count'));
	print businesses[-1].cluster_id
	print businesses[-1].review_count
	return businesses[-1].cluster_id

def main():
	review_count_thres=1000

	businesses_list=load_businesses("./dataset")
	businesses_list.sort(key=operator.attrgetter('business_id'));
	clusters=cluster_business(businesses_list)
	cluster_businesses = filter(lambda b: b.review_count > review_count_thres, clusters[28].businesses)
	load_reviews("./dataset",cluster_businesses)
	var_vec= variance_vec(cluster_businesses)
	correlation_mat(cluster_businesses,var_vec)

if __name__ == "__main__":
	main()

