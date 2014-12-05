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
	plt.figure(2)
	plt.pcolor(corr)
	plt.colorbar()
	# plt.show()
	businesses[max_i].plot_moving_avg(3)
	businesses[max_j].plot_moving_avg(4)
	return corr
	# print corr

def cluster_num(businesses):
	businesses.sort(key=operator.attrgetter('review_count'));
	print businesses[-1].cluster_id
	print businesses[-1].review_count
	return businesses[-1].cluster_id

def pair_cor():
	review_count_thres=2000

	businesses_list=load_businesses("./dataset")
	businesses_list.sort(key=operator.attrgetter('business_id'));
	clusters=cluster_business(businesses_list)
	for c in clusters:
		if(len(c.businesses)>3000):
			clus=c
	cluster_businesses = filter(lambda b: b.review_count > review_count_thres, clus.businesses)
	load_reviews("./dataset",cluster_businesses)
	var_vec= variance_vec(cluster_businesses)
	corr=correlation_mat(cluster_businesses,var_vec)
	to_file(corr,len(cluster_businesses))

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

