import matplotlib.pyplot as plt
import numpy as np
from numpy import *
from pylab import *

# Classes Import
from classes import *
from read_data import *
from clustering import *
from common import *

def delta_trend(x, y, window_size):
    # Make sure that x is the older business (swap businesses if necessary)
    if y.open_date < x.open_date:
        x, y = y, x

    start = max(0, y.open_date - window_size)
    end = min(len(x.reviews_of_days), y.open_date + window_size)

    ratings_before = x.reviews_of_days[y.open_date - window_size:y.open_date]
    ratings_after = x.reviews_of_days[y.open_date:y.open_date + window_size]

    ratings_before_xy = nested_list_to_xy_mat(ratings_before)
    ratings_after_xy = nested_list_to_xy_mat(ratings_after)

    delta = 0
    if len(ratings_before_xy) >= 10 and len(ratings_after_xy) >= 10:
        before_x = -window_size + ratings_before_xy[:, 0]
        before_y = ratings_before_xy[:, 1]
        after_x = ratings_after_xy[:, 0]
        after_y = ratings_after_xy[:, 1]

        y = np.concatenate([before_y, after_y])
        A = np.zeros([len(y), 3])
        A[:, 2] = 1
        A[0:len(before_x), 0] = before_x
        A[len(before_x):, 1] = after_x

        theta = np.linalg.lstsq(A, y)[0]

        xs = np.concatenate([before_x, after_x])

        # plt.figure(figsize=(30, 9))
        # ax = subplot(111)
        # ax.spines["top"].set_visible(False)
        # ax.spines["bottom"].set_visible(False)
        # ax.spines["right"].set_visible(False)
        # ax.spines["left"].set_visible(False)
        # ax.get_xaxis().tick_bottom()
        # ax.get_yaxis().tick_left()
        # plt.xlabel("Day", fontsize=30)
        # plt.ylabel("Rating", fontsize=30)
        # plt.plot(xs, y, 'o', color=tableau20[0], markeredgecolor=None, markersize=8.0)
        # plt.plot(xs, np.sum(A * theta, 1), lw=5, color=tableau20[3])
        # xticks(fontsize=24)
        # yticks(fontsize=24)
        # xlim(-window_size - 1, window_size + 1)
        # ylim(0,5.5)
        # plt.savefig("trend.pdf")
        # plt.show()

        deg_before = math.degrees(math.atan(theta[0]))
        deg_after = math.degrees(math.atan(theta[1]))
        delta = deg_after - deg_before

    return delta

def delta_trend_vec(businesses, window_size):
    print "Generating delta trend vector"

    result = []
    for i in range(len(businesses)):
        for j in range(i):
            result.append(delta_trend(businesses[i], businesses[j], window_size))
    return result

def delta_mean(x, y, window_size):
    # Make sure that x is the older business (swap businesses if necessary)
    if y.open_date < x.open_date:
        x, y = y, x

    start = max(0, y.open_date - window_size)
    end = min(len(x.reviews_of_days), y.open_date + window_size)
    ratings_before = x.reviews_of_days[y.open_date - window_size:y.open_date]
    ratings_after = x.reviews_of_days[y.open_date:y.open_date + window_size]
    ratings_before_xy = nested_list_to_xy_mat(ratings_before)
    ratings_after_xy = nested_list_to_xy_mat(ratings_after)

    delta = 0
    if len(ratings_before_xy) >= 10 and len(ratings_after_xy) >= 10:
        before_x = -window_size + ratings_before_xy[:, 0]
        before_y = ratings_before_xy[:, 1]
        after_x = ratings_after_xy[:, 0]
        after_y = ratings_after_xy[:, 1]

        y = np.concatenate([before_y, after_y])
        xs = np.concatenate([before_x, after_x])

        mean_after = np.mean(ratings_after_xy[:, 1])
        mean_before = np.mean(ratings_before_xy[:, 1])

        delta = mean_after - mean_before

        # plt.figure(figsize=(30, 9))
        # ax = subplot(111)
        # ax.spines["top"].set_visible(False)
        # ax.spines["bottom"].set_visible(False)
        # ax.spines["right"].set_visible(False)
        # ax.spines["left"].set_visible(False)
        # ax.get_xaxis().tick_bottom()
        # ax.get_yaxis().tick_left()
        # plt.xlabel("Day", fontsize=30)
        # plt.ylabel("Rating", fontsize=30)
        # plt.plot(xs, y, 'o', color=tableau20[0], markeredgecolor=None, markersize=8.0)
        # plt.plot([-window_size, 0], [mean_before, mean_before], lw=5, color=tableau20[3])
        # plt.plot([0, window_size], [mean_after, mean_after], lw=5, color=tableau20[3])
        # xticks(fontsize=24)
        # yticks(fontsize=24)
        # xlim(-window_size - 1, window_size + 1)
        # ylim(0,5.5)
        # plt.savefig("mean.pdf")
        # plt.show()

    return delta

def delta_mean_vec(businesses, window_size):
    print "Generating delta mean vector"

    result = []
    for i in range(len(businesses)):
        for j in range(i):
            result.append(delta_mean(businesses[i], businesses[j], window_size))
    return result

def gen_trend(x, y, window_size):
    # Make sure that x is the older business (swap businesses if necessary)
    if y.open_date < x.open_date:
        x, y = y, x

    start_date = max(0, y.open_date - window_size)
    end_date = min(len(x.reviews_of_days), y.open_date + window_size)

    ratings = x.reviews_of_days[start_date:end_date]

    ratings_xy = nested_list_to_xy_mat(ratings)

    trend = 0
    if len(ratings_xy) >= 20:
        ratings_x = -window_size + ratings_xy[:, 0]
        ratings_y = ratings_xy[:, 1]

        A = np.zeros([len(ratings_y), 2])
        A[:, 1] = 1
        A[:, 0] = ratings_x

        theta = np.linalg.lstsq(A, ratings_y)[0]

        # plt.figure(figsize=(30, 9))
        # ax = subplot(111)
        # ax.spines["top"].set_visible(False)
        # ax.spines["bottom"].set_visible(False)
        # ax.spines["right"].set_visible(False)
        # ax.spines["left"].set_visible(False)
        # ax.get_xaxis().tick_bottom()
        # ax.get_yaxis().tick_left()
        # plt.xlabel("Day", fontsize=30)
        # plt.ylabel("Rating", fontsize=30)
        # plt.plot(ratings_x, ratings_y, 'o', color=tableau20[0], markeredgecolor=None, markersize=8.0)
        # plt.plot(ratings_x, np.sum(A * theta, 1), lw=5, color=tableau20[3])
        # xticks(fontsize=24)
        # yticks(fontsize=24)
        # xlim(-window_size - 1, window_size + 1)
        # ylim(0,5.5)
        # plt.savefig("gen_trend.pdf")
        # plt.show()

        trend = math.degrees(math.atan(theta[0]))

    return trend

def gen_trend_vec(businesses, window_size):
    print "Generating general trend vector"

    result = []
    for i in range(len(businesses)):
        for j in range(i):
            result.append(gen_trend(businesses[i], businesses[j], window_size))
    return result

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
# def correlation_bus(x,y):
#     start = max(x.open_date + 15, y.open_date + 15)
#     end = min(len(x.moving_avg_ratings), start + 120)
# 
#     if(end <= start):
#         return 0
# 
#     x_se = x.moving_avg_ratings[start:end]
#     y_se = y.moving_avg_ratings[start:end]
#     xs = range(start, end)
# 
#     # plt.figure(figsize=(30, 9))
#     # ax = subplot(111)
#     # ax.spines["top"].set_visible(False)
#     # ax.spines["bottom"].set_visible(False)
#     # ax.spines["right"].set_visible(False)
#     # ax.spines["left"].set_visible(False)
#     # ax.get_xaxis().tick_bottom()
#     # ax.get_yaxis().tick_left()
#     # plt.xlabel("Day", fontsize=30)
#     # plt.ylabel("Rating", fontsize=30)
#     # plt.plot(xs, x_se, lw=5, color=tableau20[3])
#     # plt.plot(xs, y_se, lw=5, color=tableau20[0])
#     # xticks(fontsize=24)
#     # yticks(fontsize=24)
#     # xlim(start, end + 1)
#     # ylim(0,5.5)
#     # plt.savefig("corr.pdf")
#     # plt.show()
# 
#     x_avg = np.average(x_se)
#     y_avg = np.average(y_se)
# 
#     a = x_se - x_avg
#     b = y_se - y_avg
# 
#     length = end - start
#     E_XY = np.dot(a, b) / length
#     s_x = math.sqrt(np.dot(a, a) / length)
#     s_y = math.sqrt(np.dot(b, b) / length)
#     corr = E_XY / (s_x * s_y) if s_x * s_y > 0 else 0
#     return corr

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
    for c in clusters:
        if(len(c.businesses)>3000):
            clus=c
    # clus = Cluster(businesses_list)
    cluster_businesses = filter(lambda b: b.review_count > review_count_thres, clus.businesses)
    print len(cluster_businesses)
    load_reviews("./dataset",cluster_businesses)

    corr=correlation_mat(cluster_businesses)
    to_file(corr,len(cluster_businesses))

    # Plot moving average for business with most ratings
    # sorted(cluster_businesses, key=(lambda b: b.review_count))[-1].plot_moving_avg()

    for window_size in [60, 90]:
        trend_vec = gen_trend_vec(cluster_businesses, window_size)
        np.savetxt(str(window_size) + "_gen_trend_1d.txt", trend_vec)

        delta_vec = delta_trend_vec(cluster_businesses, window_size)
        np.savetxt(str(window_size) + "_delta_1d.txt", delta_vec)

        delta_mean = delta_mean_vec(cluster_businesses, window_size)
        np.savetxt(str(window_size) + "_mean_1d.txt", delta_mean)

        generate_cat_features(cluster_businesses)

        features = construct_feature_diff_matrix(cluster_businesses, False)
        np.savetxt(str(window_size) + "_feature_mat.txt", features)
        print features.shape

        rich_features = construct_feature_diff_matrix(cluster_businesses, True)
        np.savetxt(str(window_size) + "_rich_feature_mat.txt", rich_features)
        print rich_features.shape

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
    pair_cor()
