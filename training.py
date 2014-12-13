import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error, make_scorer
from sklearn.cross_validation import KFold, cross_val_score
from sklearn.linear_model import Lasso
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB
from sklearn import preprocessing
from sklearn import svm
from numpy import *
from pylab import *
import random

# Classes Import
from classes import *
from read_data import *
from clustering import *

def load_file(file_name):
	arr=np.loadtxt(file_name)
	return arr

def main3():
	corr_thresh=0.1;
	corr_1d=load_file("corr_1d.txt")
	ind=np.where(np.abs(corr_1d)>corr_thresh)
	corr_1d_filtered=corr_1d[ind]
	print corr_1d_filtered.shape

	feature_mat=load_file("feature_mat.txt")
	feature_mat_filtered=feature_mat[ind]

	print feature_mat_filtered.shape

	cv5 = KFold(len(feature_mat), n_folds=10)
	lasso = Lasso()
	mean=np.mean(np.abs(corr_1d))
	# print mean
	a=cross_val_score(lasso, feature_mat, corr_1d, cv=cv5, scoring=make_scorer(mean_squared_error))
	a=np.sqrt(a)/mean
	print a

def mysvm_3class():
	corr_thresh=0.3;
	corr_1d=load_file("corr_1d.txt")
	ind_bin_0=np.where(corr_1d > corr_thresh)
	ind_bin_1=np.where(corr_1d < -corr_thresh)
	print ind_bin_0[0].shape
	print ind_bin_1[0].shape
	corr_1d_bin = np.zeros(len(corr_1d))
	corr_1d_bin[ind_bin_0] = 1
	corr_1d_bin[ind_bin_1] = 2
	print corr_1d_bin
	print corr_1d

	feature_mat=load_file("feature_mat.txt")

	print feature_mat.shape

	clf = svm.SVC(kernel='linear', C=1)
	a=cross_val_score(clf, feature_mat, corr_1d_bin, cv=5)
	print a

def mysvm():
	corr_thresh=0.1;
	corr_1d=load_file("corr_1d.txt")
	#ind=np.where( (corr_1d>0.088) | (corr_1d<0.06) ) #gaussian
	ind=np.where( (abs(corr_1d)>0.047) | (abs(corr_1d)<0.009) ) #movave
	corr_1d_filtered=corr_1d[ind]
	#ind_bin=np.where(corr_1d_filtered > 0.08) #gaussian
	ind_bin=np.where(abs(corr_1d_filtered) > 0.026) #moveave
	corr_1d_bin = np.zeros(len(corr_1d_filtered))
	corr_1d_bin[ind_bin] = 1
	print ind[0].shape
	print "positive corr:"
	print ind_bin[0].shape
	#print corr_1d_bin
	#print corr_1d_filtered

	feature_mat=load_file("feature_mat.txt")
	feature_mat_filtered=feature_mat[ind]

	print feature_mat_filtered.shape
	#try remove the distances from the features
	feature_mat_filtered2 = feature_mat_filtered[:,0:];
	#standardize the feature 'distance' through operations in dum:
	dum = feature_mat_filtered2[:,0]
	dum = 2 * dum / np.amax(dum)
	#dum = dum-1
	feature_mat_filtered2[:,0] = dum
	#clf = svm.SVC(kernel='rbf', C=1)
	clf= LogisticRegression(penalty='l1', tol=0.01,C =1)
	a=cross_val_score(clf, feature_mat_filtered2, corr_1d_bin, cv=10)
	train_score = clf.fit(feature_mat_filtered, corr_1d_bin).score(feature_mat_filtered, corr_1d_bin)
	print a
	print "training score: " + str(train_score);
	print "average cross validation score: " + str(np.average(a))
	print(len(corr_1d))
	print(np.percentile(abs(corr_1d),50))
	dum1 = np.median(abs(corr_1d))



def svm_asymthresh():
	corr_thresh=0.026; # for the working data movave 0.71
	#corr_thresh=0.08; #for gaussian
	corr_1d=load_file("corr_1d.txt")
	ind=np.where(abs(corr_1d)>corr_thresh)
	corr_1d_bin = np.zeros(len(corr_1d))
	corr_1d_bin[ind] = 1

	print "high corr:"
	print len(ind[0])
	print corr_1d_bin


	feature_mat=load_file("feature_mat.txt")

	
	feature_mat_filtered = feature_mat[:,0:];
	print feature_mat_filtered.shape
	#try remove the distances from the features
	#standardize the feature 'distance' through operations in dum:
	dum = feature_mat_filtered[:,0]
	dum = 2 * dum / np.amax(dum)
	dum = dum-1
	feature_mat_filtered[:,0] = dum
	clf = svm.SVC(kernel='linear', C=1)
	#clf= LogisticRegression(penalty='l1', tol=0.01)
	#clf.fit(feature_mat_filtered, corr_1d_bin)
	a=cross_val_score(clf, feature_mat_filtered, corr_1d_bin, cv=10)
	#train_score = clf.fit(feature_mat_filtered, corr_1d_bin).score(feature_mat_filtered, corr_1d_bin)
	print a
	#print "training score: " + str(train_score);
	print "average cross validation score: " + str(np.average(a))


def naiveb():
	corr_thresh=0.1;
	corr_1d=load_file("corr_1d.txt")
	ind=np.where(np.abs(corr_1d)>corr_thresh)
	corr_1d_filtered=corr_1d[ind]
	ind_bin=np.where(corr_1d_filtered > 0)
	corr_1d_bin = np.zeros(len(corr_1d_filtered))
	corr_1d_bin[ind_bin] = 1
	print corr_1d_bin
	print corr_1d_filtered

	feature_mat=load_file("feature_mat.txt")
	feature_mat_filtered=feature_mat[ind]

	print feature_mat_filtered.shape
	print "naive bayes"
	clf= BernoulliNB()
	#feature_mat_filtered2 = feature_mat_filtered[:,1:];
	a=cross_val_score(clf, feature_mat_filtered, corr_1d_bin, cv=10)
	print a

def logisreg():
	corr_thresh=0.1;
	corr_1d=load_file("corr_1d.txt")
	ind=np.where(np.abs(corr_1d)>corr_thresh)
	corr_1d_filtered=corr_1d[ind]
	ind_bin=np.where(corr_1d_filtered > 0)
	corr_1d_bin = np.zeros(len(corr_1d_filtered))
	corr_1d_bin[ind_bin] = 1
	print corr_1d_bin
	print corr_1d_filtered

	feature_mat=load_file("feature_mat.txt")
	feature_mat_filtered=feature_mat[ind]

	print feature_mat_filtered.shape

	clf= LogisticRegression(penalty='l2', tol=0.01)
	a=cross_val_score(clf, feature_mat_filtered, corr_1d_bin, cv=10)
	print a

def linear_reg():
	corr_1d=load_file("corr_1d.txt")
	# print np.arange(-1,1,0.1)
	# plt.hist(corr_1d, bins=np.arange(-1,1,0.1))
	# plt.show()
	feature_mat=load_file("feature_mat.txt")
	cv5 = KFold(len(feature_mat), n_folds=10)
	lasso = Lasso()
	mean=np.mean(np.abs(corr_1d))
	# print mean
	a=cross_val_score(lasso, feature_mat, corr_1d, cv=cv5, scoring=make_scorer(mean_squared_error))
	a=np.sqrt(a)/mean
	# print a

def plot_dist_vs_corr():
    corr_1d = load_file("corr_1d.txt")
    feature_mat = load_file("feature_mat.txt")
    plt.figure()
    plt.plot(feature_mat[:, 0], corr_1d, 'ro')
    plt.axis([0, 0.2, min(corr_1d), max(corr_1d)])
    plt.show()

def plot_dist_vs_delta():
    delta_1d = load_file("delta_1d.txt")
    feature_mat = load_file("feature_mat.txt")
    plt.figure()
    plt.plot(feature_mat[:, 0], delta_1d, 'ro')
    plt.axis([-1, 1, min(delta_1d), max(delta_1d)])
    plt.show()

def filter_data(features, delta):
    inds = np.where(delta != 0.0)
    return features[inds], delta[inds]

def sampled_data(features, delta):
    buckets = np.arange(0, 1, 0.01)
    vals_in_buckets = [[] for x in range(len(buckets))]

    for i in range(len(delta)):
        bucket = 0
        v = features[i, 0]
        while (bucket < len(buckets) - 1) and (v > buckets[bucket]):
            bucket += 1
        vals_in_buckets[bucket].append(i)

    samples_per_bucket = 40
    samples = []
    for i in range(len(buckets)):
        vals = vals_in_buckets[i]
        if len(vals) > 40:
            for j in range(40):
                samples.append(vals[random.randint(0, len(vals) - 1)])

    sampled_delta_1d = delta[samples]
    sampled_feature_mat = features[samples, :]
    return sampled_feature_mat, sampled_delta_1d

def plot_dist_vs_mean():
    delta_1d = load_file("mean_1d.txt")
    feature_mat = load_file("feature_mat.txt")

    feature_mat, delta_1d = filter_data(feature_mat, delta_1d)
    sampled_feature_mat, sampled_delta_1d = sampled_data(feature_mat, delta_1d)

    h = np.histogram(delta_1d, bins = np.arange(0, 1, 0.01), density = True)

    plt.figure()
    plt.plot(h[0], h[1][:-1], 'ro')
    plt.show()

    plt.figure()
    plt.plot(sampled_feature_mat[:, 0], sampled_delta_1d, 'ro')
    plt.axis([0, 0.5, min(delta_1d), max(delta_1d)])
    plt.show()

    plt.figure()
    plt.plot(feature_mat[:, 0], delta_1d, 'ro')
    plt.axis([0, 0.5, min(delta_1d), max(delta_1d)])
    plt.show()

def plot_delta_hist():
    delta_1d = load_file("delta_1d.txt")
    plt.figure()
    plt.hist(delta_1d, bins=np.arange(-180, 180, 10))
    plt.show()

def delta_svm_3class():
    delta_thresh = 0.000000001
    dist_thresh = 1

    delta_1d = load_file("delta_1d.txt")
    feature_mat = load_file("feature_mat.txt")

    close_vals = np.where(feature_mat[:, 0] < dist_thresh)
    delta_1d_filtered = delta_1d[close_vals]
    distances = feature_mat.shape[1] + np.sum(feature_mat[:, 1:], 1)
    new_feature_mat = np.zeros([feature_mat.shape[0], 2])
    new_feature_mat[:, 0] = feature_mat[:, 0]
    new_feature_mat[:, 1] = distances
    feature_mat_filtered = new_feature_mat[close_vals]

    feature_mat_filtered = feature_mat_filtered - np.tile(np.mean(feature_mat_filtered, 0), [feature_mat_filtered.shape[0], 1])
    feature_mat_filtered = feature_mat_filtered / np.tile(np.max(np.abs(feature_mat_filtered), 0), [feature_mat_filtered.shape[0], 1])
    print feature_mat_filtered

    ind_val_0 = np.where(np.abs(delta_1d_filtered) < -delta_thresh)
    ind_val_1 = np.where(np.abs(delta_1d_filtered) >= delta_thresh)

    print "total:" + str(len(delta_1d_filtered))
    print "significant:" + str(ind_val_1[0].shape)

    delta_1d_bin = np.zeros(len(delta_1d_filtered))
    delta_1d_bin[ind_val_0] = -1
    delta_1d_bin[ind_val_1] = 1

    clf = svm.SVC(kernel='rbf', C = 1)
    a=cross_val_score(clf, feature_mat_filtered, delta_1d_bin, cv=10)
    print a

def mean_svm_sigvsnot():
    mean_thresh = 0.000000001
    dist_thresh = 1

    mean_1d = load_file("mean_1d.txt")
    feature_mat = load_file("feature_mat.txt")

    close_vals = np.where(feature_mat[:, 0] < dist_thresh)
    mean_1d_filtered = mean_1d[close_vals]
    distances = feature_mat.shape[1] + np.sum(feature_mat[:, 1:], 1)
    new_feature_mat = np.zeros([feature_mat.shape[0], 2])
    new_feature_mat[:, 0] = feature_mat[:, 0]
    new_feature_mat[:, 1] = distances
    feature_mat_filtered = new_feature_mat[close_vals]

    feature_mat_filtered = feature_mat_filtered - np.tile(np.mean(feature_mat_filtered, 0), [feature_mat_filtered.shape[0], 1])
    feature_mat_filtered = feature_mat_filtered / np.tile(np.max(np.abs(feature_mat_filtered), 0), [feature_mat_filtered.shape[0], 1])
    print feature_mat_filtered

    ind_val_0 = np.where(np.abs(mean_1d_filtered) < mean_thresh)
    ind_val_1 = np.where(np.abs(mean_1d_filtered) >= mean_thresh)


    print "total:" + str(len(mean_1d_filtered))
    print "significant:" + str(ind_val_1[0].shape)

    mean_1d_bin = np.zeros(len(mean_1d_filtered))
    mean_1d_bin[ind_val_0] = -1
    mean_1d_bin[ind_val_1] = 1

    clf = svm.SVC(kernel='rbf', C = 1)
    a=cross_val_score(clf, feature_mat_filtered, mean_1d_bin, cv=10)
    print a

def delta_svm_posvsneg():
    delta_thresh = 0.1
    dist_thresh = 1

    delta_1d = load_file("gen_trend_1d.txt")
    feature_mat = load_file("feature_mat.txt")

    feature_mat, delta_1d = filter_data(feature_mat, delta_1d)
    #feature_mat, delta_1d = sampled_data(feature_mat, delta_1d)

    close_vals = np.where(feature_mat[:, 0] < dist_thresh)
    delta_1d_filtered = delta_1d[close_vals]
    feature_mat_filtered = feature_mat[close_vals]

    plt.figure()
    plt.plot(np.sqrt(feature_mat_filtered[:, 0]), delta_1d_filtered, 'ro')
    plt.axis([0, 20, min(delta_1d_filtered), max(delta_1d_filtered)])
    plt.show()
    #keep only significant indices
    delta_1d_filtered2 = delta_1d_filtered
    #remove the corresponding entries in feature_mat_filtered
    feature_mat_filtered2 = preprocessing.scale(feature_mat_filtered)
    #do the labeling in a new vector delta_1d_bin
    indpos = np.where( abs(delta_1d_filtered2) >0.2)
    delta_1d_bin = np.zeros(len(delta_1d_filtered2))
    delta_1d_bin[:] = -1
    delta_1d_bin[indpos] = 1

    print "total:" + str(len(delta_1d_filtered2))
    print "positive:" + str(len(indpos[0]))


    clf = svm.SVC(kernel='rbf', C = 1)
    a=cross_val_score(clf, feature_mat_filtered2, delta_1d_bin, cv=10)
    print a
    print "average cross validation score: " + str(np.average(a))

def generic_sigvsnonsig(label_file,clf,remove_ratio):	
	corr_1d=load_file(label_file)
	feature_mat=load_file("feature_mat.txt")
	feature_mat, corr_1d = filter_data(feature_mat, corr_1d)
	N = len(corr_1d)
	thresh1 = (0.5 - remove_ratio/2)*100
	thresh2 = (0.5 + remove_ratio/2)*100
	minor_cutoff = np.percentile(abs(corr_1d),thresh1)
	major_cutoff = np.percentile(abs(corr_1d),thresh2)
	med_data = median(abs(corr_1d))
	ind=np.where( (abs(corr_1d)>major_cutoff) | (abs(corr_1d)<minor_cutoff) ) #movave
	corr_1d_filtered=corr_1d[ind]
	ind_bin=np.where(abs(corr_1d_filtered) > med_data)
	corr_1d_bin = np.zeros(len(corr_1d_filtered))
	corr_1d_bin[ind_bin] = 1
	print "abs med: " +str(major_cutoff)
	print "total data:"
	print len(ind[0])
	print "significant corr:"
	print len(ind_bin[0])

	
	feature_mat_filtered=feature_mat[ind]

	print feature_mat_filtered.shape
	#try remove the distances from the features
	feature_mat_filtered2 = feature_mat_filtered[:,0:];
	#standardize the feature 'distance' through operations in dum:
	dum = feature_mat_filtered2[:,0]
	dum = 2 * dum / np.amax(dum)
	feature_mat_filtered2[:,0] = dum
	a=cross_val_score(clf, feature_mat_filtered, corr_1d_bin, cv=10)
	train_score = clf.fit(feature_mat_filtered, corr_1d_bin).score(feature_mat_filtered, corr_1d_bin)
	print a
	print "training score: " + str(train_score);
	print "average cross validation score: " + str(np.average(a))
def generic_posvsneg(label_file,clf,remove_ratio):	
	corr_1d=load_file(label_file)
	feature_mat=load_file("90_feature_mat.txt")
	feature_mat, corr_1d = filter_data(feature_mat, corr_1d)
	N = len(corr_1d)
	ind_neg = np.where( (corr_1d<0)) 
	numneg = len(ind_neg[0])
	ratneg_to_keep = 100 * (1-remove_ratio)/2 
	print "ratneg to keep: " + str(ratneg_to_keep)
	neg_thresh = np.percentile(corr_1d,ratneg_to_keep)
	if neg_thresh > 0:
		neg_thresh = 0
		ratpos_to_keep = 1-numneg/N
	else:
		ratpos_to_keep = 100-ratneg_to_keep
	pos_thresh = np.percentile(corr_1d,ratpos_to_keep)
	if pos_thresh < 0:
		pos_thresh = 0;
	print pos_thresh
	print neg_thresh
	ind=np.where( (corr_1d<neg_thresh) | (corr_1d>pos_thresh) ) 
	corr_1d_filtered=corr_1d[ind]
	ind_bin=np.where(corr_1d_filtered > 0)
	corr_1d_bin = np.zeros(len(corr_1d_filtered))
	corr_1d_bin[ind_bin] = 1
	
	print "total neg: " + str(numneg)
	print "total data:"
	print len(ind[0])
	print "significant corr:"
	print len(ind_bin[0])
	feature_mat_filtered=feature_mat[ind]

	print feature_mat_filtered.shape
	#try remove the distances from the features
	feature_mat_filtered2 = feature_mat_filtered[:,0:];
	#standardize the feature 'distance' through operations in dum:
	dum = feature_mat_filtered2[:,0]
	dum = 2 * dum / np.amax(dum)
	feature_mat_filtered2[:,0] = dum
	a=cross_val_score(clf, feature_mat_filtered2, corr_1d_bin, cv=10)
	train_score = clf.fit(feature_mat_filtered, corr_1d_bin).score(feature_mat_filtered, corr_1d_bin)
	print a
	print "training score: " + str(train_score);
	print "average cross validation score: " + str(np.average(a))
	

if __name__ == "__main__":
	#clf = svm.SVC(kernel='rbf', C = 1)
	clf= LogisticRegression(penalty='l1', tol=0.01,C =1)
	generic_sigvsnonsig("90_gen_trend_1d.txt",clf,0.2)
    # plot_dist_vs_delta()
    # delta_svm_posvsneg()
