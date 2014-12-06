import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error, make_scorer
from sklearn.cross_validation import KFold, cross_val_score
from sklearn.linear_model import Lasso
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import BernoulliNB
from sklearn import svm
from numpy import *
from pylab import *

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
	ind=np.where( (corr_1d>0.65) | (corr_1d<0) )
	corr_1d_filtered=corr_1d[ind]
	ind_bin=np.where(corr_1d_filtered > 0)
	corr_1d_bin = np.zeros(len(corr_1d_filtered))
	corr_1d_bin[ind_bin] = 1
	print ind[0].shape
	print "positive corr:"
	print ind_bin[0].shape
	print corr_1d_bin
	print corr_1d_filtered

	feature_mat=load_file("feature_mat.txt")
	feature_mat_filtered=feature_mat[ind]

	print feature_mat_filtered.shape
	#try remove the distances from the features
	feature_mat_filtered2 = feature_mat_filtered[:,0:];
	#standardize the feature 'distance' through operations in dum:
	dum = feature_mat_filtered[:,0]
	dum = 2 * dum / np.amax(dum)
	dum = dum-1
	feature_mat_filtered[:,0] = dum
	clf = svm.SVC(kernel='poly', C=1)
	a=cross_val_score(clf, feature_mat_filtered2, corr_1d_bin, cv=15)
	print a
	print "average cross validation error: " + str(np.average(a))

def svm_asymthresh():
	corr_thresh=0.36;
	corr_1d=load_file("corr_1d.txt")
	ind=np.where(corr_1d>corr_thresh)
	corr_1d_bin = np.zeros(len(corr_1d))
	corr_1d_bin[ind] = 1

	print "high positive corr:"
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
	a=cross_val_score(clf, feature_mat_filtered, corr_1d_bin, cv=10)
	print a
	print "average cross validation error: " + str(np.average(a))


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

if __name__ == "__main__":
	mysvm()

