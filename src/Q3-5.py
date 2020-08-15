import sys
import scipy.io as sio
from pprint import pprint
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from math import pi
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
import matplotlib.pyplot as plt

iris_mat_fname = "iris.mat"
iris_mat_dict = sio.loadmat(iris_mat_fname)
X = iris_mat_dict['data']
y = iris_mat_dict['labels']


X = preprocessing.normalize(X)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=0.03, test_size=0.9, shuffle=True)

train_setosa = []
train_versicolor = []
train_virginica = []
for i in range(0, len(y_train)):
	if y_train[i] == 1:
		train_setosa.append(X_train[i])
	elif y_train[i] == 2:		 
		train_versicolor.append(X_train[i])
	elif y_train[i] == 3:		 
		train_virginica.append(X_train[i])

mean_setosa = np.mean(train_setosa)
variance_setosa= np.var(train_setosa)
mean_versicolor = np.mean(train_versicolor)
variance_versicolor= np.var(train_versicolor)
mean_virginica = np.mean(train_virginica)
variance_virginica= np.var(train_virginica)

#covariance square matrix
def variance_squaremat(var):
    return(np.multiply(var,[[1,0],[0,1]]))

variance_virginica = variance_squaremat(variance_virginica)
variance_setosa = variance_squaremat(variance_setosa)
variance_versicolor = variance_squaremat(variance_versicolor)

def bayes_classifier(test_point,mean,variance):
    mean = np.array([mean])
    X = test_point-mean
    p1 = ((1/(2*pi)**0.5)/(np.linalg.det(variance)))**0.5
    p2 = np.exp(-0.5*(np.matmul(np.matmul(X[0: 2],np.linalg.inv(variance)),np.transpose(X[0: 2]))))
    p2 += np.exp(-0.5*(np.matmul(np.matmul(X[2: 4],np.linalg.inv(variance)),np.transpose(X[2: 4]))))
    prior = 1/3 #prior probability
    P = prior* p1* p2     
    return(P)

counter = 0
test_size = len(X_test)
y_pred = []
for i in range(0, test_size):
	prob_virginica = bayes_classifier(X_test[i], mean_virginica, variance_virginica)
	prob_setosa = bayes_classifier(X_test[i], mean_setosa, variance_setosa)
	prob_versicolor = bayes_classifier(X_test[i], mean_versicolor, variance_versicolor)
	pred = 0
	if max(prob_virginica, prob_setosa, prob_versicolor) == prob_setosa:
		pred = 1
	elif max(prob_virginica, prob_setosa, prob_versicolor) == prob_versicolor:
		pred = 2
	elif max(prob_virginica, prob_setosa, prob_versicolor) == prob_virginica:
		pred = 3
	
	if pred == y_test[i]:
		counter += 1
	y_pred.append(pred)


np.set_printoptions(precision=2)
print("Bayes classifier using ML Accuracy: %", (counter*100/test_size))

