import sys
import scipy.io as sio
from pprint import pprint
import numpy as np
from sklearn.naive_bayes import GaussianNB

train_mat_fname = "data_train.mat"
train_mat_dict = sio.loadmat(train_mat_fname)
data_train = train_mat_dict['data_train']

X = data_train[:, [0, 1, 2, 3, 4]]
Y = []
for label in data_train[:, [5]]:
	Y.append(label[0])

clf = GaussianNB()
clf.fit(X, Y)
#Variances 
# pprint(clf.sigma_)

test_mat_fname = "data_test.mat"
test_mat_dict = sio.loadmat(test_mat_fname)
data_test = test_mat_dict['data_test']

X_test = data_train[:, [0, 1, 2, 3, 4]]
Y_test_label = []
data_test_length = 0
for label in data_test[:, [5]]:
	Y_test_label.append(label[0])
	data_test_length += 1

Y_pred = clf.predict(X_test)
counter = 0
for i in range(0, data_test_length):
	if Y_pred[i] == Y_test_label[i]:
		counter += 1

print("Accuracy: %", (100 * counter/float(data_test_length)))
