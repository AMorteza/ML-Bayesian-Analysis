import sys
from pprint import pprint
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

X_dict = {}
X_dict["Outlook"] = {"Sunny": 0, "Overcast": 1, "Rain": 2}
X_dict["Temp"] = {"Hot": 0, "Mild": 1, "Cool": 2}
X_dict["Humidity"] = {"High": 0, "Normal": 1}
X_dict["Wind"] = {"False": 0, "Weak": 1, "True": 2}

X = np.array([
	[X_dict["Outlook"]["Sunny"], X_dict["Temp"]["Hot"], X_dict["Humidity"]["High"], X_dict["Wind"]["False"]],
	[X_dict["Outlook"]["Sunny"], X_dict["Temp"]["Hot"], X_dict["Humidity"]["High"], X_dict["Wind"]["True"]],
	[X_dict["Outlook"]["Overcast"], X_dict["Temp"]["Hot"], X_dict["Humidity"]["High"], X_dict["Wind"]["Weak"]],
	[X_dict["Outlook"]["Rain"], X_dict["Temp"]["Mild"], X_dict["Humidity"]["High"], X_dict["Wind"]["Weak"]],
	[X_dict["Outlook"]["Rain"], X_dict["Temp"]["Cool"], X_dict["Humidity"]["Normal"], X_dict["Wind"]["False"]],
	[X_dict["Outlook"]["Rain"], X_dict["Temp"]["Cool"], X_dict["Humidity"]["Normal"], X_dict["Wind"]["True"]],
	[X_dict["Outlook"]["Overcast"], X_dict["Temp"]["Cool"], X_dict["Humidity"]["Normal"], X_dict["Wind"]["True"]],
	[X_dict["Outlook"]["Sunny"], X_dict["Temp"]["Mild"], X_dict["Humidity"]["High"], X_dict["Wind"]["False"]],
	[X_dict["Outlook"]["Sunny"], X_dict["Temp"]["Cool"], X_dict["Humidity"]["Normal"], X_dict["Wind"]["False"]],
	[X_dict["Outlook"]["Rain"], X_dict["Temp"]["Mild"], X_dict["Humidity"]["Normal"], X_dict["Wind"]["False"]],
	[X_dict["Outlook"]["Sunny"], X_dict["Temp"]["Mild"], X_dict["Humidity"]["Normal"], X_dict["Wind"]["True"]],
	[X_dict["Outlook"]["Overcast"], X_dict["Temp"]["Mild"], X_dict["Humidity"]["High"], X_dict["Wind"]["True"]],
	[X_dict["Outlook"]["Overcast"], X_dict["Temp"]["Hot"], X_dict["Humidity"]["Normal"], X_dict["Wind"]["False"]],
	[X_dict["Outlook"]["Rain"], X_dict["Temp"]["Mild"], X_dict["Humidity"]["High"], X_dict["Wind"]["True"]],
])

X_missed_last_two_cols = X[:, [0, 1]]
y = np.array([0, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 0])
clf = LinearDiscriminantAnalysis()
y_pred = clf.fit(X_missed_last_two_cols, y).predict(X_missed_last_two_cols)


print("Predict for other 10 couple permutations:")
for i in X_dict["Outlook"]:
	data = []
	data.append(X_dict["Outlook"][i])
	for j in X_dict["Temp"]:
		data.append(X_dict["Temp"][j])			
		print("  X: ", data, "Y: ", clf.predict([data]))
		data.pop()
	data.pop()

