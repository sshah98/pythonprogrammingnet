# https://pythonprogramming.net/k-nearest-neighbors-intro-machine-learning-tutorial
# Classification Intro with K Nearest Neighbors

# classification is the idea of best dividing or separating our data

# nearest neighbors - who is the closest point to a specific unknown data point
# k nearest neighbors - decide the number of neighbors to check
# k = 2= the two closest neighbors closest to the k data point
# k should be an odd number

# ==================================================================== #
# ==================================================================== #
# https://pythonprogramming.net/k-nearest-neighbors-application-machine-learning-tutorial
# Applying K Nearest Neighbors to Data

# label is what you try to predict 
import numpy as np
import pandas as pd
from sklearn import preprocessing, cross_validation, neighbors

df = pd.read_csv('breast-cancer-wisconsin.data.txt')
df.replace('?', -99999, inplace=True)
df.drop(['id'], 1, inplace=True)

# X is features, y is labels
X = np.array(df.drop(['class'], 1))
y = np.array(df['class'])

# 20% of separating and shuffling data 
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)

clf = neighbors.KNeighborsClassifier()
clf.fit(X_train, y_train)

accuracy = clf.score(X_test, y_test)
# print(accuracy)

example_measures = np.array([[4,2,1,1,1,2,3,2,1], [10,2,2,2,1,2,3,2,1]])
# import to do this bc it won't work without reshaping the array
# the len(example_measures) is the size of the array
example_measures = example_measures.reshape(len(example_measures),-1)


prediction = clf.predict(example_measures)
print(prediction)









