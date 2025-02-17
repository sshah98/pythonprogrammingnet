# https://pythonprogramming.net/euclidean-distance-machine-learning-tutorial
# Euclidean Distance theory

# k nearest neighbors is good for nonlinear data

# Euclidean Distance - sqrt(sum to n from i=1(q(i) - p(i))^2)
# ex: q = (1,3)
#     p = (2,5)
# two dimensions
# sqrt((1-2)^2 + (3-5)^2)
import random
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import sqrt
from matplotlib import style
from collections import Counter

style.use('fivethirtyeight')

# euclidean_distance = sqrt((plot1[0]-plot2[0])**2 + (plot1[1]-plot2[1])**2)
# print(euclidean_distance)

# ========================================================================= #
# ========================================================================= #
# https://pythonprogramming.net/programming-k-nearest-neighbors-machine-learning-tutorial
# two classes and their features
dataset = {'k':[[1,2], [2,3], [3,1]], 'r': [[6,5], [7,7], [8,6]]}
new_features = [5,7]



# for i in dataset:
    # for ii in dataset[i]:
# [[plt.scatter(ii[0], ii[1], s=100, color=i) for ii in dataset[i]] for i in dataset]

# plt.show()

def k_nearest_neighbors(data, predict, k=3):
    if len(data) >= k:
        warnings.warn('K is set to a value less than total voting groups')
        
    distances = []
    for group in data:
        for features in data[group]:
                # euclidean_distance = sqrt((features[0]-predict[0])**2 + (features[1]-predict[1])**2)
                euclidean_distance = np.linalg.norm(np.array(features)-np.array(predict))
                # distances and the groups, to sort the list later
                distances.append([euclidean_distance, group])
    
    # i[1]=group
    votes = [i[1] for i in sorted(distances)[:k]]
    # gets the most commmon in the list of votes (median)
    # most_common = list of tuples, so you have to do [0][0]
    # print(Counter(votes).most_common(1))
    vote_result = Counter(votes).most_common(1)[0][0]
    confidence = Counter(votes).most_common(1)[0][1] / k
        
    return vote_result, confidence
    
# result = k_nearest_neighbors(dataset, new_features, k=3)
# print(result)


# ========================================================================= #
# ========================================================================= #
# https://pythonprogramming.net/coding-k-nearest-neighbors-machine-learning-tutorial
# Creating a K Nearest Neighbors Classifer from scratch part 2

# [[plt.scatter(ii[0], ii[1], s=100, color=i) for ii in dataset[i]] for i in dataset]
# plt.scatter(new_features[0], new_features[1], color=result, s=100)
# plt.show()

# ========================================================================= #
# ========================================================================= #
# https://pythonprogramming.net/testing-our-k-nearest-neighbors-machine-learning-tutorial
# Testing our K Nearest Neighbors classifier
accuracies = []

for i in range(5):
    
    df = pd.read_csv('breast-cancer-wisconsin.data.txt')
    df.replace('?', -99999, inplace=True)
    df.drop(['id'], 1, inplace=True)
    # necessary to make all the data as floats
    full_data = df.astype(float).values.tolist()
    random.shuffle(full_data)
    test_size = 0.2
    train_set = {2:[], 4:[]}
    test_set = {2:[], 4:[]}
    # creating an index value and slicing it for the train_data - 80% of the data
    train_data = full_data[:-int(test_size*len(full_data))]
    # the last 20% of the datas
    test_data = full_data[-int(test_size*len(full_data)):]

    for i in train_data:
        # the last value (2 or 4); appending up to the last element
        train_set[i[-1]].append(i[:-1])

    for i in test_data:
        # the last value (2 or 4); appending up to the last element
        test_set[i[-1]].append(i[:-1])

    correct = 0
    total = 0

    for group in test_set:
        # only the list of features
        for data in test_set[group]:
            vote, confidence = k_nearest_neighbors(train_set, data, k=5)
            if group == vote:
                correct += 1
            else:
                # print votes that were incorrect
                # print(confidence)
                pass
            total += 1
            
    # print('Accuracy:', correct/total)
    accuracies.append(correct/total)

# ========================================================================= #
# ========================================================================= #
# https://pythonprogramming.net/final-thoughts-knn-machine-learning-tutorial
# Final thoughts on K Nearest Neighbors

print(sum(accuracies)/len(accuracies))




    
    
    
    
    
    
        
    
    







