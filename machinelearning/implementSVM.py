# https://pythonprogramming.net/vector-basics-machine-learning-tutorial
# Vector Basics
# vectors are coordinates
# vector magnitudes are the sum of the squares, and then the sqrt of that sum
# dot product = A[1,3], B[4,2] --> A . B = 1*4 + 3*2 = 10 scalar value

# ======================================================================== #
# ======================================================================== #
# https://pythonprogramming.net/support-vector-assertions-machine-learning-tutorial
# Support Vector Assertions

# unknown vector . main vector + bias

# +class = x vector . main vector + b = 1
# so then becomes y vector(x vector . main vector + b) - 1 = 0

# -class = x vector . main vector + b = -1
# so then becomes y vector(x vector . main vector + b) - 1 = 0

# ======================================================================== #
# ======================================================================== #

# https://pythonprogramming.net/support-vector-machine-fundamentals-machine-learning-tutorial
# Support Vector Machine Fundamentals

# equation for a hyperplane = (w*x) + b --> similar to mx+b

# ======================================================================== #
# ======================================================================== #

# https://pythonprogramming.net/svm-constraint-optimization-machine-learning-tutorial
# Constraint Optimization with Support Vector Machine

# convex - the line is the magnitude of w (main vector)
# to minimize this w, find the lowest points
# suppose you drop a ball in a bowl, it will eventually be in the middle
# therefore, the ball in this case is the vectors

# ======================================================================== #
# ======================================================================== #

# https://pythonprogramming.net/svm-in-python-machine-learning-tutorial
# Beginning SVM from Scratch in Python

# https://pythonprogramming.net/svm-optimization-python-machine-learning-tutorial
# Support Vector Machine Optimization in Python

# https://pythonprogramming.net/svm-optimization-python-2-machine-learning-tutorial
# Support Vector Machine Optimization in Python part 2

import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np
style.use('ggplot')


class Support_Vector_Machine:
    def __init__(self, visualization=True):
        self.visualization = visualization
        self.colors = {1: 'r', -1: 'b'}
        if self.visualization:
            self.fig = plt.figure()
            self.ax = self.fig.add_subplot(1, 1, 1)

    # train
    # this is the optimization part of svm
    def fit(self, data):
        self.data = data

        # the dictionary that will be populated for formulaic computations
        # {||w|| : [w,b]}
        opt_dict = {}

        # what we will applly to the vector w for stepping to get the minimized value
        transforms = [[1, 1], [-1, 1], [-1, -1], [1, -1]]

        # to get max, min ranges for the graph and where we start for a value with b and w
        all_data = []
        for yi in self.data:
            for featureset in self.data[yi]:
                for feature in featureset:
                    all_data.append(feature)
        self.max_feature_value = max(all_data)
        self.min_feature_value = min(all_data)

        all_data = None

        # take big steps first and then smaller steps for the range for the min
        step_sizes = [self.max_feature_value * 0.1,
                      self.max_feature_value * 0.01,
                      # point of expense - more accurate, but longer time
                      self.max_feature_value * 0.001,
                      ]

        # extremely expensive cost (b doesn't need as small/precise as w)
        b_range_multiple = 5

        #
        b_multiple = 5

        # first element in vector w
        latest_optimum = self.max_feature_value * 10

        for step in step_sizes:
            # major corner/optimization being cut here by keeping same value
            w = np.array([latest_optimum, latest_optimum])

            # we can do this because of the convex problem
            optimized = False

            while not optimized:

                for b in np.arange(-1 * (self.max_feature_value * b_range_multiple), self.max_feature_value * b_range_multiple, step * b_multiple):

                    for transformation in transforms:
                        # applying each transformation in the list to the original w vector (the max value we can get)
                        w_t = w * transformation
                        found_option = True
                        # weakest link in SVM fundamentally
                        # SMO attempts to fix this a bit
                        # yi(xi.w+b) >= 1 (the function)
                        for i in self.data:
                            # i is the class
                            for xi in self.data[i]:
                                yi = i
                                if not yi * (np.dot(w_t, xi) + b) >= 1:
                                    found_option = False
                                # print(xi, yi * (np.dot(w_t, xi) + b))
                                    # can speed it up here with a break
                                    
                        if found_option:
                            # how to get magnitude of a vector
                            opt_dict[np.linalg.norm(w_t)] = [w_t, b]

                if w[0] < 0:
                    optimized = True
                    print('Optimized a step.')
                else:
                    # w = [5,5], step is a scalar value.
                    w = w - step

            # these are a sorted list of ALL the magnitudes
            norms = sorted([n for n in opt_dict])
            # optimial choice is the 0th element, so the smallest magnitude
            # ||w|| : [w,b]
            opt_choice = opt_dict[norms[0]]

            self.w = opt_choice[0]
            self.b = opt_choice[1]
            latest_optimum = opt_choice[0][0] + step * 2
            
        for i in self.data:
            # i is the class
            for xi in self.data[i]:
                yi = i
                print(xi, ':', yi * (np.dot(self.w, xi) + self.b))

        # how do you know if you need to take more steps:
        # support vectors yi(xi . w + b) = 1
        # if you want to be less than 2, etc. until you hit that value, you need to keep stepping

    def predict(self, features):
        # sign(x . w + b)
        # the values needed to be found are w and b
        # formula for finding the svm of features
        classification = np.sign(np.dot(np.array(features), self.w) + self.b)

        if classification != 0 and self.visualization:
            self.ax.scatter(features[0], features[1], s=200, marker='*', c=self.colors[classification])

        return classification

    def visualize(self):
        [[self.ax.scatter(x[0], x[1], s=100, color=self.colors[i])
          for x in data_dict[i]] for i in data_dict]

        # hyperplane = x.w+b
        # objective is to plot the points and the actual support vector hyperplanes and the decision boundary hyperplane
        # v=value we are seeking
        # want v=x.w+b
        # positive support vector=1
        # negative support vector=-1
        # decision boundary = 0
        def hyperplane(x, w, b, v):
            return (-w[0] * x - b + v) / w[1]

        # to limit graph for datapoints
        datarange = (self.min_feature_value * 0.9,
                     self.max_feature_value * 1.1)

        hyp_x_min = datarange[0]
        hyp_x_max = datarange[1]

        # positive sv = (w.x+b) =  1; going to be a scalar value; on a graph will be your y's
        psv1 = hyperplane(hyp_x_min, self.w, self.b, 1)
        psv2 = hyperplane(hyp_x_max, self.w, self.b, 1)

        self.ax.plot([hyp_x_min, hyp_x_max], [psv1, psv2], 'k')

        # negative sv = (w.x+b) = -1
        nsv1 = hyperplane(hyp_x_min, self.w, self.b, -1)
        nsv2 = hyperplane(hyp_x_max, self.w, self.b, -1)
        self.ax.plot([hyp_x_min, hyp_x_max], [nsv1, nsv2], 'k')

        # decision boundary = (w.x+b) = 0
        db1 = hyperplane(hyp_x_min, self.w, self.b, 0)
        db2 = hyperplane(hyp_x_max, self.w, self.b, 0)
        self.ax.plot([hyp_x_min, hyp_x_max], [db1, db2], 'y--')

        plt.show()


data_dict = {-1: np.array([[1, 7], [2, 8], [3, 8], ]),
             1: np.array([[5, 1], [6, -1], [7, 3], ])}
             

svm = Support_Vector_Machine()
svm.fit(data=data_dict)

predict_us = [[0,10], [1,3], [3,4], [3,5], [5,5], [5,6], [6,-5],  [5,8]]
for p in predict_us:
    svm.predict(p)


svm.visualize()

             
             
             
             
             
             
             
             
             
             
             
             
             
             
