# https://pythonprogramming.net/simple-linear-regression-machine-learning-tutorial
# Regression - Theory and how it works

# slope = m = (meanX * meanY - mean(xy)) / ((meanX)^2 - mean(X^2)))
# b = meanY - m*meanX

# ============================================================= #
# ============================================================= #

# https://pythonprogramming.net/how-to-program-best-fit-line-slope-machine-learning-tutorial/
# Regression - How to program the Best Fit Slope

from statistics import mean
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style

style.use('fivethirtyeight')

xs = np.array([1, 2, 3, 4, 5, 6], dtype=np.float64)
ys = np.array([5, 4, 6, 5, 6, 7], dtype=np.float64)

# plt.scatter(xs, ys)
# plt.show()

def best_fit_slope_and_intercept(xs, ys):
    
    m = ((mean(xs) * mean(ys)) - mean(xs * ys)) / ((mean(xs)**2) - mean(xs**2))
    
    b = mean(ys) - m*mean(xs)
    
    return m, b
    
m, b = best_fit_slope_and_intercept(xs, ys)
# print(m, b)

# ============================================================= #
# ============================================================= #

# https://pythonprogramming.net/how-to-program-best-fit-line-machine-learning-tutorial
# Regression - How to program the Best Fit Line

regression_line = [(m*x) + b for x in xs]

# regression_line is the same as:
# for x in xs:
#   regression_line.append((m*x) + b)

# predict_x = 8
# predict_y = (m*predict_x + b)

# print(regression_line)
# plt.scatter(xs, ys)
# plt.scatter(predict_x, predict_y)
# plt.plot(xs, regression_line)
# plt.show()


# ============================================================= #
# ============================================================= #
# https://pythonprogramming.net/r-squared-coefficient-of-determination-machine-learning-tutorial

# Regression - R Squared and Coefficient of Determination Theory

# r^2 = 1 - (SEyhat)/(SEmean(y))
# 1 minus the division of the squared error of the regression line and the squared error of the mean y line. 

# r^2 = higher is better

# ============================================================= #
# ============================================================= #

# https://pythonprogramming.net/how-to-program-r-squared-machine-learning-tutorial
# Regression - How to Program R Squared

def squared_error(ys_orig, ys_line):
    return sum((ys_line - ys_orig) ** 2)

def coefficient_of_determination(ys_orig, ys_line):
    y_mean_line = [mean(ys_orig) for y in ys_orig]
    squared_error_regr = squared_error(ys_orig, ys_line)
    squared_error_y_mean = squared_error(ys_orig, y_mean_line)
    return 1 - (squared_error_regr / squared_error_y_mean)
    
r_squared = coefficient_of_determination(ys, regression_line)
# print(r_squared)

# ============================================================= #
# ============================================================= #

# https://pythonprogramming.net/sample-data-testing-machine-learning-tutorial
# Creating Sample Data for Testing




