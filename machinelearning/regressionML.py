import math, quandl, datetime, pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing, cross_validation, svm
from sklearn.linear_model import LinearRegression
from matplotlib import style

style.use('ggplot')

# https://pythonprogramming.net/regression-introduction-machine-learning-tutorial/
# Regression - Intro and Data

df = quandl.get('WIKI/GOOGL')

# adjusted prices account for stock splits - more reliable
df = df[['Adj. Open',  'Adj. High',  'Adj. Low',  'Adj. Close', 'Adj. Volume']]

# high - close percent change
df['HL_PCT'] = (df['Adj. High'] - df['Adj. Close']) /  df['Adj. Close'] * 100.0

# percent change
df['PCT_change'] = (df['Adj. Close'] - df['Adj. Open']) /  df['Adj. Open'] * 100.0

# only get the most important features of the dataset
df = df[['Adj. Close', 'HL_PCT', 'PCT_change', 'Adj. Volume']]
# print(df.head())

# ============================================================= #
# ============================================================= #


# https://pythonprogramming.net/features-labels-machine-learning-tutorial/
# Regression - Features and Labels

# can change this forecast later
forecast_col = 'Adj. Close'

# can't work with NaN data so need to have extreme outlier rather than dropping the entire column
df.fillna(-99999, inplace=True)

# will be the number of days out (1% of the dataframe) 30 days in advanced
forecast_out = int(math.ceil(0.01*len(df)))

# label column; shifting the columns negatively (up), each row will be the adjusted close price, 10 days into the future
df['label'] = df[forecast_col].shift(-forecast_out)
# print(df.tail())

# ============================================================= #
# ============================================================= #


# https://pythonprogramming.net/training-testing-machine-learning-tutorial
# Regression - Training and Testing

# scaling data is done on features

# define x and y - features will be X, labels will be y
X = np.array(df.drop(['label'], 1))

# this scales X to normalize with other data points - NEED to scale the new values with the other values (training data)
X = preprocessing.scale(X)

X_lately = X[-forecast_out:]
X = X[:-forecast_out]

df.dropna(inplace=True)

y = np.array(df['label'])

# print(len(X), len(y))

# takes features and labels, randomizes it, and then outputs to the training and testing data
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)

# classifier - LinearRegression() is better than svm.SVR()
# clf = svm.SVR()
# n_jobs = number of threads that the classifier can run (-1 as many as cpu can run)
# clf = LinearRegression(n_jobs=-1)
# clf.fit(X_train, y_train)

# with open('linearregression.pickle', 'wb') as f:
#     pickle.dump(clf, f)
    
pickle_in = open('linearregression.pickle', 'rb')
clf = pickle.load(pickle_in)

accuracy = clf.score(X_test, y_test)
# print(accuracy)

# ============================================================= #
# ============================================================= #

# https://pythonprogramming.net/forecasting-predicting-machine-learning-tutorial
# Regression - Forecasting and Predicting

# these are the next values for the 30 days
forecast_set = clf.predict(X_lately)
# prints the stock value, accuracy, and the number of days
# print(forecast_set, accuracy, forecast_out)

df['Forecast'] = np.nan

last_date = df.iloc[-1].name
last_unix = last_date.timestamp()
one_day = 86400
next_unix = last_unix + one_day
print(last_date)

# iterating through forecast set, setting those values in the dataframe
# to get dates on the axis
for i in forecast_set:
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += one_day
    df.loc[next_date] = [np.nan for _ in range(len(df.columns) - 1)] + [i]
    
print(df.tail(15))

df['Adj. Close'].plot()
df['Forecast'].plot()
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()

# ============================================================= #
# ============================================================= #

# https://pythonprogramming.net/pickling-scaling-machine-learning-tutorial
# Pickling and Scaling

# used pickle for the classifier after it was on the trained data

# Virtual servers can be set up in about 60 seconds, the required modules used in this tutorial can all be installed in about 15 minutes or so at a fairly leisurely pace. You could write a shell script or something to speed it up too. Consider that you need a lot of processing

# generally you can spin up a very small server, load what you need, then scale UP that server








