# https://pythonprogramming.net/features-labels-machine-learning-tutorial/
# Regression - Features and Labels
import quandl
import pandas as pd
import math
from sklearn import preprocessing, cross_validation, svm
from sklearn.linear_model import LinearRegression
import numpy as np

df = quandl.get("WIKI/GOOGL")
df = df[['Adj. Open',  'Adj. High',  'Adj. Low',  'Adj. Close', 'Adj. Volume']]
df['HL_PCT'] = (df['Adj. High'] - df['Adj. Low']) / df['Adj. Close'] * 100.0
df['PCT_change'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100.0
df = df[['Adj. Close', 'HL_PCT', 'PCT_change', 'Adj. Volume']]
# print(df.head())

# can change this forecast later
forecast_col = 'Adj. Close'

# can't work with NaN data so need to have extreme outlier rather than dropping the entire column
df.fillna(-99999, inplace=True)

# will be the number of days out (1% of the dataframe)
forecast_out = int(math.ceil(0.01*len(df)))

# label column; shifting the columns negatively (up), each row will be the adjusted close price, 10 days into the future
df['label'] = df[forecast_col].shift(-forecast_out)
df.dropna(inplace=True)
print(df.tail())

# ============================================================= #

# https://pythonprogramming.net/training-testing-machine-learning-tutorial
# Regression - Training and Testing



