# https://pythonprogramming.net/regression-introduction-machine-learning-tutorial/
import pandas as pd
import quandl

df = quandl.get('WIKI/GOOGL')

# adjusted prices account for stock splits - more reliable
df = df[['Adj. Open',  'Adj. High',  'Adj. Low',  'Adj. Close', 'Adj. Volume']]

# high - close percent change
df['HL_PCT'] = (df['Adj. High'] - df['Adj. Close']) /  df['Adj. Close'] * 100.0

# percent change
df['PCT_change'] = (df['Adj. Close'] - df['Adj. Open']) /  df['Adj. Open'] * 100.0

# only get the most important features of the dataset
df = df[['Adj. Close', 'HL_PCT', 'PCT_change', 'Adj. Volume']]

print(df.head())