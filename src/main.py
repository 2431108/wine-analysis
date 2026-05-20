import pandas as pd

df= pd.read_csv('data/wine.csv')

print(df.info())
print(df.describe())
print(df.head())
