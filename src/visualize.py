import pandas as pd
import matplotlib.pyplot as plt

df= pd.read_csv('data/wine.csv')

corr= df.corr()
plt.imshow(corr, cmap="coolwarm")
plt.colorbar()
plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
plt.yticks(range(len(corr.columns)), corr.columns)
plt.tight_layout()
plt.show()
