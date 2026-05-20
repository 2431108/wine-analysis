import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('data/wine.csv').to_numpy()
X = df[:, :-1]         # 特徴量 
y = df[:, -1]          # 品質

X_mean = X.mean(axis=0)
X_std = X.std(axis=0)
X_scaled = (X - X_mean) / X_std
cov_matrix = np.cov(X_scaled, rowvar=False)
eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
order = np.argsort(eigenvalues)[::-1]
eigenvalues  = eigenvalues[order]
eigenvectors = eigenvectors[:, order]
# 主成分スコアを計算
PC_scores = X_scaled @ eigenvectors
PC1 = PC_scores[:, 0]
PC2 = PC_scores[:, 1]

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
axes[0].plot(PC1, y, '.')
axes[0].set_xlabel("PC1")
axes[0].set_ylabel("quality")
axes[1].plot(PC2, y, '.')
axes[1].set_xlabel("PC2")
axes[1].set_ylabel("quality")
plt.show()

