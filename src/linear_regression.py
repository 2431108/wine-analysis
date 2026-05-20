import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("data/wine.csv")
X = df['alcohol'].values.reshape(-1, 1) # 説明変数
y = df['quality'].values # ⽬的変数

# 切⽚項追加（1の列を追加）
X_b = np.hstack([np.ones_like(X), X])

# 最⼩⼆乗法で回帰係数を求める
beta, residuals, rank, s = np.linalg.lstsq(X_b, y,
rcond=None)

# 予測値の計算
y_pred = X_b @ beta
# プロット
plt.scatter(X, y, label='Data points')
plt.plot(X, y_pred, color='red', label='Fitted line')
plt.xlabel('alcohol')
plt.ylabel('quality')
plt.legend()
plt.show()
