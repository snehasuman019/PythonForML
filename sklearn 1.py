import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_diabetes

X, y = load_diabetes(return_X_y=True)
##print(X.shape, y.shape)
##print(X, y)
##from 
from sklearn.neighbors import KNeighborsRegressor
mod = KNeighborsRegressor()
print(mod.fit(X, y))
print(mod)
pred=mod.predict(X)
plt.scatter(pred, y)
plt.show()
