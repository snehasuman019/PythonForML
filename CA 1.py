

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv(("C:\\Users\\Sneha\\Downloads\\swiss.csv"),encoding = "Latin 1")
##print(data.head())
x=data.iloc[:,:-1].values
y=data.iloc[:,-1].values
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(x_train,y_train)
y_pred = model.predict(x_test)
r2=r2_score(y_test,y_pred)
print("r2_score: ",r2)

plt.scatter(y_test, y_pred)
plt.xlabel("Actual Fertility")
plt.ylabel("Predicted Fertility")
plt.title("Actual vs Predicted Fertility (Swiss Data)")
plt.show()
