import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
data=pd.read_csv(('E:\\D drive\\SEM 5\\234\\dataeg.csv'),encoding="Latin1")
x=data.iloc[:,:-1].values
y=data.iloc[:,-1].values
model=LinearRegression()
model.fit(x,y)
y_pred=model.predict(x)
r2=r2_score(y,y_pred)
print("R_Squarederror",r2)
