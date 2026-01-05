#Example 1
'''
from sklearn.preprocessing import PolynomialFeatures
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error
import matplotlib.pyplot as plt
df={
    'x':[1,2,3,4,5],
    'y':[20000,40000,60000,80000,100000]
    }
data=pd.DataFrame(df)
x=data.iloc[:,:-1].values
y=data.iloc[:,-1].values
poly=PolynomialFeatures(degree=4)
x_poly=poly.fit_transform(x)
model=LinearRegression()
model.fit(x_poly,y)
y_pred=model.predict(x_poly)
r2=r2_score(x,y_pred)
print("R2_Error: ",r2)
plt.scatter(x,y, color='blue',label='Actual data')
plt.plot(x,y_pred,color='red',label='Regression data')
plt.show()

#Example 2 [compare r2 value]
'''
from sklearn.preprocessing import PolynomialFeatures,OneHotEncoder
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
data=pd.read_csv(("E:\\D drive\\SEM 5\\234\\Position_Salaries.csv"),encoding ="Latin 1")
##pre=OneHotEncoder(sparse_output=False)
##data['Position']=pre.fit_transform(data[['Position']])
x=data.iloc[:,1:-1].values
y=data.iloc[:,-1].values
##x_train,x_test,y_train,y_test=train_test_split(x,y)
##random_state=2
##test_size=0.2
poly=PolynomialFeatures(degree=4)
x_poly=poly.fit_transform(x)
model=LinearRegression()
model.fit(x_poly,y)
y_pred=model.predict(x_poly)
model_lr=LinearRegression()
model_lr.fit(x,y)
y_pred2=model_lr.predict(x)
r2=r2_score(x,y_pred)
r2_1=r2_score(x,y_pred2)
print("R2_Error: ",r2)
print("R2_Error: ",r2_1)
plt.scatter(x,y,color='red',label='Actual data')
plt.plot(x,y_pred,color='blue',label='reg data')
plt.show()
plt.scatter(x,y,color='red',label='Actual data')
plt.plot(x,y_pred2,color='blue',label='reg data')
plt.show()

