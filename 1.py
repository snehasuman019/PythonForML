import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder


##from sklearn.preprocessing import
data=pd.read_csv(("E:\\D drive\\SEM 5\\234\\Data.csv"),encoding="Latin1")
##print(data)

data['Age']=data['Age'].fillna(value=data['Age'].mean())
data['Salary']=data['Salary'].fillna(value=data['Salary'].mean())
sc=StandardScaler()
data['Age']=sc.fit_transform(data[['Age']])
data['Salary']=sc.fit_transform(data[['Salary']])
##print(data)
                                    
ho=OneHotEncoder(sparse_output=False)
data['Country']=ho.fit_transform(data[['Country']])
print(data)
