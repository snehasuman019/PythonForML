import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix
##
##data = load_iris()
##x=data.data
##y=data.target
##x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=21,test_size=0.25)
##nb=GaussianNB()
##nb.fit(x_train,y_train)
##nb_pred = nb.predict(x_test)
##print(y)
##print(x)
##print("Accuracy Score: ",accuracy_score(y_test,nb_pred))
##print("Confusion matrix: ",confusion_matrix(y_test, nb_pred))
##


data = pd.read_csv(("E:\\D drive\\SEM 5\\234\\Social_Network_Ads.csv"),encoding="Latin1")
##print(data)
x=data.iloc[:,:-1].values
y=data.iloc[:,-1].values
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25, random_state=42)
nb = GaussianNB()
nb.fit(x_train, y_train)
nb_pred = nb.predict(x_test)
##print(x)
##print(y)
print("Accuracy score: ",accuracy_score(y_test, nb_pred))
print("Confusion matrix: ",confusion_matrix(y_test, nb_pred))
