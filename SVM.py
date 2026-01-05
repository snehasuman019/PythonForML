import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

data = pd.read_csv("E:\\D drive\\SEM 5\\234\\Social_Network_Ads.csv",encoding="Latin 1")
data.head()

sc=StandardScaler()
data['EstimatedSalary']=sc.fit_transform(data[['EstimatedSalary']])
print(data.head())
x=data.iloc[:,:-1].values
y=data.iloc[:,-1].values

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.25,random_state=2)
model=SVC(kernel='linear')
model.fit(x_train,y_train)

pred=model.predict(x_test)
print("AccuracyScore: ",accuracy_score(pred,y_test))

plt.figure(figsize=(10, 7))
plt.scatter(x_train[:,0],x_train[:,1],c=y_train,cmap='coolwarm',edgecolors='k')
plt.title("Training Data")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
#plt.show()


ax=plt.gca()                                             #plot decision boundary
xlim = ax.get_xlim()
ylim = ax.get_ylim()

xx = np.linspace(xlim[0],xlim[1],100)               
yy = np.linspace(ylim[0],ylim[1],100)
yy,xx=np.meshgrid(yy,xx)
xy=np.vstack([xx.ravel(), yy.ravel()]).T

z=model.decision_function(xy).reshape(xx.shape)          #predict on grid

ax.contour(xx, yy, z, colors='k',
           levels=[-1, 0, 1], alpha=0.7,
           linestyles=['--', '-', '--'])                 #Draw boundary & margins

ax.scatter(model.support_vectors_[:, 0], model.support_vectors_[:, 1], s=100,
           linewidth=1, facecolors='none', edgecolors='green', label='Support Vectors')
plt.legend()
plt.show()


