##import pandas as pd
##import numpy as np
##from sklearn.linear_model import LogisticRegression
##from sklearn.model_selection import train_test_split
##from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, log_loss
##from sklearn.datasets import load_iris
##iris = load_iris()
##data = pd.DataFrame(data=iris.data, columns=iris.feature_names)
##data['target'] = iris.target
####print(iris.data)
##X = data.iloc[:, :-1].values
##Y = data.iloc[:, -1].values
##X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
##
##model = LogisticRegression(max_iter=1000)
##model.fit(X_train,Y_train)
##Y_pred = model.predict(X_test)
##Y_prob = model.predict_proba(X_test) 
##conf=confusion_matrix(Y_test,Y_pred)
##print("Confusion Matrix:")
##print(conf)
##acc = accuracy_score(Y_test, Y_pred)
##report = classification_report(Y_test, Y_pred, target_names=iris.target_names)
##
##print("\nAccuracy Score:", acc)
##print("\nClassification Report:")
##print(report)
##
##
##loss = log_loss(Y_test, Y_prob)
##print("\nLog Loss:", loss)


'''

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, log_loss
data = sns.load_dataset('titanic')
print(data.head())
data = data[['survived', 'pclass', 'sex', 'age', 'fare', 'alone']]
data['age'] = data['age'].fillna(data['age'].median())
data = pd.get_dummies(data, columns=['sex', 'alone'], drop_first=True)
X = data.drop('survived', axis=1)
Y = data['survived']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
model = LogisticRegression(max_iter=1000)
model.fit(X_train,Y_train)
Y_pred = model.predict(X_test)
Y_prob = model.predict_proba(X_test) 
conf=confusion_matrix(Y_test,Y_pred)
print("Confusion Matrix:")
print(conf)
acc = accuracy_score(Y_test, Y_pred)
report = classification_report(Y_test, Y_pred)
print("\nAccuracy Score:", acc)
print("\nClassification Report:")
print(report)
loss = log_loss(Y_test, Y_prob)
print("\nLog Loss:", loss)
'''


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error, confusion_matrix
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
data = sns.load_dataset("titanic")
print(data.head())
print(data.columns)
print(data.dtypes)

#data['age'].fillna(data['age'].mean(), inplace=True)    ##causes warning
##data['embarked'].fillna(data['embarked'].mode()[0], inplace=True)
data['age'] = data['age'].fillna(data['age'].mean())
data['embarked'] = data['embarked'].fillna(data['embarked'].mode()[0])

data.drop(
    ['deck', 'alive', 'class', 'who', 'adult_male', 'embark_town'],
    axis=1,
    inplace=True
)
le = LabelEncoder()
data['sex'] = le.fit_transform(data['sex'])
data['embarked'] = le.fit_transform(data['embarked'])
data['alone'] = le.fit_transform(data['alone'])

x = data.drop('survived', axis=1)
y = data['survived']
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2, random_state = 42)
model = LogisticRegression(max_iter=1000)
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))



