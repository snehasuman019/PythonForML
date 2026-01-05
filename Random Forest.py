import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score

data = pd.read_csv(("E:\\D drive\\SEM 5\\234\\Social_Network_Ads.csv"),encoding = "Latin 1")
print(data.info())
print(data.describe())
print(data.head())

x=data[['Age','EstimatedSalary']]
y=data['Purchased']
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.25, random_state = 42)
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

rf = RandomForestClassifier(
    n_estimators=100, random_state=42)
rf.fit(x_train_scaled, y_train)
y_pred = rf.predict(x_test_scaled)

cm = confusion_matrix(y_test, y_pred)
acc = accuracy_score(y_test, y_pred)

print("\nconfusion_matrix: ")
print(cm)
print("\nAccuracy: ",acc)








