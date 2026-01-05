import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, log_loss, roc_auc_score
from sklearn.preprocessing import LabelEncoder,OneHotEncoder,StandardScaler
data=pd.read_csv(("E:\\D drive\\SEM 5\\234\\customer_churn.csv"),encoding = "Latin 1")
data
data.isna().sum()
data['InternetService']=data['InternetService'].fillna(data['InternetService'].mode()[0])
data.isna().sum()
le=LabelEncoder()
data['Gender']=le.fit_transform(data['Gender'])
categorial_cols=['ContractType','PaymentMethod','InternetService']
data = pd.get_dummies(data,columns=categorial_cols,drop_first=True,dtype=int)
data 
x = data.drop("Churn", axis=1)
y = data["Churn"]

sc = StandardScaler()
x_scaled = sc.fit_transform(x)

x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, test_size=0.3, random_state=2)

model = LogisticRegression()
model.fit(x_train, y_train)

y_pred = model.predict(x_test)
y_pred_p = model.predict_proba(x_test)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
ll = log_loss(y_test, y_pred_p)
roc = roc_auc_score(y_test, y_pred_p)

print("Precision:", precision)
print("Recall:", recall)
print("Accuracy:", accuracy)
print("Log Loss:", ll)
print("AUC ROC:", roc)
