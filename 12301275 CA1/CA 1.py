'''

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

'''

#Question 2



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, confusion_matrix, classification_report

data = pd.read_csv(("E:\\D drive\\SEM 5\\234\\customer_churn.csv"),encoding = "Latin 1")
##print(data.head)

data=data.replace(" ",np.nan)
data=data.dropna()

le=LabelEncoder()
x=data.drop("Churn",axis=1)
y=data["Churn"]

sc=StandardScaler()
x=sc.fit_transform(x)
x_train, x_test, y_train , y_test = train_test_split(x, y, test_size=0.2, random_state=42)

plt.figure(figsize=(5,4))
df["Churn"].value_counts().plot(kind="bar")
plt.title("Churn Distribution")
plt.show()

plt.figure(figsize=(6,5))
sns.countplot(x=df["Contract"],hue=df["Churn"])
plt.title("Churn vs Contract Type")
plt.show()

plt.figure(figsize=(6,5))
sns.countplot(x=df["PaymentMethod"],hue=df["Churn"])
plt.title("Churn vs Payment Method")
plt.xticks(rotation=45)
plt.show()

plt.figure(figsize=(10,8))
sns.heatmap(df.corr(),annot=False,cmap="viridis")
plt.title("Correlation Heatmap")
plt.show()

model=LogisticRegression(max_iter=1000)
model.fit(y_train.reshape(-1,1),y_train)
model =LogisticRegression(max_iter=1000)
model.fit(x_train, y_train)

y_pred=model.predict(x_test)
y_pred_prob=model.predict_proba(x_test)

acc=accuracy_score(y_test,y_pred)
prec=precision_score(y_test,y_pred)
rec=recall_score(y_test,y_pred)
auc=roc_auc_score(y_test,y_prob)


print("Accuracy:",acc)
print("Precision:",prec)
print("Recall:",rec)
print("AUC:",auc)
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
