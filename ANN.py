import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, accuracy_score

#
data=pd.read_csv(("E:\\D drive\\SEM 5\\234\\Social_Network_Ads.csv"),encoding = "Latin 1")

x=data[['Age','EstimatedSalary']]
y = data ['Purchased']

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.25, random_state = 42)
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

ann = MLPClassifier(
    hidden_layer_sizes = (4,),
    activation ='relu',
    solver='adam',
    max_iter = 500,
    random_state = 42
)
ann.fit(x_train, y_train)

y_pred = ann.predict(x_test)

cm=confusion_matrix(y_test, y_pred)
acc=accuracy_score(y_test, y_pred)
print("Confusion matrix: ")
print(cm)

print("\nAccuracy: ",acc)
