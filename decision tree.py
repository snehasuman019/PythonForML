####from sklearn.svm import SVC
##from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn import tree
import matplotlib.pyplot as plt

iris = load_iris()

x = iris.data
y = iris.target

clf = DecisionTreeClassifier(criterion = 'entropy',max_depth=3,random_state=42)
clf.fit(x,y)
y_pred = clf.predict(x)
accuracy=metrics.accuracy_score(y, y_pred)
print(f"Accuracy of decision tree classifier: {accuracy:.2f}")
plt.figure(figsize=(12,8))
tree.plot_tree(clf,
               feature_names = iris.feature_names,
               class_names = iris.target_names, filled=True)
plt.show()
