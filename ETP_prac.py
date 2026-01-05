#1 Develop a Random Forest model and analyze feature importance. Compare its performance with Decision Tree and KNN classifiers
'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2)
dt = DecisionTreeClassifier()
knn = KNeighborsClassifier(n_neighbors=5)
rf = RandomForestClassifier(n_estimators=100)

dt.fit(X_train, y_train)
knn.fit(X_train, y_train)
rf.fit(X_train, y_train)

print("Decision Tree:", accuracy_score(y_test, dt.predict(X_test)))
print("KNN:", accuracy_score(y_test, knn.predict(X_test)))
print("Random Forest:", accuracy_score(y_test, rf.predict(X_test)))
print(rf.feature_importances_)
'''

#2. Implement simple linear regression, multiple linear regression, and polynomial regression. Compare models using MAE, MSE, RMSE, and R². Analyze model complexity and overfitting.
'''
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
# Simple dataset
data = pd.DataFrame({
    'YearsExperience': [1,2,3,4,5,6,7,8,9,10],
    'Age': [22,23,24,25,26,27,28,29,30,31],
    'Salary': [30000,35000,40000,45000,50000,60000,65000,70000,75000,80000]
})

X_simple = data[['YearsExperience']]
X_multi = data[['YearsExperience', 'Age']]
y = data['Salary']
Xs_train, Xs_test, y_train, y_test = train_test_split(
    X_simple, y, test_size=0.2, random_state=42
)

Xm_train, Xm_test, _, _ = train_test_split(
    X_multi, y, test_size=0.2, random_state=42
)
slr = LinearRegression()
slr.fit(Xs_train, y_train)
y_pred_slr = slr.predict(Xs_test)
mlr = LinearRegression()
mlr.fit(Xm_train, y_train)
y_pred_mlr = mlr.predict(Xm_test)
poly = PolynomialFeatures(degree=2)
Xs_poly = poly.fit_transform(X_simple)

Xp_train, Xp_test, yp_train, yp_test = train_test_split(
    Xs_poly, y, test_size=0.2, random_state=42
)

pr = LinearRegression()
pr.fit(Xp_train, yp_train)
y_pred_pr = pr.predict(Xp_test)
def evaluate(y_true, y_pred):
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "MSE": mean_squared_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "R2": r2_score(y_true, y_pred)
    }

print("Simple Linear Regression:", evaluate(y_test, y_pred_slr))
print("Multiple Linear Regression:", evaluate(y_test, y_pred_mlr))
print("Polynomial Regression:", evaluate(yp_test, y_pred_pr))
'''




#3. Apply PCA to reduce dimensionality. Compare model performance before and after PCA.
'''
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
X, y = load_iris(return_X_y=True)

print("Original shape:", X.shape)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model_before = LogisticRegression(max_iter=1000)
model_before.fit(X_train, y_train)

y_pred_before = model_before.predict(X_test)
acc_before = accuracy_score(y_test, y_pred_before)

print("Accuracy before PCA:", acc_before)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
print("Shape after PCA:", X_pca.shape)
Xp_train, Xp_test, yp_train, yp_test = train_test_split(
    X_pca, y, test_size=0.2, random_state=42
)

model_after = LogisticRegression(max_iter=1000)
model_after.fit(Xp_train, yp_train)

y_pred_after = model_after.predict(Xp_test)
acc_after = accuracy_score(yp_test, y_pred_after)

print("Accuracy after PCA:", acc_after)
print("Accuracy Before PCA:", acc_before)
print("Accuracy After PCA:", acc_after)
'''



#4. Apply K-Means clustering. Determine optimal K using Elbow Method and visualize results
'''
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
X, _ = load_iris(return_X_y=True)
print(X.shape)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
wcss = []

for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

plt.plot(range(1, 11), wcss, marker='o')
plt.xlabel("Number of Clusters (K)")
plt.ylabel("WCSS")
plt.title("Elbow Method")
plt.show()
kmeans = KMeans(n_clusters=3, init='k-means++', random_state=42)
labels = kmeans.fit_predict(X_scaled)
plt.figure(figsize=(7,5))
plt.scatter(X_scaled[:,0], X_scaled[:,1], c=labels)
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("K-Means Clustering (K=3)")
plt.show()
'''


#5. Perform hierarchical clustering using different linkage methods. Plot dendrograms and analyze hierarchy.
'''
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
X, _ = load_iris(return_X_y=True)
print(X.shape)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
linked = linkage(X_scaled, method='ward')

plt.figure(figsize=(10,5))
dendrogram(linked)
plt.title("Dendrogram (Ward Linkage)")
plt.xlabel("Data Points")
plt.ylabel("Distance")
plt.show()
model = AgglomerativeClustering(
    n_clusters=3,
    linkage='ward'
)

labels = model.fit_predict(X_scaled)
linkage_methods = ['single', 'complete', 'average', 'ward']

for method in linkage_methods:
    linked = linkage(X_scaled, method=method)
    plt.figure(figsize=(6,3))
    dendrogram(linked)
    plt.title(f"Dendrogram ({method.capitalize()} Linkage)")
    plt.show()
'''

#6. Implement a boosting algorithm such as AdaBoost or Gradient Boosting and evaluate its performance.
'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score
X, y = load_iris(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

ada = AdaBoostClassifier(
    n_estimators=50,
    random_state=42
)

ada.fit(X_train, y_train)
y_pred_ada = ada.predict(X_test)

print("AdaBoost Accuracy:", accuracy_score(y_test, y_pred_ada))
gb = GradientBoostingClassifier(
    n_estimators=100,
    random_state=42
)

gb.fit(X_train, y_train)
y_pred_gb = gb.predict(X_test)

print("Gradient Boosting Accuracy:", accuracy_score(y_test, y_pred_gb))
'''


#7. Use association rule mining to extract patterns using support, confidence, and lift.

import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

data = pd.DataFrame({
    'Bread': [1,1,0,1,1],
    'Milk': [1,0,1,1,1],
    'Diaper': [0,1,1,1,0],
    'Beer': [0,1,1,1,0]
}).astype(bool)


frequent_itemsets = apriori(data, min_support=0.4, use_colnames=True)

rules = association_rules(
    frequent_itemsets,
    metric="lift",
    min_threshold=1
)

print(rules[['antecedents','consequents','support','confidence','lift']])


#8. Implement Train-Test Split, K-Fold Cross-Validation, and Leave-One-Out (LOO) Cross-Validation. Compare their reliability and computational cost.
'''
import pandas as pd
import numpy as np

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, KFold, LeaveOneOut, cross_val_score
from sklearn.linear_model import LogisticRegression
X, y = load_iris(return_X_y=True)
model = LogisticRegression(max_iter=1000)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model.fit(X_train, y_train)
accuracy = model.score(X_test, y_test)

print("Train-Test Accuracy:", accuracy)
kfold = KFold(n_splits=5, shuffle=True, random_state=42)

scores = cross_val_score(model, X, y, cv=kfold)

print("K-Fold Accuracies:", scores)
print("Mean Accuracy:", scores.mean())
loo = LeaveOneOut()

scores = cross_val_score(model, X, y, cv=loo)

print("LOO Mean Accuracy:", scores.mean())

'''


#9. Implement Logistic Regression and evaluate using Accuracy, Precision,
#Recall, F1-Score, Confusion Matrix, and ROC-AUC.
'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
data = sns.load_dataset("titanic")
# Handle missing values
data['age'] = data['age'].fillna(data['age'].mean())
data['embarked'] = data['embarked'].fillna(data['embarked'].mode()[0])
# Drop unnecessary columns
data.drop(['deck', 'alive', 'class', 'who', 'adult_male', 'embark_town'], axis=1, inplace=True)
# Encode categorical variables
le = LabelEncoder()
data['sex'] = le.fit_transform(data['sex'])
data['embarked'] = le.fit_transform(data['embarked'])
data['alone'] = le.fit_transform(data['alone'])
X = data.drop('survived', axis=1)
y = data['survived']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("ROC-AUC Score:", roc_auc_score(y_test, y_prob))
'''



#10. Choose a real-world application (healthcare, finance, retail, or education). Build a predictive model, evaluate its performance, and propose improvements based on analytical findings.
'''
import numpy as np
import pandas as pd

from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
data = load_diabetes()
X = data.data
y = (data.target > data.target.mean()).astype(int)  # Convert to binary

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))
'''
