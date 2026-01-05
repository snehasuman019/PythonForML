##from sklearn.cluster import KMeans
##import numpy as np
##import pandas as pd
##from sklearn.metrics import silhouette_score
##import matplotlib.pyplot as plt
##from sklearn.datasets import load_iris
##from sklearn.preprocessing import StandardScaler
##
##iris = load_iris()
##
##
######x=np.array([[1], [2], [3], [8], [9], [10]])
####x=np.array([[1,2], [1,4], [1,0], [10,2], [10,4], [10,0]])
##
##
##kmeans = KMeans(n_clusters=2, random_state=42)
##kmeans.fit(x)
##
####
####print("Cluster centers: ",KMeans.cluster_centers_)
####print("Labels:",KMeans.labels_)
####
####inertia = KMeans.inertia_
#####quality of cluster seperation
####sil_score = silhouette_score(x, KMeans.labels_)
####
####print("Inertia: ",inertia)
####print("Silhouette_Score: ", sil_score)
####
####centers = KMeans.cluster_centers_
####cluster_sizes = np.bincount(KMeans.labels_)
####print("centers",centers)
##
##wcss = []
##for k in range(1,6):
##    km = KMeans(n_clusters=k)
##    km.fit(x)
##    wcss.append(km.inertia_)
##print("*****",wcss)
##plt.plot(range(1,6),wcss, marker='o')
##plt.xlabel("Number of clusters (k)")
##plt.ylabel("WCSSS")
##plt.title("Elbow Method for choosing K")
##plt.show()



##
##from sklearn.cluster import KMeans
##import numpy as np
##import pandas as pd
##from sklearn.metrics import silhouette_score
##import matplotlib.pyplot as plt
##from sklearn.datasets import load_iris
##from sklearn.preprocessing import StandardScaler
##
##iris = load_iris()
##x=iris.data
##y=iris.target
##
##scaler = StandardScaler()
##x_scaled = scaler.fit_transform(x)
##
##kmeans = KMeans(n_clusters=3, random_state=42)
##kmeans.fit(x_scaled)
##
##
##print("Cluster centers: ",KMeans.cluster_centers_)
##print("Labels:",KMeans.labels_)
##
##inertia = KMeans.inertia_
##print("Inertia: ",inertia)
##
##sil_score = silhouette_score(x_scaled, KMeans.labels_)
##print("Silhouette_Score: ", sil_score)
##
##wcss = []
##for k in range(1,7):
##    kmeans = KMeans(n_clusters=k, random_state=42)
##    kmeans.fit(x)
##    wcss.append(km.inertia_)
##print("WCSS Values: ",wcss)
##
##plt.plot(range(1,7),wcss, marker='o')
##plt.xlabel("Number of clusters (k)")
##plt.ylabel("WCSSS")
##plt.title("Elbow Method for choosing K")
##plt.show()


from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler

iris = load_iris()
x = iris.data
y = iris.target

scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(x_scaled)

print("Cluster centers:\n", kmeans.cluster_centers_)
print("Labels:\n", kmeans.labels_)
print("Inertia:", kmeans.inertia_)

sil_score = silhouette_score(x_scaled, kmeans.labels_)
print("Silhouette Score:", sil_score)

wcss = []
for k in range(1, 7):
    km = KMeans(n_clusters=k, random_state=42)
    km.fit(x_scaled)
    wcss.append(km.inertia_)

print("WCSS Values:", wcss)

plt.plot(range(1, 7), wcss, marker='o')
plt.xlabel("Number of clusters (k)")
plt.ylabel("WCSS")
plt.title("Elbow Method for choosing K")
plt.show()


