##from sklearn.cluster import AgglomerativeClustering
##import matplotlib.pyplot as plt
##from scipy.cluster.hierarchy import dendrogram, linkage
##import numpy as np
##
##X= np.array([[1,2], [2,1], [8,9], [9,8], [8,8], [2,2]])
##Z=linkage(X, method='ward')
##
##plt.figure(figsize=(6,4))
##dendrogram(Z)
##plt.title("Dendrogram (Ward Linkage)")
##plt.xlabel("Data Points")
##plt.ylabel("Distance")
##plt.tight_layout()
##plt.show()
##model = AgglomerativeClustering(n_clusters=2, linkage='ward')
##labels=model.fit_predict(X)
##print(labels)




#AgglomerativeClustering on iris dataset

##from sklearn.cluster import AgglomerativeClustering
##import matplotlib.pyplot as plt
##from scipy.cluster.hierarchy import dendrogram, linkage
##import numpy as np
##from sklearn.datasets import load_iris
##from sklearn.preprocessing import StandardScaler
##iris = load_iris()
##X = iris.data
##
##scaler = StandardScaler()
##X_scaled = scaler.fit_transform(X)
##Z = linkage(X_scaled, method='ward')
##
##
##plt.figure(figsize=(10, 5))
##dendrogram(Z)
##plt.title("Dendrogram (Ward Linkage) - Iris Dataset")
##plt.xlabel("Data Points")
##plt.ylabel("Euclidean Distance")
##plt.tight_layout()
##plt.show()
##
##model = AgglomerativeClustering(n_clusters=3, linkage='ward')
##labels = model.fit_predict(X_scaled)
##print("Cluster labels:\n", labels)
##



###DivisiveClustering on iris dataset
##from sklearn.cluster import KMeans
##import matplotlib.pyplot as plt
##import numpy as np
##from sklearn.datasets import load_iris
##from sklearn.preprocessing import StandardScaler
##
### Load Iris dataset
##iris = load_iris()
##X = iris.data
##
### Scale features
##scaler = StandardScaler()
##X_scaled = scaler.fit_transform(X)
##
### -----------------------------
### Divisive Clustering Function
### -----------------------------
##def divisive_clustering(X, k):
##    clusters = [X]
##
##    while len(clusters) < k:
##        # Select cluster with maximum variance
##        variances = [np.var(cluster) for cluster in clusters]
##        idx = np.argmax(variances)
##
##        cluster_to_split = clusters.pop(idx)
##
##        # Split cluster using KMeans (k=2)
##        km = KMeans(n_clusters=2, random_state=42)
##        labels = km.fit_predict(cluster_to_split)
##
##        clusters.append(cluster_to_split[labels == 0])
##        clusters.append(cluster_to_split[labels == 1])
##
##    return clusters
##
### Apply divisive clustering (k = 3)
##clusters = divisive_clustering(X_scaled, k=3)
##
### Create label array
##labels = np.zeros(len(X_scaled), dtype=int)
##start = 0
##for i, cluster in enumerate(clusters):
##    size = len(cluster)
##    labels[start:start + size] = i
##    start += size
##
##print("Cluster sizes:", [len(c) for c in clusters])
##print("Cluster labels:\n", labels)
##
### -----------------------------
### Visualization (2D)
### -----------------------------
##plt.figure(figsize=(6, 4))
##plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=labels)
##plt.xlabel("Feature 1 (scaled)")
##plt.ylabel("Feature 2 (scaled)")
##plt.title("Divisive (Top-Down) Clustering - Iris Dataset")
##plt.show()




from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np

# Original dataset
X = np.array([[1,2], [2,1], [8,9], [9,8], [8,8], [2,2]])

# -----------------------------
# Divisive Clustering Function
# -----------------------------
def divisive_clustering(X, k):
    clusters = [X]   # start with one big cluster

    while len(clusters) < k:
        # choose cluster with maximum variance
        variances = [np.var(cluster) for cluster in clusters]
        idx = np.argmax(variances)

        cluster_to_split = clusters.pop(idx)

        # split into 2 using KMeans
        kmeans = KMeans(n_clusters=2, random_state=42)
        labels = kmeans.fit_predict(cluster_to_split)

        clusters.append(cluster_to_split[labels == 0])
        clusters.append(cluster_to_split[labels == 1])

    return clusters

# Apply divisive clustering (k = 2)
clusters = divisive_clustering(X, k=2)

# Create final labels
labels = np.zeros(len(X), dtype=int)
index = 0
for i, cluster in enumerate(clusters):
    for _ in cluster:
        labels[index] = i
        index += 1

print("Cluster labels:", labels)

# -----------------------------
# Visualization
# -----------------------------
plt.scatter(X[:, 0], X[:, 1], c=labels)
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.title("Divisive (Top-Down) Clustering")
plt.show()

