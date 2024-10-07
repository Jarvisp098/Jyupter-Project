import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

np.random.seed(40)
X = np.random.rand(100, 2)

#create two clusters
X[:50] +=1
X[50:] +=2

#Apply Kmeans
Kmeans= KMeans(n_clusters=2)
Kmeans.fit(X)

#Get cluster labels and centroids
labels = Kmeans.labels_
centroids = Kmeans.cluster_centers_

#Plot the result
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], marker= 'o', c=labels, cmap='viridis', edgecolors='k', s =50)
plt.scatter(centroids[:, 0], centroids[:, 1], marker='x', s=200, c='red', label='Centroids')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('K-Means Clustering')
plt.legend()
plt.grid()
plt.show()

