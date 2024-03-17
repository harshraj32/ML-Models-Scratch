import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


class KMeans:

    def __init__(self, in_clusters=2, seed=23, kmeans_plus_plus = False):
        self.in_clusters = in_clusters
        self.seed = seed
        self.kmeans_plus_plus = kmeans_plus_plus
        np.random.seed(seed)

    

    def initialize_clusters(self, X):
        centroids_ids = np.random.choice(X.shape[0], size=self.in_clusters, replace=False)
        centroids = []
        for i in centroids_ids:
          centroids.append(X.iloc[i].values)
        return centroids
    
    def intialize_clusters_k_plus_plus(self, X):

        centroids = []
        #randomly choosing the first centroid
        centroid_id = np.random.choice(X.shape[0])
        centroids.append(X.iloc[centroid_id].values)

        for _ in range(1, self.in_clusters):
            distances = np.array([min(np.linalg.norm(x - c) ** 2 for c in centroids) for x in X.values])
            # Choose the next centroid with probability proportional to its distance squared
            new_centroid = X.iloc[np.random.choice(X.shape[0], p=distances / distances.sum())].values
            centroids.append(new_centroid)

        return centroids


    def calc_distances(self, X, centroids):
        labels = np.zeros(len(X), dtype=int)
        for i, x in X.iterrows():
            min_dist = float('inf')
            for j, centroid in enumerate(centroids):
                dist = np.sqrt(np.sum((x - centroid)**2))
                if dist < min_dist:
                    min_dist = dist
                    labels[i] = j
        return labels

    def update_centroids(self, X, labels):
        centroids = np.zeros((self.in_clusters, X.shape[1]))
        for k in range(self.in_clusters):
            cluster_points = X[labels == k]
            if len(cluster_points) > 0:
                centroids[k] = np.mean(cluster_points, axis=0)
        return centroids

    def fit(self, X, max_iters=100):
        centroids = self.initialize_clusters(X)
        for i in range(max_iters):
            labels = self.calc_distances(X, centroids)
            new_centroids = self.update_centroids(X, labels)
            #if i == 99:

              #print("reached max iteration")
            #else:
              
              #print("num of iters ",i)
              #print(centroids)
              #print(new_centroids)
            if np.all(centroids == new_centroids):
                break
            centroids = new_centroids
        return centroids, labels


    def elbow_method(self, X, max_clusters=10):
        labels_all = {}
        cost_function_values = []
        for k in range(1, max_clusters + 1):
            if self.kmeans_plus_plus:
              kmeans = KMeans(in_clusters=k, kmeans_plus_plus=True)
            else:
              kmeans = KMeans(in_clusters=k)
            
            centroids, labels = kmeans.fit(X)
            labels_all[k] = labels
            cost_function = 0
            for i, x in X.iterrows():
                cost_function += np.sqrt(np.sum((x - centroids[labels[i]]) ** 2))
            cost_function_values.append(cost_function)
        return cost_function_values, labels_all



df = pd.read_csv('/content/cluster_data1.csv')
df.shape

start_time = time.time()
kmeans = KMeans()
cost_function_values , labels_all = kmeans.elbow_method(df)
end_time = time.time()
print("kmeans time taken: ", end_time - start_time)
plt.plot(range(1, len(cost_function_values) + 1), cost_function_values)
plt.xlabel('Number of Clusters')
plt.ylabel('Cost function')
plt.title('Elbow Method')
plt.show()

#print(centroids)
#print(labels)

#print(df.shape)
