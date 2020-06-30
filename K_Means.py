import numpy as np
from sklearn.datasets import make_blobs

class K_means:
    def __init__(self, X, num_clusters):
        self.K = num_clusters
        self.max_itrs = 100
        self.num_examples, self.num_features = X.shape

    def initialize_random_centroids(self, X):
        centroids = np.zeros((self.K, self.num_features))   # create centroids containing 0s of size k and features
        for k in range(self.K):
            # Select random centroid points from entire list of examples
            centroid = X[np.random.choice(range(self.num_examples))]
            centroids[k] = centroid

            return centroids

    # We now have random centroids, next we need to create clusters
    def create_clusters(self, centroids, X):

        # Every cluster will have several points belonging to each cluster.

        clusters = [[] for x in range(self.K)]
         # For every point check which cluster is closest to it based on euclidean distance
        for point_idx, point in enumerate(X):
            closest_centroid = np.argmin(np.sqrt(np.sum(point - centroids) ** 2, axis = 1))  # Take min euclidean distance
            clusters[closest_centroid].append(point_idx)

        return clusters

    # We now must update centroids to converge our algorithm

    def calculate_new_centroids(self, clusters, X):
        centroids = np.zeros((self.K, self.num_features))

        for idx, cluster in enumerate(clusters):
            new_centroid = np.mean(X[cluster], axis = 0)
            centroids[idx] = new_centroid

        return centroids

    def predict_cluster(self, clusters, X):
        y_pred = np.zeros(self.num_examples)

        for cluster_idx, cluster in enumerate(clusters):
            for sample_idx in cluster:
                y_pred[sample_idx] = cluster_idx

        return y_pred

    # The below method will be used to train our model

    def fit(self, X):
        centroids = self.initialize_random_centroids(X)

        for x in range(self.max_itrs):
            clusters = self.create_clusters(centroids, X)
            previous_centroid = centroids
            centroids = self.calculate_new_centroids(clusters, X)
            # Till this part we have assigned random centroids, created clusters and adjusted centroids to get new ones
            # Next we should have a condition to terminate the loop if the model converges

            diff = centroids - previous_centroid
            if not diff.any():
                print("Termination Criteria reached.")
                break

        y_pred = self.predict_cluster(clusters, X)

if __name__ == '__main__':
    np.random.seed(10)
    num_clusters = 5
    X, _ = make_blobs(n_samples=1000, n_features=2, centers=num_clusters)
    k_means_clustering = K_means(X, num_clusters)
    y_pred = K_means.fit(X)









