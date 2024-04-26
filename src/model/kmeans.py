import numpy as np
from scipy.spatial.distance import cdist
from scipy.sparse import csr_matrix
from src.data.preprocess import load_data, convert_to_zero_one_matrix
from src.helper.config import MODEL, DATA


class KMeans:
    def __init__(self, n_clusters=8, max_iter=300):
        self.n_clusters = n_clusters
        self.max_iter = max_iter

    @staticmethod
    def get_centroid(points):
        """
        This function computes the mean value of cluster points and transforms them into a 0/1
        matrix resembling a tweet vector that is the centroid to all the tweets in that cluster.
        :param points: tweets in a cluster
        :return: centroid tweet of the cluster
        """
        actual_mean = np.mean(points, axis=0)
        minimum, maximum = np.min(actual_mean), np.max(actual_mean)
        centroid = np.where(actual_mean >= (maximum + minimum) / 2, 1, 0)
        return centroid

    def fit(self, X):
        """
        This finds the clusters in X
        :param X: A 2D numpy array
        :return:
        """
        self.centroids = X[np.random.choice(X.shape[0], self.n_clusters, replace=False)]
        self.labels = np.zeros(X.shape[0])

        # making 2D array of X and centroid for dist calculation in cdist function
        X_arr = X.toarray()
        centroid_arr = self.centroids.toarray()

        for _ in range(self.max_iter):
            distances = cdist(X_arr, centroid_arr, metric='jaccard')
            new_labels = np.argmin(distances, axis=1)
            if np.all(new_labels == self.labels):
                break
            self.labels = new_labels

            for i in range(self.n_clusters):
                cluster_points = X_arr[self.labels == i]
                if len(cluster_points) > 0:
                    new_centroid = self.get_centroid(points=cluster_points)
                centroid_arr[i] = new_centroid

            # stop kmeans if centroids did not change from the previous iteration
            if np.array_equal(centroid_arr, self.centroids.toarray()):
                break

            # store the new centroids
            self.centroids = csr_matrix(centroid_arr)

            # initialize inertia to 0
            self.inertia_ = 0

            # calculating intertia (a.k.a within cluster sum of squares)
            for i in range(self.n_clusters):
                cluster_points = X_arr[self.labels == i]
                if len(cluster_points) > 0:
                    distances = cdist(cluster_points, [centroid_arr[i]], metric='jaccard')
                    self.inertia_ += np.sum(distances ** 2)

        return self

    def summary(self):
        print("K: ", self.n_clusters)
        print("SSE: ", self.inertia_)
        print("Size of each cluster:")
        # Get the unique values and their counts
        unique_values, counts = np.unique(self.labels, return_counts=True)
        # Print the results
        for value, count in zip(unique_values, counts):
            print(f"{value}: {count} tweets")

    def show_sample_from(self, cluster, tweets):
        print("Samples of tweets from cluster: ", cluster)
        print(tweets[self.labels == cluster].head())


if __name__ == "__main__":
    # load the data
    filename = DATA["file"]
    df = load_data(filename)

    # convert it to 0/1 matrix
    X = convert_to_zero_one_matrix(df)

    # create kmeans objects
    models = dict()
    for model in MODEL:
        models[model] = KMeans(n_clusters=model["k"], max_iter=model["max-iterations"])

    # run all of them
    for model in models:
        models[model].fit(X)
        models[model].summary()