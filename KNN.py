from collections import Counter
import numpy as np

class KNN:
    def __init__(self, k):
        self.k = k

    def fit(self, X, y):
        self.X_train  = X
        self.y_train = y

    def predict(self, X):
        y_pred = [self.predict_individual(x) for x in X]
        return np.array(y_pred)

    def predict_individual(self, x):

        # We need to compute the distance between x and all other points in training set
        dist = [euclidean_dist(x, x_train) for x_train in self.X_train]
        # Sort all the distances in ascending order to return k nearest points.
        k_index = np.argsort(dist)[:self.k]
        # Get the labels of k nearest points
        k_labels = [self.y_train[x] for x in k_index]
        # Return the most frequent label
        label = Counter(k_labels).most_common(1)
        return label[0][0]

    def euclidean_dist(self, x, y):
        return np.sqrt(np.sum((x - y) ** 2))

if __name__ == '__main__':
    classifier = KNN(X_train, y_train)
    preds = classifier.predict(X_test)
    print("custom KNN classification accuracy", accuracy(y_test, predictions))


    def accuracy(y_true, y_pred):
        acc = np.sum(y_pred == y_true)/len(y_true)
        return acc



