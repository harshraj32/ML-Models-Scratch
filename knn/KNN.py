import math
from collections import Counter

import numpy as np


def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))


def manhattan_distance(x1, x2):
    return np.sqrt(np.sum(math.abs(x1 - x2)))


class KNN:

    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        predictions = [self._predict(x) for x in X]
        return predictions

    def _predict(self, x):

        # compute the distance
        # distances = [euclidean_distance(x, x_train) for x_train in self.X_train]
        distances = np.sqrt(np.sum((self.X_train - x) ** 2, axis=1))
        # get the closest k neighbors
        k_ind = np.argsort(distances)[: self.k]
        k_labels = [self.y_train[i] for i in k_ind]

        # get votes
        most_common = Counter(k_labels).most_common(1)
        return most_common[0][0]
