import time

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

from KNN import KNN

cmap = ListedColormap(["#FF0000", "#00FF00", "#0000FF"])

iris = datasets.load_iris()

X, y = iris.data, iris.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.6, random_state=23
)

# plt.figure()
# plt.scatter(X[:, 2], X[:, 3], c=y, cmap=cmap, edgecolors="k", s=20)
# plt.show()

# custom model time
start_time = time.time()
clf = KNN(k=5)
clf.fit(X_train, y_train)
predictions = clf.predict(X_test)
acc = np.sum(predictions == y_test) / len(y_test)
end_time = time.time()
print("knn accuracy: ", acc, "with time taken: ", end_time - start_time)

# sklearn
start_time = time.time()
clf1 = KNeighborsClassifier(n_neighbors=5)
# custom model time
clf1.fit(X_train, y_train)
predictions = clf1.predict(X_test)
sk_acc = np.sum(predictions == y_test) / len(y_test)
end_time = time.time()

# sklearn model time
print("sklearn knn accuracy", sk_acc, "with time taken: ", end_time - start_time)
