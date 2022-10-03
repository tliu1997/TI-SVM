from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

"""This script implements the tangent distance-based methods.
"""


# kNN with tangent distance
class KNN(object):
    def __init__(self, metrics):
        self.metrics = metrics

    def train(self, x, y):
        neigh = KNeighborsClassifier(n_neighbors=3, metric=self.metrics)
        neigh.fit(x, y)
        y_pred = neigh.predict(x)
        train_acc = accuracy_score(y, y_pred)
        return neigh, train_acc

    def evaluate(self, x, y, neigh):
        y_pred = neigh.predict(x)
        eval_acc = accuracy_score(y, y_pred)
        return eval_acc