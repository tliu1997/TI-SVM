from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.metrics import pairwise_distances
import numpy as np

"""This script implements the functions for tangent distance-based methods.
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


# SVM with tangent distance
class TDSVM(object):
    def __init__(self, metrics):
        self.metrics = metrics

    def TD(self, x, y):
        return np.exp(-pairwise_distances(x, y, metric=self.metrics)**2 / 784)

    def train(self, x, y):
        svclassifier = SVC(gamma=1, kernel=self.TD)
        svclassifier.fit(x, y)
        y_pred = svclassifier.predict(x)
        train_acc = accuracy_score(y, y_pred)
        return svclassifier, train_acc

    def evaluate(self, x, y, svclassifier):
        y_pred = svclassifier.predict(x)
        eval_acc = accuracy_score(y, y_pred)
        return eval_acc