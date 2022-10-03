from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

"""This script implements the functions for reading data.
"""


def load_data():
    """Load the MNIST dataset"""
    mnist = fetch_openml('mnist_784')
    x = mnist.data.astype('float32')
    y = mnist.target.astype('int64')
    x = x / 255
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=6/7, random_state=4)

    return x_train, y_train, x_test, y_test


def train_valid_split(x_train, y_train):
    """Split the original training data into a new training dataset and a validation dataset."""
    x_train_new, x_valid, y_train_new, y_valid = train_test_split(x_train, y_train, train_size=5/6, random_state=4)
    return x_train_new, y_train_new, x_valid, y_valid

