from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from emnist import extract_training_samples
from emnist import extract_test_samples

"""This script implements the functions for reading data.
"""


def load_data():
    """Load the EMNIST Letters dataset"""
    x_train, y_train = extract_training_samples('letters')
    x_test, y_test = extract_test_samples('letters')
    x_train = (x_train / 255).reshape((-1, 784))
    x_test = (x_test / 255).reshape((-1, 784))

    return x_train, y_train, x_test, y_test


def train_valid_split(x_train, y_train):
    """Split the original training data into a new training dataset and a validation dataset."""
    x_train_new, x_valid, y_train_new, y_valid = train_test_split(x_train, y_train, train_size=5/6, random_state=4)
    return x_train_new, y_train_new, x_valid, y_valid
