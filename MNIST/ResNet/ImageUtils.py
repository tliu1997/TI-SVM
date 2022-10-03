import numpy as np

"""This script implements the functions for data augmentation
and preprocessing.
"""


def parse_record(record, training):
    """Parse a record to an image and perform data preprocessing."""
    depth_major = record.reshape((1, 28, 28))
    image = np.transpose(depth_major, [1, 2, 0])

    return image

