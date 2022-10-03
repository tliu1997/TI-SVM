import numpy as np
import random
from scipy import ndimage

"""This script implements the functions for data augmentation
and preprocessing.
"""


def parse_record(record, training):
    """Parse a record to an image and perform data preprocessing."""
    depth_major = record.reshape((1, 64, 64))
    image = np.transpose(depth_major, [1, 2, 0])
    return image


def translation_preprocess(image):
    image = image.reshape((image.shape[0], 28, 28))
    # randomly put the object in 64*64 image
    image_pad = np.zeros((image.shape[0], 64, 64))
    for idx in range(image_pad.shape[0]):
        left = random.randint(0, 36)
        top = random.randint(0, 36)
        image_pad[idx] = np.pad(image[idx], ((top, 36-top), (left, 36-left)))
    # add Gaussian noise to the picture (mean=0, std=0.1)
    # gaus_noise = np.random.normal(0, 0.1, image_pad.shape)
    # image_pad = image_pad + gaus_noise

    return image_pad.reshape((image.shape[0], -1))


def rotation_preprocess(image):
    image = image.reshape((image.shape[0], 28, 28))
    # randomly put the object in 64*64 image
    image_pad = np.zeros((image.shape[0], 64, 64))
    for idx in range(image_pad.shape[0]):
        degree = random.randint(-180, 179)
        temp = ndimage.rotate(image[idx], degree, reshape=True)
        left = int((64 - temp.shape[0]) // 2)
        top = int((64 - temp.shape[1]) // 2)
        image_pad[idx] = np.pad(temp, ((top, 64-top-temp.shape[0]), (left, 64-left-temp.shape[1])))
    # add Gaussian noise to the picture (mean=0, std=0.1)
    # gaus_noise = np.random.normal(0, 0.1, image_pad.shape)
    # image_pad = image_pad + gaus_noise

    return image_pad.reshape((image.shape[0], -1))
