import os
import random

import h5py
import numpy as np
import scipy

random.seed(0)
np.random.seed(0)


def build_dataset(type, degradation_type):
    if type == 'dem':
        path_train = './data/dem/' + degradation_type + '/Train/'
        path_test = './data/dem/' + degradation_type + '/Test/'
        file_train, file_test = sorted(os.listdir(path_train)), sorted(os.listdir(path_test))
        x_train, y_train, x_test, y_test = [], [], [], []
        for i, j in zip(file_train, file_test):
            y = scipy.io.loadmat(path_train + i)['x']
            x = scipy.io.loadmat(path_train + i)['y']
            x_train.append(x)
            y_train.append(y)
            y = scipy.io.loadmat(path_test + i)['x']
            x = scipy.io.loadmat(path_test + i)['y']
            x_test.append(x)
            y_test.append(y)

        x_train = np.concatenate(x_train, axis=0)
        x_test = np.concatenate(x_test, axis=0)
        y_train = np.concatenate(y_train, axis=0)
        y_test = np.concatenate(y_test, axis=0)

        xmax, xmin = np.max(x_test, axis=(1, 2), keepdims=True), np.min(x_test, axis=(1, 2), keepdims=True)

        x, y = [], []
        for i, j in zip(x_train, y_train):
            if (np.max(i) - np.min(i)) == 0.0:
                x.append(np.zeros_like(i))
                y.append(np.zeros_like(j))
            else:
                y.append((j - np.min(i)) / (np.max(i) - np.min(i)))
                x.append((i - np.min(i)) / (np.max(i) - np.min(i)))

        x_train = np.array(x)
        y_train = np.array(y)

        x, y = [], []
        for i, j in zip(x_test, y_test):
            if (np.max(i) - np.min(i)) == 0.0:
                x.append(np.zeros_like(i))
                y.append(np.zeros_like(j))
            else:
                y.append((j - np.min(i)) / (np.max(i) - np.min(i)))
                x.append((i - np.min(i)) / (np.max(i) - np.min(i)))

        x_test = np.array(x)
        y_test = np.array(y)

    elif type == 'seismic':
        path = './data/seismic/'
        y_train = scipy.io.loadmat(path + degradation_type + '.mat')['x']
        x_train = scipy.io.loadmat(path + degradation_type + '.mat')['y']

        x_test = x_train[2560:, :, :]
        x_train = x_train[0:2560, :, :]
        y_test = y_train[2560:, :, :]
        y_train = y_train[0:2560, :, :]

        y_train = (y_train - np.min(x_train, axis=(1, 2), keepdims=True)) / (np.max(x_train, axis=(1, 2), keepdims=True) - np.min(x_train, axis=(1, 2), keepdims=True))
        x_train = (x_train - np.min(x_train, axis=(1, 2), keepdims=True)) / (np.max(x_train, axis=(1, 2), keepdims=True) - np.min(x_train, axis=(1, 2), keepdims=True))

        xmax, xmin = np.max(x_test, axis=(1, 2), keepdims=True), np.min(x_test, axis=(1, 2), keepdims=True)

        y_test = (y_test - np.min(x_test, axis=(1, 2), keepdims=True)) / (np.max(x_test, axis=(1, 2), keepdims=True) - np.min(x_test, axis=(1, 2), keepdims=True))
        x_test = (x_test - np.min(x_test, axis=(1, 2), keepdims=True)) / (np.max(x_test, axis=(1, 2), keepdims=True) - np.min(x_test, axis=(1, 2), keepdims=True))

    elif type == 'rock':
        path = './data/rock/Train/'

        y_train = scipy.io.loadmat(path + degradation_type + '.mat')['x']
        x_train = scipy.io.loadmat(path + degradation_type + '.mat')['y']

        shuffle_ix = np.random.permutation(np.arange(y_train.shape[0]))
        y_train = y_train[shuffle_ix, :, :]
        x_train = x_train[shuffle_ix, :, :]

        path = './data/rock/Test/'
        y_test = scipy.io.loadmat(path + degradation_type + '.mat')['x']
        x_test = scipy.io.loadmat(path + degradation_type + '.mat')['y']

        xmax, xmin = 1.0, 0

    return x_train, y_train, x_test, y_test, xmax, xmin