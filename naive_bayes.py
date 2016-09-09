# implement multi-class Gaussian Naive Bayes classifier
# labels must be 0, 1, 2,..., num_labels-1

import numpy as np
import operator as op
import random
import math
import sys
from utils import Gaussian_Density, OpenData
from __future__ import division

class naive_bayes(object):
    def __init__(self):
        pass

    def fit(self, train_set, train_labels, num_labels):
        """
        :param train_set: numpy array of shape (num_samples, num_features)
        :param train_labels: numpy array of shape (num_samples, )
        :param num_labels: 2 for binary, n for n-class classification
        :return:
        """
        self.__num_features = train_set.shape[1]
        self.__num_train_samples = train_set.shape[0]
        self.__num_labels = num_labels
        self.__prior = [0] * self.__num_labels
        self.__mean = np.zeros((self.__num_labels, self.__num_features))
        self.__sd = np.zeros((self.__num_labels, self.__num_features))
        for i in range(num_labels):
            subset_index = np.where(train_labels == i)
            subset = train_set[subset_index]
            self.__mean[i] = subset.mean(axis = 0)
            self.__sd[i] = subset.std(axis = 0)
            self.__prior[i] = len(subset) / self.__num_train_samples

    def predict(self, test_set):
        result = [0] * len(test_set)
        for i in range(len(test_set)):
            posterior_arr = np.zeros((self.__num_labels, self.__num_features))
            for j in range(self.__num_labels):
                for k in range(self.__num_features):
                    posterior_arr[j, k] = Gaussian_Density(test_set[i, k], mu = self.__mean[j, k],
                                                           sigma = self.__sd[j, k])
            posterior_list = map(lambda x: reduce(op.mul, x), posterior_arr)
            posterior_list = [posterior_list[m] * self.__prior[m] for m in range(self.__num_labels)]
            result[i] = np.argmax(posterior_list)
        return result

    def get_mean(self):
        return self.__mean

    def get_sd(self):
        return self.__sd

    def get_prior(self):
        return self.__prior

