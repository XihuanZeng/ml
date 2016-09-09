import numpy
import math
import urllib
import os
import cPickle
import gzip
from sklearn.cross_validation import train_test_split


class GeneralDataSet(object):
    def __init__(self, data_set, labels, onehot = False):
        """
        :param train_set: numpy array, matrix of predictors(no label)
        :param train_label: numpy array, dimension must agree with train_set
        :param onehot:
        :return:
        """

        assert data_set.shape[0] == labels.shape[0], (
              'date_set.shape: %s labels.shape: %s do not agree' % (data_set.shape, labels.shape))
        self._num_examples = data_set.shape[0]
        self._data_set = data_set
        self._labels = labels
        self._epochs_completed = 0
        self._index_in_epoch = 0

    @property
    def data_set(self):
        return self._data_set

    @property
    def labels(self):
        return self._labels

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def next_batch(self, batch_size, fake_data=False):
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._num_examples:
             # Finished epoch
             self._epochs_completed += 1
             # Shuffle the data, in this way every epoch our data in each batch is different
             perm = numpy.arange(self._num_examples)
             numpy.random.shuffle(perm)
             self._data_set = self._data_set[perm]
             self._labels = self._labels[perm]
             # Start next epoch
             start = 0
             self._index_in_epoch = batch_size
             assert batch_size <= self._num_examples
        end = self._index_in_epoch
        return self._data_set[start:end], self._labels[start:end]



def Gaussian_Density(x, mu, sigma):
    if sigma == 0:
        return 0
    else:
        const = 1 / (sigma * (math.sqrt(2 * math.pi)))
        exp = ((x - mu) ** 2) / (2 * sigma ** 2)
        result = const * math.exp(-exp)
        return result


def OpenData(data_name):
    assert data_name == 'MNIST' or 'Spam', "data name should be either 'MNIST' or 'Spam' "
    if data_name == 'MNIST':
        # download(one time) and open MNIST data
        url = 'http://deeplearning.net/data/mnist/mnist.pkl.gz'
        if not 'mnist.pkl.gz' in os.listdir('.'):
            urllib.urlretrieve(url, 'mnist.pkl.gz')
        f = gzip.open('mnist.pkl.gz', 'rb')
        train_set, valid_set, test_set = cPickle.load(f)
        f.close()
        return train_set, test_set
    if data_name == 'Spam':
        # download(one time) and save Spam data
        url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/spambase/spambase.data'
        if not 'spam.csv' in os.listdir('.'):
            urllib.urlretrieve(url, 'spam.csv')
        data = []
        with open('spam.csv', 'rb') as f:
            for line in f:
                fields=line.split(',')
                row_data=[float(x) for x in fields]
                data.append(row_data)
            data = numpy.array(data)
        f.close()
        X_train, X_test, y_train, y_test = train_test_split(data[:,1:], data[:,-1], test_size = 0.16)
        return (numpy.concatenate([X_train, y_train.reshape(len(y_train),1)], axis = 1),
                numpy.concatenate([X_test, y_test.reshape(len(y_test),1)], axis = 1))
