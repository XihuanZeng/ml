import numpy
from tensorflow.python.framework import dtypes


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