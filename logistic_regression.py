# implement a regularized(L1 or L2) multi-class logistic regression(optimized by stochastic gradient descent) with TensorFlow
# you can specify the percentage of hold out set from training set for validation purpose

import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from utils import GeneralDataSet


class logistic_regression:
    def __init__(self, regulizer = None, beta = 0.5, batchsize = 128, epoch = 10):
        self.__batchsize = 128
        self.__regulizer = regulizer
        self.__epoch = epoch
        if regulizer:
            self.__beta = beta
        else:
            self.__beta = 0
        self.__scaler = StandardScaler()
        self.__onehot = OneHotEncoder()

    def fit(self, train_set, train_label, num_labels):
        """
        :param train_set: numpy array, shape is (num_sample, num_feature)
        :param train_label: numpy array, shape is (num_sample,)
        :return:
        """
        # standard scaler and one hot encoding for the label
        self.__scaler.fit(train_set)
        self.__onehot.fit([[i] for i in train_label])

        # create a batch generator
        model_input = GeneralDataSet(self.__scaler.transform(train_set),
                             np.array(self.__onehot.transform([[i] for i in train_label]).toarray(), dtype = 'float32'))

        # building computation graph
        num_train_samples = train_set.shape[0]
        self.__num_feature = train_set.shape[1]
        self.__num_labels = num_labels
        x = tf.placeholder(tf.float32, [None, self.__num_feature])
        W = tf.Variable(tf.zeros([self.__num_feature, self.__num_labels]))
        b = tf.Variable(tf.zeros([self.__num_labels]))
        y = tf.nn.softmax(tf.matmul(x, W) + b)
        y_truth = tf.placeholder(tf.float32, [None, self.__num_labels])
        cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_truth * tf.log(y), reduction_indices=[1]))
        if self.__regulizer == 'l1':
            loss = cross_entropy + self.__beta * tf.reduce_sum(W)
        elif self.__regulizer == 'l2':
            loss = cross_entropy + self.__beta *tf.nn.l2_loss(W)
        else:
            loss = cross_entropy
        optimizer = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

        # start a session to train in batch
        init = tf.initialize_all_variables()
        sess = tf.Session()
        sess.run(init)
        for i in range(self.__epoch * int(num_train_samples / self.__batchsize)):
            batch_xs, batch_ys = model_input.next_batch(self.__batchsize)
            sess.run(optimizer, feed_dict={x: batch_xs, y_truth: batch_ys})

        # fetch the trained model parameters W and b
        self.__W = sess.run(W)
        self.__b = sess.run(b)
        sess.close()

    def get_weight(self):
        return self.__W

    def get_bias(self):
        return self.__b

    def get_params(self):
        return {'regulizer': self.__regulizer,
                'beta': self.__beta,
                'batch size': self.__batchsize,
                'epoch': self.__epoch}

    def predict(self, test_set):
        """
        :param test_set: numpy array, shape is (num_test_sample, num_feature)
        :return:
        """
        x = tf.placeholder(tf.float32, [None, self.__num_feature])
        W = tf.constant(self.__W)
        b = tf.constant(self.__b)
        y = tf.nn.softmax(tf.matmul(x, W) + b)
        sess = tf.Session()
        predict = sess.run(tf.argmax(y, 1), feed_dict = {x:self.__scaler.transform(test_set)})
        sess.close()
        return predict


