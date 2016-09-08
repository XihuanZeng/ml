# implement a regularized(L1 or L2) multi-class logistic regression(optimized by stochastic gradient descent) with TensorFlow
# you can specify the percentage of hold out set from training set for validation purpose

import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from tensorflow.examples.tutorials.mnist import input_data

class logistic_regression:
    def __init__(self, regulizer = None, beta = 0.5, batchsize = 128, epoch = 30):
        self.batchsize = 128
        self.regulizer = regulizer
        if regulizer:
            self.beta = beta
        else:
            self.beta = 0

    def fit(self, train_set, train_label):
        # standard scaler
        self.scaler = StandardScaler()
        self.scaler.fit(train_set)
        training_data = self.scaler.transform(train_set)

        # building computation graph
        x = tf.placeholder(tf.float32, [None, 784])
        W = tf.Variable(tf.zeros([784, 10]))
        b = tf.Variable(tf.zeros([10]))
        y = tf.nn.softmax(tf.matmul(x, W) + b)
        y_truth = tf.placeholder(tf.float32, [None, 10])
        cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_truth * tf.log(y), reduction_indices=[1]))
        if self.regulizer == 'l1':
            loss = cross_entropy + self.beta * tf.reduce_sum(W)
        elif self.regulizer == 'l2':
            loss = cross_entropy + tf.nn.l2_loss(W)
        optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

        # start a session to train in batch
        init = tf.initialize_all_variables()
        sess = tf.Session()
        sess.run(init)
        train = input_data.DataSet(training_data, train_label)
        for i in range(1000):
            batch_xs, batch_ys = mnist.train.next_batch(100)
            sess.run(optimizer, feed_dict={x: batch_xs, y_truth: batch_ys})

        # fetch the model parameters W and b

    def predict(self, test_set):
        pass



