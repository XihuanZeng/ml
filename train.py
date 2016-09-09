# download Mnist dataset(one time) and train the model with proposed ckassifier
import urllib
import os
import gzip
import cPickle
from logistic_regression import logistic_regression


# download and save data(one time)
url = 'http://deeplearning.net/data/mnist/mnist.pkl.gz'
if not 'mnist.pkl.gz' in os.listdir('.'):
    urllib.urlretrieve(url, 'mnist.pkl.gz')
f = gzip.open('mnist.pkl.gz', 'rb')
train_set, valid_set, test_set = cPickle.load(f)
f.close()





