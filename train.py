# download Mnist dataset(one time) and train the model with proposed ckassifier
import urllib
import os
import gzip
import cPickle
import numpy as np
from sklearn.cross_validation import train_test_split
from logistic_regression import logistic_regression
from utils import OpenData


