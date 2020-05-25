import tensorflow as tf
import numpy as np

def preprocess(x_train, x_test, y_train, y_test):
    x_train = np.rollaxis(x_train,3)
    x_test = np.rollaxis(x_test,3)

    y_train[y_train==10]=0
    y_test[y_test==10]=0

    x_train = x_train[:,:,7:24,:]
    x_test = x_test[:,:,7:24,:]

    x_train = x_train/255
    x_test = x_test/255

    return x_train, x_test, y_train, y_test


