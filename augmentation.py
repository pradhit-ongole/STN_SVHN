import numpy as np
from skimage import transform
from skimage.transform import warp, AffineTransform
import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import *

def pad_distort_im_fn(x):
    b = np.zeros((40, 40, 3))
    o = int((40-32)/2)
    o1 = int((40-17)/2)
    b[o:o+32, o1:o1+17,:] = x
    x = b
    x = tl.prepro.rotation(x, rg=90, is_random=True, fill_mode='constant')
    x = tl.prepro.shear(x, 0.05, is_random=True, fill_mode='constant')
    x = tl.prepro.shift(x, wrg=0.2, hrg=0.2, is_random=True, fill_mode='constant')
    x = tl.prepro.zoom(x, zoom_range=(0.9, 1.5))
    return x

def pad_distort_ims_fn(X):
    """ Zero pads images to 40x40, and distort them. """
    X_40 = []
    for X_a, _ in tl.iterate.minibatches(X, X, 50, shuffle=False):
        X_40.extend(tl.prepro.threading_data(X_a, fn=pad_distort_im_fn))
    X_40 = np.asarray(X_40)
    return X_40

# create dataset with size of 40x40 with distortion
def aug_data(X_train,X_test):
    X_train_40 = pad_distort_ims_fn(X_train)
    #X_val_40 = pad_distort_ims_fn(X_val)
    X_test_40 = pad_distort_ims_fn(X_test)
    return X_train_40,X_test_40
