import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import math

from samplerNinterpolation import sample_interpolate


def get_initial_weights(output_size):
	b = np.zeros((2, 3), dtype='float32')
	b[0, 0] = 1
	b[1, 1] = 1
	W = np.zeros((output_size, 6), dtype='float32')
	weights = [W, b.flatten()]
	return weights


def random_mini_batches(X, Y, mini_batch_size = 64):
	m = X.shape[0]
	mini_batches = []
	num_complete_minibatches = math.floor(m/mini_batch_size)

	for k in range(0, num_complete_minibatches):
		mini_batch_X = X[k * mini_batch_size : k * mini_batch_size + mini_batch_size, :, :]
		mini_batch_Y = Y[k * mini_batch_size : k * mini_batch_size + mini_batch_size, :]
		mini_batch = (mini_batch_X, mini_batch_Y)
		mini_batches.append(mini_batch)

	if m % mini_batch_size != 0:
		mini_batch_X = X[num_complete_minibatches * mini_batch_size : m, :, :]
		mini_batch_Y = Y[num_complete_minibatches * mini_batch_size : m, :]
		mini_batch = (mini_batch_X, mini_batch_Y)
		mini_batches.append(mini_batch)
	return mini_batches

def append_shuffle(train_data,train_label,val_data,val_label,test_data,test_label,train_aug,val_aug,test_aug):
	m = np.size(train_data,0)
	b = np.zeros((m, 40, 40, 1))
	o = int((40-28)/2)
	b[:, o:o+28, o:o+28] = train_data
	train_data = np.squeeze(b)

	m = np.size(val_data,0)
	b = np.zeros((m, 40, 40, 1))
	o = int((40-28)/2)
	b[:, o:o+28, o:o+28] = val_data
	val_data = np.squeeze(b)

	m = np.size(test_data,0)
	b = np.zeros((m, 40, 40, 1))
	o = int((40-28)/2)
	b[:, o:o+28, o:o+28] = test_data
	test_data = np.squeeze(b)

	train_app = np.append(train_data,train_aug,axis=0)
	val_app = np.append(val_data,val_aug,axis=0)
	test_app = np.append(test_data,test_aug,axis=0)

	train_appl = np.append(train_label,train_label,axis=0)
	val_appl = np.append(val_label,val_label,axis=0)
	test_appl = np.append(test_label,test_label,axis=0)

	m = np.size(train_app,0)
	permutation = list(np.random.permutation(m))
	train_sh = train_app[permutation,:,:]
	train_sh_label = train_appl[permutation,:]

	m = np.size(val_app,0)
	permutation = list(np.random.permutation(m))
	val_sh = val_app[permutation,:,:]
	val_sh_label = val_appl[permutation,:]

	m = np.size(test_app,0)
	permutation = list(np.random.permutation(m))
	test_sh = test_app[permutation,:,:]
	test_sh_label = test_appl[permutation,:]

	return train_sh,train_sh_label,val_sh,val_sh_label,test_sh,test_sh_label

def plot_imagesNlabels(data, labels):
	rand_mine = np.random.randint(0,data.shape[0],12)
	sampled_x = data[rand_mine]
	sampled_y = labels[rand_mine].reshape(12,10)
	num_rows = 2
	num_cols = 6
	f, ax = plt.subplots(num_rows, num_cols, figsize = (12,5), gridspec_kw = {'wspace':0.03 , 'hspace':0.01}, squeeze = True)
	for i in range (num_rows):
		for j in range (num_cols):
			image_index = i*6 + j
			ax[i,j].axis("off")
			ax[i,j].imshow(np.squeeze(sampled_x[image_index]), cmap='gray')
			ax[i,j].set_title('No. %d' % np.where(sampled_y[image_index] == 1))
	plt.show()