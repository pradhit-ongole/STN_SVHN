import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import math


class sample_interpolate(tf.keras.layers.Layer):
	def __init__(self, output_size, **kwargs):
		self.output_size = output_size
		super(sample_interpolate, self).__init__(**kwargs)
		
	def get_config(self):
		return {
			'output_size': self.output_size,
		}
		
	def call(self, tensors, mask=None):
		X, transformation = tensors
		output = self._network(X, transformation, self.output_size)
		return output
		
	def _network(self, U, theta, out_size=None):
		
		
		num = tf.shape(U)[0]
		Height = tf.shape(U)[1]
		Width = tf.shape(U)[2]
        num_channels = tf.shape(U)[3]
		theta_mat = tf.reshape(theta, [num, 2, 3])
        theta_mat = tf.cast(theta_mat, 'float32')
        Height_f = tf.cast(Height, 'float32')
        Width_f = tf.cast(Width, 'float32')
		if out_size:
			out_H = out_size[0]
			out_W = out_size[1]

			batch_grids = self._grid_gen(out_H,out_W,theta_mat)

		else:
			batch_grids = self._grid_gen(Height, Width, theta_mat)

		x_s = batch_grids[:, 0, :, :].squeeze()
		y_s = batch_grids[:, 1, :, :].squeeze()

		out_fmap = self._bilinear_sampler(U, x_s, y_s)
		return out_fmap

        #completed

	'''
    def _get_pixel_value(self, img , x, y):
		
			
		#should be used for the other dataset
		shape = tf.shape(img)
		batch_size = shape[0]
		height = shape[1]
		width = shape[2]
        channels = shape[3]

		batch_index = tf.range(0, batch_size)
		batch_index = tf.reshape(batch_index, (batch_size,1,1))

		b = tf.tile(batch_index, (1,height, width))

		indices = tf.stack([b, y, x], 3)

		return tf.gather_nd(img, indices)
    '''

	def _grid_gen(self, height, width, theta):
		
			
		#makes grids

		num_batch = tf.shape(theta)[0]

		x = tf.linspace(-1.0, 1.0, width)
		y = tf.linspace(-1.0, 1.0, height)
		x_t, y_t = tf.meshgrid(x,y)

		x_t_flat = tf.reshape(x_t, [-1])    #used to flatten a tensor
		y_t_flat = tf.reshape(y_t, [-1])

		ones = tf.ones_like(x_t_flat)  #homogenous coordinates
		sampling_grid = tf.stack([x_t_flat, y_t_flat, ones]) 

		sampling_grid = tf.expand_dims(sampling_grid, axis=0)
		sampling_grid = tf.tile(sampling_grid, [num_batch, 1, 1])

		theta = tf.cast(theta, 'float32')
		sampling_grid = tf.cast(sampling_grid, 'float32')

		batch_grids = tf.matmul(theta, sampling_grid)

		batch_grids = tf.reshape(batch_grids, [num_batch, 2, height, width])

		return batch_grids

	def _bilinear_sampler(self, img, x, y):
		
        num_batch_1 = tf.shape(img)[0]
		H = tf.shape(img)[1]
		W = tf.shape(img)[2]
        channels = tf.shape(img)[3]
		
        max_y = tf.cast(H-1, 'int32')
		max_x = tf.cast(W-1, 'int32')
		
        zero = tf.zeros([], dtype='int32')
        
        W_f = tf.cast(W, 'float32')
        H_f = tf.cast(H, 'float32')
		
        x_f = tf.cast(x, 'float32')
		y_f = tf.cast(y, 'float32')
		
        x = 0.5 * ((x_f + 1.0) * W_f)
		y = 0.5 * ((y_f + 1.0) * H_f)

		x0 = tf.cast(tf.floor(x), 'int32')
		x1 = x0 + 1
		y0 = tf.cast(tf.floor(y), 'int32')
		y1 = y0 + 1

		x0 = tf.clip_by_value(x0, zero, max_x)
		x1 = tf.clip_by_value(x1, zero, max_x)
		y0 = tf.clip_by_value(y0, zero, max_y)
		y1 = tf.clip_by_value(y1, zero, max_y)    

		Ia = input_img[np.arange(num_batch)[:,None,None], y0, x0]
        Ib = input_img[np.arange(num_batch)[:,None,None], y1, x0]
        Ic = input_img[np.arange(num_batch)[:,None,None], y0, x1]
        Id = input_img[np.arange(num_batch)[:,None,None], y1, x1]

		x0_f = tf.cast(x0, 'float32')
		x1_f = tf.cast(x1, 'float32')
		y0_f = tf.cast(y0, 'float32')
		y1_f = tf.cast(y1, 'float32')

		wa = (x1_f-x) * (y1_f-y)
		wb = (x1_f-x) * (y-y0_f)
		wc = (x-x0_f) * (y1_f-y)
		wd = (x-x0_f) * (y-y0_f)

		wa = tf.expand_dims(wa, axis=3)
		wb = tf.expand_dims(wb, axis=3)
		wc = tf.expand_dims(wc, axis=3)
		wd = tf.expand_dims(wd, axis=3)

		out = tf.add_n([wa*Ia, wb*Ib, wc*Ic, wd*Id])

		return out