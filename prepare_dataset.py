# 	path="C:/Users/AK/Desktop/attendance system using github/CONVOLUTINAL/capture/capture102.jpg"
# 	image=np.array(ndimage.imread(path))
# 	my_image = scipy.misc.imresize(image, size=(64, 64,3))
# 	print(image.shape)
# 	plt.imshow(my_image)
# 	plt.show()
# 	plt.imshow(image)
# 	plt.show()
# prepare_dataset()
from numpy import *
import numpy as np
import matplotlib.pyplot as plt

from one_hot_matrix import *
import tensorflow as tf
# import onehot
def prepare_dataset():
	train="C:/Users/AK/Desktop/Tensorflow/dataset/train.csv"
	x_train=genfromtxt(train,delimiter=',')
	# no_of_train,fetures=x[1:,1:].shape
	X_train =x_train[1:,1:]#np.zeros((no_of_train,fetures))
	# print(X.shape)
	Y_train =x_train[1:,0:1]#np.zeros((no_of_train,10))
	# print(Y.shape)
	Y_train=one_hot_matrix(Y_train,10)
	with tf.Session() as sess:
		Y_train=sess.run(Y_train)
		sess.close()
	Y_train=np.reshape(Y_train,(Y_train.shape[0],10))
	X_train=np.reshape(X_train,(X_train.shape[0],28,28,1))

	# Test
	test="C:/Users/AK/Desktop/Tensorflow/dataset/mnist_test.csv"
	x_test=genfromtxt(test,delimiter=',')

	# no_of_test,fetures=x[1:,1:].shape
	X_test =x_test[1:,1:]#np.zeros((no_of_test,fetures))
	# print(X.shape)
	Y_test =x_test[1:,0:1]#np.zeros((no_of_train,10))
	Y_test=one_hot_matrix(Y_test,10)
	with tf.Session() as sess:
		Y_test=sess.run(Y_test)
	# print(X.shape)
	Y_test=np.reshape(Y_test,(Y_test.shape[0],10))
	X_test=np.reshape(X_test,(X_test.shape[0],28,28,1))
	
	return X_train,Y_train,X_test,Y_test