import tensorflow as tf
import numpy as np
from create_placeholder import *
from initialize_parameters import *
def forward_propagation(X,parameters):
	W1=parameters["W1"]
	W2=parameters["W2"]
	Z1=tf.nn.conv2d(X,W1,strides=[1,1,1,1],padding="SAME")
	A1=tf.nn.relu(Z1)
	P1=tf.nn.max_pool(A1,ksize =[1,8,8,1],strides=[1,8,8,1],padding="SAME")
	Z2=tf.nn.conv2d(P1,W2,strides=[1,1,1,1],padding="SAME")
	A2=tf.nn.relu(Z2)
	P2=tf.nn.max_pool(A2,ksize=[1,4,4,1],strides=[1,4,4,1],padding="SAME")	
	P = tf.contrib.layers.flatten(P2)
	Z3 = tf.contrib.layers.fully_connected(P, 5, activation_fn=None)
	return Z3