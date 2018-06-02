import tensorflow as tf
import numpy as np
from create_placeholder import *
from initialize_parameters import *
def forward_propagation_using_dropout(X,parameters,is_training):
	W1=parameters["W1"]
	W2=parameters["W2"]
	Z1=tf.nn.conv2d(X,W1,strides=[1,1,1,1],padding="SAME")
	A1=tf.nn.relu(Z1)
	P1=tf.nn.max_pool(A1,ksize =[1,8,8,1],strides=[1,8,8,1],padding="SAME")
	Z2=tf.nn.conv2d(P1,W2,strides=[1,1,1,1],padding="SAME")
	A2=tf.nn.relu(Z2)
	P2=tf.nn.max_pool(A2,ksize=[1,4,4,1],strides=[1,4,4,1],padding="SAME")	
	P = tf.contrib.layers.flatten(P2)
	# print(P2.shape[0],P2.shape[1],P2.shape[2])
	Z3 = tf.contrib.layers.fully_connected(P, 1024, activation_fn=tf.nn.relu)
	# pool2_flat = tf.reshape(P2, [-1, P2.shape[1] * P2.shape[2] * P2.shape[3]])
	# dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
	dropout = tf.nn.dropout(x=Z3, keep_prob=0.40)
	out = tf.contrib.layers.fully_connected(dropout, 4, activation_fn=tf.nn.softmax)

  # Logits Layer
	# Z3 = tf.layers.dense(inputs=dropout, units=4)
	return out