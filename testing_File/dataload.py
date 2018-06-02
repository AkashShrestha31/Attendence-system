import sys
import tensorflow as tf
from scipy import ndimage
import scipy.misc
import numpy as np
import cv2
import matplotlib.pyplot as plt
sys.path.append("C:/Users/AK/Desktop/Tensorflow")
from forward_propogation import *
def result(X,parameters):
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
def prediction(parameters):
	X_val=tf.placeholder(tf.float32,[None,75*75])
	X=tf.reshape(X_val, [-1, 75, 75, 3])
	Z3=forward_propagation(X,parameters)
	init=tf.global_variables_initializer()
	with tf.Session() as sess:
		sess.run(init)
		for i in range(1,55):
					path = "C:/Users/AK/Desktop/Tensorflow/dataset/test/"+"set"+str(i)+".jpg"
					image=np.array(ndimage.imread(path,flatten=False))
					if image.shape[2]>3:
						image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
					X_test=scipy.misc.imresize(image,size=(75,75)).reshape(1,75*75*3)
					X_test=X_test/255
					convert=np.reshape(X_test,(1,75,75,3))
					prediction=tf.argmax(Z3,1)
					value=prediction.eval(feed_dict={X:convert})
					print(value)
					plt.imshow(image)
					plt.show()
def load_Data():
	tf.reset_default_graph()
	# Create some variables.
	W1 = tf.get_variable("v1", [4, 4, 3, 8])
	W2 = tf.get_variable("v2", [2, 2, 8, 16])

	# Add ops to save and restore all the variables.
	saver = tf.train.Saver()

	# Later, launch the model, use the saver to restore variables from disk, and
	# do some work with the model.
	init=tf.global_variables_initializer()
	with tf.Session() as sess:
	  # Restore variables from disk.
	  sess.run(init)
	  saver.restore(sess, "C:/Users/AK/Desktop/Tensorflow/saved_weights/model.ckpt-1000")
	  print("Model restored.")
	  print(W1.eval())
	  W1=W1.eval()
	  W2=W2.eval()
	  sess.close()
	  return W1,W2

# W1,W2=load_Data()
# parameters={"W1":W1,"W2":W2}
# prediction(parameters)
