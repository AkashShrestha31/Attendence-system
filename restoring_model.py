import tensorflow as tf
import numpy as np
from scipy import ndimage
import scipy.misc
import numpy as np
import cv2
from tensorflow.python.framework import ops
import matplotlib.pyplot as plt
ops.reset_default_graph() 
tf.set_random_seed(1) 
def forward_propagation(X,W1,W2):
	Z1=tf.nn.conv2d(X,W1,strides=[1,1,1,1],padding="SAME")
	A1=tf.nn.relu(Z1)
	P1=tf.nn.max_pool(A1,ksize =[1,8,8,1],strides=[1,8,8,1],padding="SAME")
	Z2=tf.nn.conv2d(P1,W2,strides=[1,1,1,1],padding="SAME")
	A2=tf.nn.relu(Z2)
	P2=tf.nn.max_pool(A2,ksize=[1,4,4,1],strides=[1,4,4,1],padding="SAME")	
	P = tf.contrib.layers.flatten(P2)
	Z3 = tf.contrib.layers.fully_connected(P, 50, activation_fn=tf.nn.relu)
	Z3 = tf.contrib.layers.fully_connected(Z3, 5, activation_fn=None)
	return Z3
X=tf.placeholder(tf.float32,[None,75*75])
X= tf.reshape(X, [-1, 75, 75, 3])
W1 = tf.get_variable("v1", [4, 4, 3, 8])
W2 = tf.get_variable("v2", [2, 2, 8, 16])
Z3=forward_propagation(X,W1,W2)
init = tf.global_variables_initializer();
with tf.Session() as sess: 
	sess.run(init)
	saver= tf.train.import_meta_graph('C:/Users/AK/Desktop/Tensorflow/saved_model/my_model.meta')
	saver.restore(sess,tf.train.latest_checkpoint("C:/Users/AK/Desktop/Tensorflow/saved_model/."))
	# sess.run(tf.initialize_all_variables())
    # saver.restore(sess, "C:/Users/AK/Desktop/Tensorflow/saved_model/my_model.meta")
	w1 = sess.run("W1:0")
	print("The value restoring W1",W1)
	w2 = sess.run("W2:0")
	print("The value restoring W1",W2)
	for i in range(1,55):
				path = "C:/Users/AK/Desktop/Tensorflow/dataset/test/Dagina/dagina"+str(i)+".jpg"
				image=np.array(ndimage.imread(path,flatten=False))
				if image.shape[2]>3:
					image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
				X_test=scipy.misc.imresize(image,size=(75,75)).reshape(1,75*75*3)
				X_test=X_test/255
				convert=np.reshape(X_test,(1,75,75,3))
				prediction=tf.argmax(Z3,1)
				value=sess.run(prediction,{X:convert,W1:w1,W2:w2})
				# value=prediction.eval({X:convert})
				print(value)
				plt.imshow(image)
				plt.show()