from compute_cost import *
from create_placeholder import *
from forward_propogation import *
from forward_propogation_using_Droupout import *
import matplotlib.pyplot as plt
from initialize_parameters import *
from random_mini_batches import *
from predict import *
import scipy.misc
import cv2
from testing_File import tensorflow_save
from scipy import ndimage
import sys
sys.path.append("C:/Users/AK/Desktop/Tensorflow/testing_File")
from testing_File import dataload
def model(X_train, Y_train,X_test,Y_test,learning_rate=0.009,
          num_epochs=1, minibatch_size=64, print_cost=True):
	#create placeholder
	tf.set_random_seed(1) 
	(m,n_H0,n_W0,n_C0)=X_train.shape
	n_y = Y_train.shape[1]
	costs=[]
	X,Y=create_placeholders(n_H0,n_W0,n_C0,n_y)
	# np.random.seed(0)
	seed=3 
	#initailize parameter
	parameters=initialize_parameters()
	#forward propogatation
	Z3=forward_propagation(X,parameters)
	#compute the cost
	cost=compute_cost(Z3,Y)
	#create an optimizer
	optimizer=tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

	init=tf.global_variables_initializer()
	with tf.Session() as sess:
		sess.run(init)
		for epoch in range(num_epochs):
			minibatch_cost=0
			num_minibatches = int(m / minibatch_size)
			batches=random_mini_batches(X_train,Y_train,64,seed)
			seed=seed+1
			for batch in batches:
				minibatch_X,minibatch_Y=batch
				_ , temp_cost = sess.run([optimizer, cost], feed_dict={X:minibatch_X, Y:minibatch_Y})
				minibatch_cost += temp_cost / num_minibatches
            # Print the cost every epoch
			if print_cost == True and epoch % 5 == 0:
				print ("Cost after epoch %i: %f" % (epoch, minibatch_cost))
			if print_cost == True and epoch % 1 == 0:
				costs.append(minibatch_cost)
        # plot the cost
		plt.plot(np.squeeze(costs))
		plt.ylabel('cost')
		plt.xlabel('iterations (per tens)')
		plt.title("Learning rate =" + str(learning_rate))
		plt.show()
		saver=tf.train.Saver()
		saver.save(sess,"C:/Users/AK/Desktop/Tensorflow/saved_model/my_model")
        # lets save the parameters in a variable
		parameters = sess.run(parameters)
		# print("Parameters have been trained!")
		# print(parameters["W1"].shape)
		print("The value of W1",parameters["W1"])
		print("The value of W2",parameters["W2"])
		W1=tf.convert_to_tensor(parameters["W1"])
		W2=tf.convert_to_tensor(parameters["W2"])
		tensorflow_save.save_variables(W1,W2)
		
		# correct_prediction = tf.equal(tf.argmax(Z3,1), tf.argmax(Y,1))
  #      # Calculate accuracy on the test set
		# accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
		# print("Train Accuracy:", accuracy.eval({X: X_train, Y: Y_train}))
		# print("Train Accuracy:", accuracy.eval({X: X_test, Y: Y_test}))
	store_W1,store_W2=tensorflow_save.load_Data()
	if store_W1.all()!=parameters["W1"].all():
		print("Sroring and retriving is same in W1")
	else:
		print("Not same result")
	if store_W2.all()!=parameters["W2"].all():
		print("Sroring and retriving is same in W2")
	else:
		print("Not same result")
		#test
		# for i in range(1,104):
		# 	try:
		# 		print("fotm here")
		# 		path = "C:/Users/AK/Desktop/Tensorflow/dataset/test/"+"set"+str(i)+".jpg"
		# 		image=np.array(ndimage.imread(path,flatten=False))
		# 		if image.shape[2]>3:
		# 			image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
		# 		X_test=scipy.misc.imresize(image,size=(75,75)).reshape(1,75*75*3)
		# 		X_test=X_test/255
		# 		convert=np.reshape(X_test,(1,75,75,3))
		# 		prediction=tf.argmax(Z3,1)
		# 		value=sess.run(prediction,{X:convert})
		# 		# value=prediction.eval({X:convert})
		# 		print(value)
		# 		plt.imshow(image)
		# 		plt.show()
		# 	except:
		# 		path = "C:/Users/AK/Desktop/Tensorflow/dataset/test/"+"set"+str(i)+".jpeg"
		# 		image=np.array(ndimage.imread(path,flatten=False))
		# 		if image.shape[2]>3:
		# 			image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
		# 		X_test=scipy.misc.imresize(image,size=(75,75)).reshape(1,75*75*3)
		# 		X_test=X_test/255
		# 		convert=np.reshape(X_test,(1,75,75,3))
		# 		prediction=tf.argmax(Z3,1)
		# 		value=prediction.eval({X:convert})
		# 		print(value)
		# 		plt.imshow(image)
		# 		plt.show()
			

