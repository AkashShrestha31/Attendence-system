import numpy as np
from scipy import ndimage
import scipy.misc
import matplotlib.pyplot as plt
import csv
import tensorflow as tf
from one_hot_matrix import *
import cv2
import csv
def prepare_dataset_for_images(No_of_image,image_size,output,folder_path):
	print(folder_path)
	X = np.zeros((No_of_image*output, image_size*image_size*3))#6632
	Y = np.zeros((No_of_image*output, 1))
	# X_test=np.zeros((48,12288))
	np.random.seed(0)
	names=["Dagina","Abinash","Pawan","Gokul","Sirsha"]
	index=0
	for i in range(0,output):
		for j in range(1,(No_of_image+1)):#1image_size9
			try:
				path = folder_path+names[i]+"/"+names[i].lower()+str(j)+".jpg"
				image=np.array(ndimage.imread(path,flatten=False))
				if image.shape[2]>3:
					image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
				image=scipy.misc.imresize(image,size=(image_size,image_size)).reshape(1,image_size*image_size*3)
				X[index, :] = image
				Y[index] = i
				index=index+1
				print(index,image.shape)
			except:
				path = folder_path+names[i]+"/"+names[i].lower()+str(j)+".jpeg"
				image=np.array(ndimage.imread(path,flatten=False))
				if image.shape[2]>3:
					image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
				image=scipy.misc.imresize(image,size=(image_size,image_size)).reshape(1,image_size*image_size*3)
				X[index, :] = image
				Y[index] = i
				index=index+1
				print(index)

	Y=one_hot_matrix(Y,output)
	with tf.Session() as sess:
		Y=sess.run(Y)
		sess.close()
	
	Y=np.reshape(Y,(Y.shape[0],output))
	permutation=list(np.random.permutation(X.shape[0]))
	X_shuffled=X[permutation,:]
	Y_shuffled=Y[permutation,:]
	X_shuffled=np.reshape(X_shuffled,(X_shuffled.shape[0],image_size,image_size,3))


	# path = "C:/Users/AK/Desktop/Tensorflow/dataset/gokul.jpg"
	# image=np.array(ndimage.imread(path,flatten=False))
	# X_test=scipy.misc.imresize(image,size=(64,64)).reshape(1,64*64*3)

	#This is testSET
	#Testset start
	# for i in range(1,49):
	# 	try:
	# 		path = "C:/Users/AK/Desktop/Tensorflow/dataset/test/"+"set"+str(i)+".jpg"
	# 		image=np.array(ndimage.imread(path,flatten=False))
	# 		image=scipy.misc.imresize(image,size=(64,64)).reshape(1,64*64*3)
	# 		X_test[i, :] = image
	# 	except:
	# 		path = "C:/Users/AK/Desktop/Tensorflow/dataset/test/"+"set"+str(i)+".jpeg"
	# 		image=np.array(ndimage.imread(path,flatten=False))
	# 		image=scipy.misc.imresize(image,size=(64,64)).reshape(1,64*64*3)
	# 		X_test[i, :] = image
	# permutation=list(np.random.permutation(X_test.shape[0]))
	# X_test=X_test[permutation,:]
	# X_test=np.reshape(X_test,(X_test.shape[0],64,64,3))
	#Testset end
	# print(X_shuffled.shape)
	# print(Y_shuffled.shape)
	# f="C:/Users/AK/Desktop/Tensorflow/X.csv"
	# with open(f, 'w') as f:
 #   		writer = csv.writer(f, delimiter=',')
 #   		writer.writerows(X_shuffled)  #considering my_list is a list of lists.
	# np.save("C:/Users/AK/Desktop/Tensorflow/X_shuffled.npy",X_shuffled)
	# np.save("C:/Users/AK/Desktop/Tensorflow/Y_shuffled.npy",Y_shuffled)
	# X=np.load("C:/Users/AK/Desktop/Tensorflow/X_shuffled.npy")
	# Y=np.load("C:/Users/AK/Desktop/Tensorflow/Y_shuffled.npy")
	return X_shuffled,Y_shuffled

		
