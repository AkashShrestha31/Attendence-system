import numpy as np
import scipy.misc
from scipy import ndimage
import cv2
import matplotlib.pyplot as plt
# for i in range(1,16):
# path="E:/dagina/d"+str(i)+".jpg"
def compute_average(im): 
	""" Compute the average of a list of images. """
# open first image and make into array of type float 
	averageim = np.array(im)
	for imname in im: 
		try: 
			averageim += np.array(imname,np.float)
		except: 
			print("")
	averageim = averageim/len(im)
# return average as uint8 return 
	return np.array(averageim, np.uint8)

path="C:/Users/AK/Desktop/Tensorflow/dataset/gokul.jpg"
image=np.array(ndimage.imread(path,flatten=False))
image=scipy.misc.imresize(image,(300,300)).reshape(1,300*300*3)
image=compute_average(image)
print(image.shape)
plt.imshow(np.reshape(image,(300,300,3)))
plt.show()
# image=cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
# cv2.imwrite("C:/Users/AK/Desktop/Tensorflow/dataset/8521.jpg",image)
