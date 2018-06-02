# import numpy as np
# import scipy.misc
# from scipy import ndimage
# import matplotlib.pyplot as plt
# path = "C:/Users/AK/Desktop/Tensorflow/dataset/filter/Gokul/gokul1.jpg"
# image=np.array(ndimage.imread(path,flatten=False))
# print(image.shape)
# X_test=scipy.misc.imresize(image,size=(75,82)).reshape(75,82,3)
# plt.imshow(X_test)
# plt.show()
import cv2
import numpy as np
from scipy import ndimage
import scipy.misc
def capture():
	data=cv2.VideoCapture("C:/Users/AK/Desktop/images/b.mp4")
	count=0
	ret=True 
	while ret:
		ret,img=data.read()
		print(ret)
		# if cv2.waitKey(30) & 0xFF== ord('w'): # you can increase delay to 2 seconds here
		cv2.imwrite("C:/Users/AK/Desktop/Tensorflow/dataset/captureim/capture"+str(count)+".jpg",img)
		count=count+1
		if cv2.waitKey(30) & 0xFF== ord('q'):
			return
	cv2.destroyAllWindows()
	cv2.VideoCapture(1).release()
capture()
