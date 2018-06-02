import cv2
import matplotlib.pyplot as plt
import time
haar_face_cascade=cv2.CascadeClassifier("C:/Users/AK/Desktop/Tensorflow/dataset/haarcascade_frontalface_alt.xml")
for i in range(0,2571):
	path="C:/Users/AK/Desktop/Tensorflow/dataset/captureim/capture"+str(i)+".jpg"
	# path="C:/Users/AK/Desktop/Tensorflow/dataset/faces/a"+str(i)+".jpg"
	image=cv2.imread(path)
	image2=image.copy()
	print(image.shape,i)
	gray_image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
	# plt.imshow(gray_image,cmap="gray")
	# plt.show()
	#let's detect multiscale (some images may be closer to camera than others) images 
	faces = haar_face_cascade.detectMultiScale(gray_image, scaleFactor=1.2, minNeighbors=5);  #1.3
	j=0
	#print the number of faces found 
	print('Faces found: ', len(faces))
	#go over list of faces and draw them as rectangles on original colored 
	for (x, y, w, h) in faces:     
	         cv2.rectangle(image, (x-10, y-10), (x+w+10, y+h+10), (0, 255, 0), 4)
	         # plt.imshow(cv2.cvtColor(image2[y-50:y+50+h,x-50:x+w+50],cv2.COLOR_BGR2RGB))
	         # plt.imshow(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
	         # plt.show()
	         try:
	         	plt.imsave("C:/Users/AK/Desktop/Tensorflow/dataset/faces/"+str(j)+str(i)+".jpg",cv2.cvtColor(image2[y-10:y+10+h,x-10:x+w+10],cv2.COLOR_BGR2RGB))
	         	j=j+1
	         except:
	         	print("error message")
	         	continue
	         
	# plt.imshow(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
	# plt.show()