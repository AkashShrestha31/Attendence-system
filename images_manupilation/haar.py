import cv2
import matplotlib.pyplot as plt
import time

path="C:/Users/AK/Desktop/Tensorflow/dataset/q.jpg"
# path="C:/Users/AK/Desktop/Tensorflow/dataset/faces/a"+str(i)+".jpg"
image=cv2.imread(path)
image2=image.copy()

gray_image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
	# plt.imshow(gray_image,cmap="gray")	#lbpcascade_frontalface
	# plt.show()
haar_face_cascade=cv2.CascadeClassifier("C:/Users/AK/Desktop/Tensorflow/dataset/haarcascade_frontalface_alt.xml")
	#let's detect multiscale (some images may be closer to camera than others) images 
faces = haar_face_cascade.detectMultiScale(gray_image, scaleFactor=1.2, minNeighbors=5);  #1.3
	#print the number of faces found 
print('Faces found: ', len(faces))
	#go over list of faces and draw them as rectangles on original colored 
for (x, y, w, h) in faces:     
	  cv2.rectangle(image, (x-10, y-10), (x+w+10, y+h+10), (0, 255, 0), 4)
	         # plt.imshow(cv2.cvtColor(image2[y-50:y+50+h,x-50:x+w+50],cv2.COLOR_BGR2RGB))
	         # plt.imshow(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
	         # plt.show()
plt.imshow(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
plt.show()