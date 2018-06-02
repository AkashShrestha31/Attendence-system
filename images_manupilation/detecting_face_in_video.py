import cv2
import matplotlib.pyplot as plt
import time
haar_face_cascade=cv2.CascadeClassifier("C:/Users/AK/Desktop/Tensorflow/dataset/haarcascade_frontalface_alt.xml")
data=cv2.VideoCapture("C:/Users/AK/Desktop/images/1 (1).mp4")
def detecting_face(image,i):
		j=0
		image2=image.copy()
		print(image.shape)
		gray_image=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
		faces = haar_face_cascade.detectMultiScale(gray_image, scaleFactor=1.2, minNeighbors=5)  #1.3
		print('Faces found: ', len(faces))
		for (x, y, w, h) in faces:     
		         cv2.rectangle(image, (x-10, y-10), (x+w+10, y+h+10), (0, 255, 0), 4)
		         try:
		         	plt.imsave("C:/Users/AK/Desktop/Tensorflow/dataset/faces/"+str(j)+str(i)+".jpg",cv2.cvtColor(image2[y-10:y+10+h,x-10:x+w+10],cv2.COLOR_BGR2RGB))
		         	j=j+1
		         except:
		         	print("Error found")
		         	continue
def capture():
	count=0
	i=0
	ret=True 
	while ret:
		ret,img=data.read()
		detecting_face(img,i)
		i=i+1
		# if cv2.waitKey(30) & 0xFF== ord('w'): # you can increase delay to 2 seconds here
		# cv2.imwrite("C:/Users/AK/Desktop/Tensorflow/dataset/captureim/capture"+str(count)+".jpg",img)
		# count=count+1
		if cv2.waitKey(30) & 0xFF== ord('q'):
			return
	cv2.destroyAllWindows()
	cv2.VideoCapture(1).release()
capture()