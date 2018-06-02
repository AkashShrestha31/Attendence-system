import urllib.request
import cv2
import numpy as np
url='http://192.168.1.66:8080/shot.jpg'
i=0
while True:
    imgResp=urllib.request.urlopen(url)
    imgNp=np.array(bytearray(imgResp.read()),dtype=np.uint8)
    img=cv2.imdecode(imgNp,-1)
    cv2.imwrite("C:/Users/AK/Desktop/Tensorflow/dataset/capture"+str(i)+".jpg",img)
    i=i+1
    cv2.imshow('test',img)
    if ord('q')==cv2.waitKey(10):
        exit(0)