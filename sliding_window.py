#SLIDING WINDOW TEST
import numpy as np
from numpy import *
from scipy import ndimage
import scipy.misc
import matplotlib.pyplot as plt
import cv2
import csv
obj=open("C:/Users/AK/Desktop/Tensorflow/dataset/color.csv", "r",)
file=csv.reader(obj)
np.random.seed(0)
color=[]
def choose():
	for line in file:
		color.append([line[1],line[2],line[3]])
def sliding(A_prev,stride,f,im):	
	color_index=0
	n_W,n_H,n_C=A_prev.shape
	print(A_prev.shape)
	for i in range(1):
		for w in range(n_H):
			for  h in range(n_W):
					vert_start=h*stride
					vert_end=vert_start+f
					horiz_start=w*stride
					horiz_end=horiz_start+f
					a_slice=A_prev[vert_start:vert_end,horiz_start:horiz_end,:]
					if color_index>154:
						color_index=0
					i_h,i_w,cha=a_slice.shape
					if i_h>f or i_w<f or i_w>f or i_h<f:#vert_start>23
						break
					else:
						im=im+1
					a_slice=cv2.cvtColor(a_slice, cv2.COLOR_BGR2RGB)
					cv2.imwrite("C:/Users/AK/Desktop/Tensorflow/dataset/capture/"+str(im)+".jpg",a_slice)
					color_index=color_index+1
					if cv2.waitKey=='q':
						cv2.destroyAllWindows()
	plt.imshow(A_prev)
	plt.show()
	return im
choose()
print(color[4])
path="C:/Users/AK/Desktop/Tensorflow/dataset/2.jpg"
s=100
f=300
im=0
for i in range(5):
	image=np.array(ndimage.imread(path,flatten=False))
	print(im)
	im=sliding(image,s,f,im)
	s=s+10






