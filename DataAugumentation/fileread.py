import os
import numpy as np
from scipy import ndimage,misc
import matplotlib.pyplot as plt
import cv2
from PIL import Image
def changing_channel():
	folder="C:/Users/AK/Desktop/Tensorflow/dataset/filter2/Gokul"
	files=os.listdir(folder)
	files.sort()
	# files = ['{}/{}'.format(folder, file) for file in files]
	k=1
	for i in files:
		path=folder+"/"+str(i)
		ima=np.array(ndimage.imread(path))
		if ima.shape[2]==4:
			img = cv2.cvtColor(ima, cv2.COLOR_BGRA2BGR)
			im=Image.fromarray(img)
			im.save(folder+"/"+str(i))
			print(img.shape)
changing_channel()
		
	# im=Image.fromarray(ima)
	# im.save("E:/pngfile/"+str(k)+".png")
	# k=k+1
	# print(ima.shape)
# ima=np.array(ndimage.imread(files[1]))
# img = cv2.cvtColor(ima, cv2.COLOR_BGRA2BGR)
# print(img.shape)
# plt.imshow(img)
# plt.show()

