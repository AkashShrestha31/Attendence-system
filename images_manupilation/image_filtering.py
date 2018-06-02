import cv2
import numpy as np
import matplotlib.pyplot as plt
# alpha = 1.6     # Simple contrast control
# beta = 10.0
# img = cv2.imread('C:/Users/AK/Desktop/Tensorflow/dataset/faces/1.jpg', 1)


# # CLAHE (Contrast Limited Adaptive Histogram Equalization)
# mul_img = cv2.multiply(img,np.array([alpha]))                    # mul_img = img*alpha
# new_img = cv2.add(mul_img,np.array([beta]))                      # new_img = img*alpha + beta

# cv2.imshow('original_image', img)
# cv2.imshow('new_image',new_img)

# cv2.waitKey(0)
# cv2.destroyAllWindows()


#ADJUSTING GAMMA
# def adjust_gamma(image, gamma=1.0):
# 	# build a lookup table mapping the pixel values [0, 255] to
# 	# their adjusted gamma values
# 	invGamma = 1.0 / gamma
# 	table = np.array([((i / 255.0) ** invGamma) * 255
# 		for i in np.arange(0, 256)]).astype("uint8")
 
# 	# apply gamma correction using the lookup table
# 	return cv2.LUT(image, table)
# # load the original image
# original = cv2.imread('C:/Users/AK/Desktop/Tensorflow/dataset/faces/2.jpg')
# for gamma in np.arange(0.0, 3.5, 0.1):
# 	# ignore when gamma is 1 (there will be no change to the image)
# 	if gamma == 1:
# 		continue
 
# 	# apply gamma correction and show the images
# 	gamma = gamma if gamma > 0 else 0.1
# 	adjusted = adjust_gamma(original, gamma=gamma)
# 	cv2.putText(adjusted, "g={}".format(gamma), (10, 30),
# 		cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 3)
# 	cv2.imshow("Images", np.hstack([original, adjusted]))
# 	cv2.waitKey(0)
img = cv2.imread('C:/Users/AK/Desktop/Tensorflow/dataset/faces/2.jpg')
kernel = np.array([[-1,-1,-1], [-1,11,-1], [-1,-1,-1]])
im = cv2.filter2D(img, -1, kernel)
cv2.imshow("image",im)
cv2.waitKey(0)
