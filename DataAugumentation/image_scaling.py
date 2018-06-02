from scipy import misc,ndimage
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
from PIL import Image
import os
# Produce each image at scaling of 90%, 75% and 60% of original image.
def central_scale_images(X_imgs, scales,height,width):
    # Various settings needed for Tensorflow operation
    boxes = np.zeros((len(scales), 4), dtype = np.float32)
    for index, scale in enumerate(scales):
        x1 = y1 = 0.5 - 0.5 * scale # To scale centrally
        x2 = y2 = 0.5 + 0.5 * scale
        boxes[index] = np.array([y1, x1, y2, x2], dtype = np.float32)
    box_ind = np.zeros((len(scales)), dtype = np.int32)
    crop_size = np.array([height,width], dtype = np.int32)
    
    X_scale_data = []
    tf.reset_default_graph()
    X = tf.placeholder(tf.float32, shape = (1, height,width, 3))
    # Define Tensorflow operation for all scales but only one base image at a time
    tf_img = tf.image.crop_and_resize(X, boxes, box_ind, crop_size)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        
        for img_data in X_imgs:
            batch_img = np.expand_dims(img_data, axis = 0)
            scaled_imgs = sess.run(tf_img, feed_dict = {X: batch_img})
            X_scale_data.extend(scaled_imgs)
    
    X_scale_data = np.array(X_scale_data, dtype = np.float32)
    return X_scale_data

def path_File():
    folder="E:/pngfile/"
    files=os.listdir(folder)
    files.sort()
    files = ['{}/{}'.format(folder, file) for file in files]
    return files




# X_imgs = tf_resize_images(X_img_paths)
# scaled_imgs = central_scale_images(X_imgs, [0.90, 0.75, 0.60])

# plt.imshow(scaled_imgs[2])
# plt.show()
##converting images into different formate
# ima=np.array(ndimage.imread("E:/2.jpg"))
# im=Image.fromarray(ima)
# im.save("E:/20.png")
def reading_jpg_saving_png():
    folder="C:/Users/AK/Desktop/Tensorflow/dataset/filter2/Sirsha"
    files=os.listdir(folder)
    files.sort()
    files = ['{}/{}'.format(folder, file) for file in files]
    k=1
    for i in files:
        ima=np.array(ndimage.imread(i))
        im=Image.fromarray(ima)
        im.save("E:/pngfile/"+str(k)+".png")
        k=k+1
reading_jpg_saving_png()
X_img_paths=path_File()
print(X_img_paths)
index=0
for i in X_img_paths:
    print(i)
    img=np.array(ndimage.imread(i))
    img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    print(img.shape)
    X_imgs=misc.imresize(img,(img.shape[0],img.shape[1])).reshape(1,img.shape[0],img.shape[1],3)
    scaled_imgs = central_scale_images(X_imgs, [0.90, 0.75, 0.60],img.shape[0],img.shape[1])
    for i in range(3):
        im=misc.imresize(scaled_imgs[i],(img.shape[0],img.shape[1])).reshape(img.shape[0],img.shape[1],3)
        im=Image.fromarray(im)
        im.save("E:/scaling/scaling"+str(index)+str(i)+".jpg")
        index=index+1

