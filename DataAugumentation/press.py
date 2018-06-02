import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage,misc
import tensorflow as tf
# def rotate_images(X_imgs):
#     X_rotate = []
#     tf.reset_default_graph()
#     X = tf.placeholder(tf.float32, shape = (197, 182, 3))
#     k = tf.placeholder(tf.int32)
#     tf_img = tf.image.rot90(X, k = k)
#     with tf.Session() as sess:
#         sess.run(tf.global_variables_initializer())
#         for img in X_imgs:
#             for i in range(3):  # Rotation at 90, 180 and 270 degrees
#                 rotated_img = sess.run(tf_img, feed_dict = {X: img, k: i + 1})
#                 X_rotate.append(rotated_img)
#     X_rotate = np.array(X_rotate[1], dtype = np.float32)
#     return X_rotate
# face = np.array(ndimage.imread("E:/1.jpg")) 
# print(face.shape)
# plt.imshow(face)
# plt.show()
# face=misc.imresize(face,(197,182,3)).reshape(1,197,182,3)
# rotated_imgs = rotate_images(face)
# print(rotated_imgs.shape)
# plt.imshow(rotated_imgs)
# plt.show()
import math
for i in range(1,115):
    images = np.array(ndimage.imread("C:/Users/AK/Desktop/Tensorflow/dataset/filter2/Sirsha/sirsha"+str(i)+".jpg")) 
# angles=90
    tf.reset_default_graph()
    # image=tf.contrib.image.rotate(images, 270* math.pi / 180, interpolation='BILINEAR') #rotating image
    image=tf.image.transpose_image(images)
    # image=tf.image.flip_left_right(images)
    # image=tf.image.flip_up_down(images)
    with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            new_image=sess.run(image) 
            plt.imsave("C:/Users/AK/Desktop/Tensorflow/dataset/filter2/new/sirsha/transpose_image"+str(i)+".jpg",new_image)