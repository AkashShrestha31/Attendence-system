from model import *
from prepare_dataset import *

from prepare_dataset_for_images import *
X_train,Y_train=prepare_dataset_for_images(762,75,5,"C:/Users/AK/Desktop/Tensorflow/dataset/filter2/new/")#762
X_test,Y_test=prepare_dataset_for_images(25,75,5,"C:/Users/AK/Desktop/Tensorflow/dataset/test/")
# X=np.load("C:/Users/AK/Desktop/Tensorflow/X_shuffled.npy")
# Y=np.load("C:/Users/AK/Desktop/Tensorflow/Y_shuffled.npy")
X_train=X_train/255
X_test=X_test/255
model(X_train,Y_train,X_test,Y_test)
# model(X,Y)  