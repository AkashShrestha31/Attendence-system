import tensorflow as tf
import numpy as np
def one_hot_matrix(labels,C):
	one_hot_matrix=tf.one_hot(indices=labels,depth=C,axis=1)
	return one_hot_matrix
