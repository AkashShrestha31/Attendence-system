import tensorflow as tf
import numpy as np
from create_placeholder import *
from initialize_parameters import *
from forward_propogation import *
def compute_cost(Z3,Y):
	cost=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=Z3,labels=Y))
	return cost
