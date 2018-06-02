import tensorflow as tf
def initialize_parameters():
	tf.set_random_seed(1)                            
	W1 = tf.get_variable("W1", [4, 4, 3, 8], initializer=tf.contrib.layers.xavier_initializer(seed=0))
	W2 = tf.get_variable("W2", [2, 2, 8, 16], initializer=tf.contrib.layers.xavier_initializer(seed=0))
	parameters = {"W1": W1,"W2": W2}
	return parameters
