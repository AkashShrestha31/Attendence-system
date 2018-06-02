# Create some variables.
import tensorflow as tf
def save_variables(W1,W2):
	init_op = tf.global_variables_initializer()
	v1 = tf.get_variable("v1", shape=[4, 4, 3, 8], initializer = tf.zeros_initializer)
	v2 = tf.get_variable("v2", shape=[2, 2, 8, 16], initializer = tf.zeros_initializer)
	w1_v1=v1.assign(W1)
	w2_v2=v2.assign(W2)
	saver = tf.train.Saver()
	with tf.Session() as sess:
	  sess.run(init_op)
	  # Do some work with the model.
	  w1_v1.op.run()
	  w2_v2.op.run()
	  # print(v1.eval())
	  # Save the variables to disk.
	  save_path = saver.save(sess, "C:/Users/AK/Desktop/Tensorflow/saved_weights/model.ckpt",global_step=1000)
	  # print(sess.run(w1_v1))
	  # print(sess.run(w2_v2))
	  print("Model saved in path: %s" % save_path)
def load_Data():
	tf.reset_default_graph()
	# Create some variables.
	W1 = tf.get_variable("v1", [4, 4, 3, 8])
	W2 = tf.get_variable("v2", [2, 2, 8, 16])

	# Add ops to save and restore all the variables.
	saver = tf.train.Saver()

	# Later, launch the model, use the saver to restore variables from disk, and
	# do some work with the model.
	with tf.Session() as sess:
	  # Restore variables from disk.
	  saver.restore(sess, "C:/Users/AK/Desktop/Tensorflow/saved_weights/model.ckpt-1000")
	  print("Model restored.")
	  return W1.eval(),W2.eval()
