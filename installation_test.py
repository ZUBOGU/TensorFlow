#	import TensorFlow
import tensorflow as tf
sess = tf.Session()

#	Verify we can print a string
hello = tf.constant("Hello, TensorFlow.")
print(sess.run(hello))

#	Perform some simple math
a = tf.constant(1)
b = tf.constant(2)
print('a + b = {0}'.format(sess.run(a + b)))
