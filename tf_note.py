# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np


def use_graph():
	graph = tf.Graph()
	with graph.as_default():
		a = tf.Variable(8, tf.float32)
		b = tf.Variable(tf.zeros([2, 2]), tf.float32)
		w1 = tf.Variable(tf.random_normal([2, 3], stddev=1, seed=1))

	# open a session to run the graph
	with tf.Session(graph=graph) as session:
		tf.global_variables_initializer().run()
		print(a)
		print(session.run(a))
		print(session.run(b))
		print(session.run(w1))
# use_graph()


def variables():
	with tf.Session() as sess:
		a = tf.constant([1, 2, 3, 4])
		b = tf.constant([1, 2, 3, 4])
		result = a + b
		print(sess.run(result))

		g = tf.constant(np.zeros(shape=(2, 2), dtype=np.float32))

		h = tf.zeros([11], tf.int16)
		i = tf.ones([2, 2], tf.float32)
		print(sess.run(g))
		print(sess.run(h))
		print(sess.run(i))

		weights = tf.Variable(tf.truncated_normal([256 * 256, 10]))
		bias = tf.Variable(tf.zeros([10]))

		print(weights.get_shape().as_list())
		print(bias.get_shape().as_list())

		# place holder
		w1 = tf.Variable(tf.random_normal([1, 2], stddev=1, seed=1))
		x = tf.placeholder(tf.float32, shape=(1, 2))
		a = x + w1
		sess.run(tf.global_variables_initializer())
		y_1 = sess.run(a, feed_dict={x: [[0.7, 0.9]]})
		print(y_1)

# variables()


def placeholder_test():
	list_of_points1_ = [[1, 2], [3, 4], [5, 6], [7, 8]]
	list_of_points2_ = [[15, 16], [13, 14], [11, 12], [9, 10]]

	list_of_points1 = np.array([np.array(elem).reshape(1, 2) for elem in list_of_points1_])
	list_of_points2 = np.array([np.array(elem).reshape(1, 2) for elem in list_of_points2_])

	graph = tf.Graph()

	with graph.as_default():
		# 使用 tf.placeholder() 创建占位符 ，在 session.run() 过程中再投递数据
		point1 = tf.placeholder(tf.float32, shape=(1, 2))
		point2 = tf.placeholder(tf.float32, shape=(1, 2))

		def calculate_eucledian_distance(point1, point2):
			difference = tf.subtract(point1, point2)
			power2 = tf.pow(difference, tf.constant(2.0, shape=(1, 2)))
			add = tf.reduce_sum(power2)
			eucledian_distance = tf.sqrt(add)
			return eucledian_distance
		dist = calculate_eucledian_distance(point1, point2)

	with tf.Session(graph=graph) as session:
		tf.global_variables_initializer().run()
		for ii in range(len(list_of_points1)):
			point1_ = list_of_points1[ii]
			point2_ = list_of_points2[ii]
			# 使用feed_dict将数据投入到[dist]中
			feed_dict = {point1: point1_, point2: point2_}
			distance = session.run([dist], feed_dict=feed_dict)
			print("The distance between {} and {} -> {}".format(point1_, point2_, distance))

placeholder_test()