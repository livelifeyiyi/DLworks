# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import pickle
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.contrib.layers import flatten
# load dataset
cifar10_folder = "E:\\data\\CIFAR\\cifar-10-batches-py\\"
train_datasets = ['data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4', 'data_batch_5', ]
test_dataset = ['test_batch']
c10_image_height = 32
c10_image_width = 32
c10_image_depth = 3
c10_num_labels = 10
c10_image_size = 32


LENET5_BATCH_SIZE = 32
LENET5_PATCH_SIZE = 5
LENET5_PATCH_DEPTH_1 = 6
LENET5_PATCH_DEPTH_2 = 16
LENET5_NUM_HIDDEN_1 = 120
LENET5_NUM_HIDDEN_2 = 84


# pre-defined functions
def randomize(dataset, labels):
	permutation = np.random.permutation(labels.shape[0])
	shuffled_dataset = dataset[permutation, :, :]
	shuffled_labels = labels[permutation]
	return shuffled_dataset, shuffled_labels


def one_hot_encode(np_array):
	return (np.arange(10) == np_array[:, None]).astype(np.float32)


def reformat_data(dataset, labels, image_width, image_height, image_depth):
	np_dataset_ = np.array([np.array(image_data).reshape(image_width, image_height, image_depth) for image_data in dataset])
	np_labels_ = one_hot_encode(np.array(labels, dtype=np.float32))
	np_dataset, np_labels = randomize(np_dataset_, np_labels_)
	return np_dataset, np_labels


def flatten_tf_array(array):
	shape = array.get_shape().as_list()
	return tf.reshape(array, [shape[0], shape[1] * shape[2] * shape[3]])


def accuracy(predictions, labels):
	return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1)) / predictions.shape[0])


def variables_lenet5(patch_size=LENET5_PATCH_SIZE, patch_depth1=LENET5_PATCH_DEPTH_1,
					 patch_depth2=LENET5_PATCH_DEPTH_2,
					 num_hidden1=LENET5_NUM_HIDDEN_1, num_hidden2=LENET5_NUM_HIDDEN_2,
					 image_depth=1, num_labels=10):
	w1 = tf.Variable(tf.truncated_normal([patch_size, patch_size, image_depth, patch_depth1], stddev=0.1))
	b1 = tf.Variable(tf.zeros([patch_depth1]))

	w2 = tf.Variable(tf.truncated_normal([patch_size, patch_size, patch_depth1, patch_depth2], stddev=0.1))
	b2 = tf.Variable(tf.constant(1.0, shape=[patch_depth2]))

	w3 = tf.Variable(tf.truncated_normal([5 * 5 * patch_depth2, num_hidden1], stddev=0.1))
	b3 = tf.Variable(tf.constant(1.0, shape=[num_hidden1]))

	w4 = tf.Variable(tf.truncated_normal([num_hidden1, num_hidden2], stddev=0.1))
	b4 = tf.Variable(tf.constant(1.0, shape=[num_hidden2]))

	w5 = tf.Variable(tf.truncated_normal([num_hidden2, num_labels], stddev=0.1))
	b5 = tf.Variable(tf.constant(1.0, shape=[num_labels]))
	variables = {
		'w1': w1, 'w2': w2, 'w3': w3, 'w4': w4, 'w5': w5,
		'b1': b1, 'b2': b2, 'b3': b3, 'b4': b4, 'b5': b5
	}
	return variables


def model_lenet5(data, variables):
	# conv2d(input: a 4-D tensor, filter/kernel: a 4-D tensor [filter_height, filter_width, in_channels, out_channels],
	# strides: A list of ints. 1-D tensor of length 4. 第一个维度代表着图像的批量数，这个维度肯定每次只能移动一张图片。
	# 最后一个维度为图片深度（即色彩通道数，1 代表灰度图片，而 3 代表 RGB 图片）
	# padding: "SAME"/"VALID". 是否需要使用 0 来填补图像周边(SAME是需要)，以确保图像输出尺寸在步幅参数设定为 1 的情况下保持不变)
	layer1_conv = tf.nn.conv2d(data, variables['w1'], [1, 1, 1, 1], padding='VALID')
	layer1_actv = tf.sigmoid(layer1_conv + variables['b1'])
	layer1_pool = tf.nn.avg_pool(layer1_actv, [1, 2, 2, 1], [1, 2, 2, 1], padding='VALID')

	layer2_conv = tf.nn.conv2d(layer1_pool, variables['w2'], [1, 1, 1, 1], padding='VALID')
	layer2_actv = tf.sigmoid(layer2_conv + variables['b2'])
	layer2_pool = tf.nn.avg_pool(layer2_actv, [1, 2, 2, 1], [1, 2, 2, 1], padding='VALID')

	flat_layer = flatten_tf_array(layer2_pool)
	# flat_layer = flatten(layer2_pool)
	layer3_fccd = tf.matmul(flat_layer, variables['w3']) + variables['b3']
	layer3_actv = tf.nn.sigmoid(layer3_fccd)

	layer4_fccd = tf.matmul(layer3_actv, variables['w4']) + variables['b4']
	layer4_actv = tf.nn.sigmoid(layer4_fccd)
	logits = tf.matmul(layer4_actv, variables['w5']) + variables['b5']
	return logits


# Variables used in the constructing and running the graph
num_steps = 10001
display_step = 1000
learning_rate = 0.001
batch_size = 16

# 定义数据的基本信息，传入变量
image_width = 32
image_height = 32
image_depth = 3
num_labels = 10

with open(cifar10_folder + test_dataset[0], 'rb') as f0:
	c10_test_dict = pickle.load(f0, encoding='bytes')

c10_test_dataset, c10_test_labels = c10_test_dict[b'data'], c10_test_dict[b'labels']
test_dataset_cifar10, test_labels_cifar10 = reformat_data(c10_test_dataset, c10_test_labels, c10_image_size,
														  c10_image_size, c10_image_depth)

c10_train_dataset, c10_train_labels = [], []
for train_dataset in train_datasets:
	with open(cifar10_folder + train_dataset, 'rb') as f0:
		c10_train_dict = pickle.load(f0, encoding='bytes')
		c10_train_dataset_, c10_train_labels_ = c10_train_dict[b'data'], c10_train_dict[b'labels']

		c10_train_dataset.append(c10_train_dataset_)
		c10_train_labels += c10_train_labels_

c10_train_dataset = np.concatenate(c10_train_dataset, axis=0)
train_dataset_cifar10, train_labels_cifar10 = reformat_data(c10_train_dataset, c10_train_labels, c10_image_size,
															c10_image_size, c10_image_depth)
test_dataset = test_dataset_cifar10
test_labels = test_labels_cifar10
train_dataset = train_dataset_cifar10
train_labels = train_labels_cifar10

graph = tf.Graph()
with graph.as_default():
	# 1 首先使用占位符定义数据变量的维度
	tf_train_dataset = tf.placeholder(tf.float32, shape=(batch_size, image_width, image_height, image_depth))
	tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
	tf_test_dataset = tf.constant(test_dataset, tf.float32)

	# 2 然后初始化权重矩阵和偏置向量
	variables = variables_lenet5(image_depth=image_depth, num_labels=num_labels)

	# 3 使用模型计算分类
	logits = model_lenet5(tf_train_dataset, variables)

	# 4 使用带softmax的交叉熵函数计算预测标签和真实标签之间的损失函数
	loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=tf_train_labels))

	# 5  采用Adam优化算法优化上一步定义的损失函数，给定学习率
	optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

	# 执行预测推断
	train_prediction = tf.nn.softmax(logits)
	test_prediction = tf.nn.softmax(model_lenet5(tf_test_dataset, variables))

with tf.Session(graph=graph) as session:
	# 初始化全部变量
	tf.global_variables_initializer().run()
	print('Initialized with learning_rate', learning_rate)
	for step in range(num_steps):
		offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
		batch_data = train_dataset[offset:(offset + batch_size), :, :, :]
		batch_labels = train_labels[offset:(offset + batch_size), :]
		# 在每一次批量中，获取当前的训练数据，并传入feed_dict以馈送到占位符中
		feed_dict = {tf_train_dataset: batch_data, tf_train_labels: batch_labels}
		_, l, predictions = session.run([optimizer, loss, train_prediction], feed_dict=feed_dict)
		train_accuracy = accuracy(predictions, batch_labels)

		if step % display_step == 0:
			test_accuracy = accuracy(test_prediction.eval(), test_labels)
			message = "step {:04d} : loss is {:06.2f}, accuracy on training set {:02.2f} %, accuracy on test set {:02.2f} %".format(
				step, l, train_accuracy, test_accuracy)
			print(message)
