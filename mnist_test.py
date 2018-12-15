# -*- coding: utf-8 -*-
# referred to http://www.tensorfly.cn/tfdoc/tutorials/mnist_beginners.html
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
# mnist = tf.keras.datasets.mnist
# (x_train, y_train), (x_test, y_test) = mnist.load_data()

x = tf.placeholder("float", [None, 784])  # 初始化占位符x，维度是a*784，None表示a的维度可以是任意长度
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

y = tf.nn.softmax(tf.matmul(x, W) + b)
y_ = tf.placeholder("float", [None, 10])  # 真实值
cross_entropy = -tf.reduce_sum(y_*tf.log(y))  # 交叉熵作为损失函数

train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)  # 梯度下降最小化交叉熵

init = tf.initialize_all_variables()  # 初始化所创建的向量

sess = tf.Session()
sess.run(init)

for i in range(1000):
	batch_xs, batch_ys = mnist.train.next_batch(100)
	sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
	# 随机抓取训练数据中的100个批处理数据点，然后用这些数据点作为参数替换之前的占位符来运行train_step
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
# argmax 返回某个tensor对象在某一维上最大值所在的索引值
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
