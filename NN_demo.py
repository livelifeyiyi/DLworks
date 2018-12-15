import tensorflow as tf
from numpy.random import RandomState
from tensorflow.examples.tutorials.mnist import input_data


def sample_nn():
    batch_size = 10
    w1 = tf.Variable(tf.random_normal([2, 3], stddev=1, seed=1))
    w2 = tf.Variable(tf.random_normal([3, 1], stddev=1, seed=1))

    # None 可以根据batch 大小确定维度，在shape的一个维度上使用None
    x = tf.placeholder(tf.float32, shape=(None, 2))
    y = tf.placeholder(tf.float32, shape=(None, 1))

    # 激活函数使用ReLU
    a = tf.nn.relu(tf.matmul(x, w1))
    yhat = tf.nn.relu(tf.matmul(a, w2))

    # 定义交叉熵为损失函数，训练过程使用Adam算法最小化交叉熵
    cross_entropy = -tf.reduce_mean(y*tf.log(tf.clip_by_value(yhat, 1e-10, 1.0)))
    train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)

    rdm = RandomState(1)
    data_size = 516

    # 生成两个特征，共data_size个样本
    X = rdm.rand(data_size, 2)
    # 定义规则给出样本标签，所有x1+x2<1的样本认为是正样本，其他为负样本。Y，1为正样本
    Y = [[int(x1+x2 < 1)] for (x1, x2) in X]

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print(sess.run(w1))
        print(sess.run(w2))
        steps = 11000
        for i in range(steps):
            # 选定每一个批量读取的首尾位置，确保在1个epoch内采样训练
            start = i * batch_size % data_size
            end = min(start + batch_size, data_size)
            sess.run(train_step, feed_dict={x: X[start:end], y: Y[start:end]})
            if i % 1000 == 0:
                training_loss = sess.run(cross_entropy, feed_dict={x: X, y: Y})
                print("在迭代 %d 次后，训练损失为 %g" % (i, training_loss))


def fully_connect():
    # 3-layer 全连接神经网络
    # MNIST 的像素为 28×28=784, 每一个输入神经元对应于一个灰度像素点
    INPUT_NODE = 784
    OUTPUT_NODE = 10
    LAYER1_NODE = 500

    BATCH_SIZE = 100

    # 模型相关的参数
    LEARNING_RATE_BASE = 0.8
    LEARNING_RATE_DECAY = 0.99
    REGULARAZTION_RATE = 0.0001
    TRAINING_STEPS = 10000
    MOVING_AVERAGE_DECAY = 0.99

    def inference(input_tensor, avg_class, weights1, biases1, weights2, biases2):
        # 使用滑动平均类
        if avg_class == None:
            layer1 = tf.nn.relu(tf.matmul(input_tensor, weights1) + biases1)
            return tf.matmul(layer1, weights2) + biases2

        else:

            layer1 = tf.nn.relu(tf.matmul(input_tensor, avg_class.average(weights1)) + avg_class.average(biases1))
            return tf.matmul(layer1, avg_class.average(weights2)) + avg_class.average(biases2)

    def train(mnist):
        x = tf.placeholder(tf.float32, [None, INPUT_NODE], name='x-input')
        y_ = tf.placeholder(tf.float32, [None, OUTPUT_NODE], name='y-input')
        # 生成隐藏层的参数。
        weights1 = tf.Variable(tf.truncated_normal([INPUT_NODE, LAYER1_NODE], stddev=0.1))
        biases1 = tf.Variable(tf.constant(0.1, shape=[LAYER1_NODE]))
        # 生成输出层的参数。
        weights2 = tf.Variable(tf.truncated_normal([LAYER1_NODE, OUTPUT_NODE], stddev=0.1))
        biases2 = tf.Variable(tf.constant(0.1, shape=[OUTPUT_NODE]))

        # 计算不含滑动平均类的前向传播结果
        y = inference(x, None, weights1, biases1, weights2, biases2)

        # 定义训练轮数及相关的滑动平均类
        global_step = tf.Variable(0, trainable=False)
        variable_averages = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
        variables_averages_op = variable_averages.apply(tf.trainable_variables())
        average_y = inference(x, variable_averages, weights1, biases1, weights2, biases2)

        # 计算交叉熵及其平均值
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))
        cross_entropy_mean = tf.reduce_mean(cross_entropy)

        # 损失函数的计算
        regularizer = tf.contrib.layers.l2_regularizer(REGULARAZTION_RATE)
        regularaztion = regularizer(weights1) + regularizer(weights2)
        loss = cross_entropy_mean + regularaztion

        # 设置指数衰减的学习率。
        learning_rate = tf.train.exponential_decay(
            LEARNING_RATE_BASE,
            global_step,
            mnist.train.num_examples / BATCH_SIZE,
            LEARNING_RATE_DECAY,
            staircase=True)

        # 优化损失函数
        train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)

        # 反向传播更新参数和更新每一个参数的滑动平均值
        with tf.control_dependencies([train_step, variables_averages_op]):
            train_op = tf.no_op(name='train')

        # 计算正确率
        correct_prediction = tf.equal(tf.argmax(average_y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        # 初始化回话并开始训练过程。
        with tf.Session() as sess:
            tf.global_variables_initializer().run()
            validate_feed = {x: mnist.validation.images, y_: mnist.validation.labels}
            test_feed = {x: mnist.test.images, y_: mnist.test.labels}

            # 循环的训练神经网络。
            for i in range(TRAINING_STEPS):
                if i % 1000 == 0:
                    validate_acc = sess.run(accuracy, feed_dict=validate_feed)
                    print(
                        "After %d training step(s), validation accuracy using average model is %g " % (i, validate_acc))

                xs, ys = mnist.train.next_batch(BATCH_SIZE)
                sess.run(train_op, feed_dict={x: xs, y_: ys})

            test_acc = sess.run(accuracy, feed_dict=test_feed)
            print(("After %d training step(s), test accuracy using average model is %g" % (TRAINING_STEPS, test_acc)))

    # load dataset
    mnist = input_data.read_data_sets("E:/data/MNIST/", one_hot=True)
    train(mnist)