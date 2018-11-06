import tensorflow as tf

from learning_tf import input_data


def weight_variable(shape):
    init = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(init)


def bias_variable(shape):
    init = tf.constant(0.1, shape=shape)
    return tf.Variable(init)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x,
                          ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1],
                          padding='SAME')


if __name__ == '__main__':
    # 创建运算的后端session
    sess = tf.InteractiveSession()

    # 导入MINIST数据集
    minist = input_data.read_data_sets("MNIST_data/", one_hot=True)

    # 定义权重W, 图片输入x, 偏置b
    # (W · x + b) x代表任意张大小为28*28的图片
    x = tf.placeholder("float", [None, 784])
    W = tf.Variable(tf.zeros([784, 10]))
    b = tf.Variable(tf.zeros([10]))

    # 第一层卷积: 卷积+max pooling
    W_conv1 = weight_variable([5, 5, 1, 32])
    b_conv1 = bias_variable([32])
    x_img = tf.reshape(x, [-1, 28, 28, 1])
    h_conv1 = tf.nn.relu(conv2d(x_img, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    # 第二层卷积
    W_conv2 = weight_variable([5, 5, 32, 64])
    b_conv2 = bias_variable([64])
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)

    # FC层
    W_fc1 = weight_variable([7 * 7 * 64, 1024])
    b_fc1 = bias_variable([1024])
    h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

    # dropout
    keep_prob = tf.placeholder("float")
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    # softmax
    W_fc2 = weight_variable([1024, 10])
    b_fc2 = bias_variable([10])
    y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2))

    # y = W · x + b
    y = tf.nn.softmax(tf.matmul(x, W) + b)
    y_ = tf.placeholder("float", [None, 10])
    cross_entropy = -tf.reduce_sum(y_ * tf.log(y))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    sess.run(tf.global_variables_initializer())

    for i in range(20000):
        batch = minist.train.next_batch(50)
        if i % 100 == 0:
            train_accuracy = accuracy.eval(feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
            print("step {}, training accuracy:{}".format(i, train_accuracy))
        train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
    print("test accuracy {}".format(accuracy.eval(feed_dict={x: minist.test.images, y_: minist.test.labels, keep_prob: 1.0})))


    # # (W · x + b) x代表任意张大小为28*28的图片
    # x = tf.placeholder("float", [None, 784])
    # W = tf.Variable(tf.zeros([784, 10]))
    # b = tf.Variable(tf.zeros([10]))
    #
    # # y = W · x + b
    # y = tf.nn.softmax(tf.matmul(x, W) + b)
    # y_ = tf.placeholder("float", [None, 10])
    # cross_entropy = -tf.reduce_sum(y_ * tf.log(y))
    # train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
    #
    # init = tf.global_variables_initializer()
    # sess = tf.Session()
    # sess.run(init)
    #
    # for i in range(1000):
    #     batch_xs, batch_ys = minist.train.next_batch(100)
    #     sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
    #
    # correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    # accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    # print(sess.run(accuracy, feed_dict={x: minist.test.images, y_: minist.test.labels}))
