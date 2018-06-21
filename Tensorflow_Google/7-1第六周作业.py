# -*- coding: utf-8 -*-
"""
    python文件说明：卷积神经网络的程序
"""
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
batch_size = 100
n_batch = mnist.train.num_examples // batch_size

#参数概要
def variable_summaries(var):
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('historgram', var)

#初始化权值
def weight_variable(shape, name):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial, name=name)

#初始化偏执值
def bias_variable(shape, name):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial, name=name)

#卷基层
def conv2d(x, w):
    return tf.nn.conv2d(x, w, strides=[1,1,1,1], padding='SAME')

#池化层
def max_pool_2x2(h_conv):
    return tf.nn.max_pool(h_conv, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
#命名空间
with tf.name_scope('input'):
    x = tf.placeholder(tf.float32, [None, 784], name='x-input')
    y = tf.placeholder(tf.float32, [None, 10], name='y-input')

    with tf.name_scope('x_image'):
        x_image = tf.reshape(x, [-1, 28, 28, 1], name='x_image')

with tf.name_scope('conv1'):
    with tf.name_scope('w_conv1'):
        w_conv1 = weight_variable([5, 5, 1, 32], 'w_conv1')
    with tf.name_scope('b_conv1'):
        b_conv1 = bias_variable([32], name='b_conv1')
    with tf.name_scope('conv2d_1'):
        conv2d_1 = conv2d(x_image, w_conv1) + b_conv1
    with tf.name_scope('relu_1'):
        h_conv1 = tf.nn.relu(conv2d_1)
    with tf.name_scope('pool_1'):
        h_pool1 = max_pool_2x2(h_conv1)
with tf.name_scope('conv2'):
    with tf.name_scope('w_conv2'):
        w_conv2 = weight_variable([5, 5, 32, 64], 'w_conv2')
    with tf.name_scope('b_conv2'):
        b_conv2 = bias_variable([64], name='b_conv2')
    with tf.name_scope('conv2d_2'):
        conv2d_2 = conv2d(h_pool1, w_conv2) + b_conv2
    with tf.name_scope('relu_2'):
        h_conv2 = tf.nn.relu(conv2d_2)
    with tf.name_scope('pool_2'):
        h_pool2 = max_pool_2x2(h_conv2)
#全连接层1
with tf.name_scope('fc1'):
    with tf.name_scope('w_fc1'):
        w_fc1 = weight_variable([7*7*64, 1024], 'w_fc1')
    with tf.name_scope('b_fc1'):
        b_fc1 = bias_variable([1024], 'b_fc1')
    # 把池化层2的输出扁平化为1维
    with tf.name_scope('h_pool2_flat'):
        h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64], name='h_pool2_flat')
    with tf.name_scope('fc1_result'):
        fc1_result1 = tf.nn.relu(tf.matmul(h_pool2_flat, w_fc1) + b_fc1)

    #drpoout防止过拟合
    with tf.name_scope('keep_prob1'):
        keep_prob = tf.placeholder(tf.float32, name='keep_prob1')
    with tf.name_scope('h_fc1_drop1'):
        h_fc1_drop1 = tf.nn.dropout(fc1_result1, keep_prob=keep_prob, name='h_fc1_drop1')

# 全连接层2
with tf.name_scope('fc2'):
    with tf.name_scope('w_fc2'):
        w_fc2 = weight_variable([1024, 10], 'w_fc2')
    with tf.name_scope('b_fc2'):
        b_fc2 = bias_variable([10], 'b_fc2')
    with tf.name_scope('fc1_result2'):
        fc1_result2 = tf.nn.softmax(tf.matmul(h_fc1_drop1, w_fc2) + b_fc2)

# 交叉熵代价函数
with tf.name_scope('cross_entropy'):
    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=fc1_result2),
                                   name='cross_entropy')
    tf.summary.scalar('cross_entropy', cross_entropy)
#优化函数
with tf.name_scope('train'):
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

with tf.name_scope('accuracy'):
    with tf.name_scope('correct_prediction'):
        correct_prediction = tf.equal(tf.arg_max(fc1_result2, 1),tf.arg_max(y, 1))
    with tf.name_scope('accuracy'):
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar('accuracy', accuracy)
#合并所有的summary
merged = tf.summary.merge_all()

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    train_writer = tf.summary.FileWriter('/home/wangheng/path/train', sess.graph)
    test_writer = tf.summary.FileWriter('/home/wangheng/path/test', sess.graph)

    for i in range(3001):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        sess.run(train_step, feed_dict={x: batch_xs, y:batch_ys, keep_prob:0.5})
        #记录训练集计算的参数
        summary = sess.run(merged, feed_dict={x: batch_xs, y:batch_ys, keep_prob:1.0})
        train_writer.add_summary(summary, i)

        #记录测试集计算的参数
        batch_xs,  batch_ys = mnist.test.next_batch(batch_size)
        summary = sess.run(merged, feed_dict={x:batch_xs, y:batch_ys, keep_prob:1.0})
        test_writer.add_summary(summary, i)

        if i% 100 ==0:
            test_acc = sess.run(accuracy, feed_dict={x:mnist.test.images, y:mnist.test.labels, keep_prob:1.0})
            train_acc = sess.run(accuracy, feed_dict={x:mnist.train.images[:10000], y:mnist.train.labels[:10000], keep_prob:1.0})
            print('Iter'+str(i)+',Testing Accuracy='+str(test_acc)+',Train Accuracy='+str(train_acc))



