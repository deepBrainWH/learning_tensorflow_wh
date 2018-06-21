# -*- coding: utf-8 -*-
"""
    Python文件说明：
"""
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

#读入数据集
mnist = input_data.read_data_sets("../../MNIST_data", one_hot=True)
#每个批次的大小
batch_size = 100
#计算总共有多少个批次
n_batch = mnist.train.num_examples // batch_size  #num_example is the number of train set

#define two place holder
x = tf.placeholder(tf.float32, [None, 784])
y = tf.placeholder(tf.float32, [None, 10])

#创建一个简单的神经网络，只有输入层和输出层，没有隐藏层s
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
prediction = tf.nn.softmax(tf.matmul(x, W) + b)  #信号的综合通过一个softmax函数转换为一个概率的值

#定义二次代价函数
loss = tf.reduce_mean(tf.square(prediction - y))
#使用梯度下降算法
train_step = tf.train.GradientDescentOptimizer(0.2).minimize(loss)

#初始化全局变量
init = tf.global_variables_initializer()
#结果存放在一个bool型列表中
correct_prediction = tf.equal(tf.arg_max(y, 1), tf.arg_max(prediction, 1))
#tf.equal()函数返回true和false
#tf.arg_max(y,1)返回最大的概率的下标（位置）

#求准确率
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
    sess.run(init)
    for epoch in range(131):
        for batch in range(n_batch):
            #每个批次大小是100
            batch_x, batch_y = mnist.train.next_batch(batch_size=batch_size)
            sess.run(train_step, feed_dict={x: batch_x, y: batch_y})
        acc = sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels})
        print("Iter " + str(epoch) + ",Testing Accuracy" + str(acc))
