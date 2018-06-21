# -*- coding: utf-8 -*-
"""
    python文件说明：完整的神经网络程序
"""
import tensorflow as tf
from numpy.random import RandomState

batch_size = 8

w1 = tf.Variable(tf.random_normal([2, 3], stddev=1, seed=1))
w2 = tf.Variable(tf.random_normal([3, 1], stddev=1, seed=1))

x = tf.placeholder(tf.float32, shape=(None, 2), name="x-input")
y_ = tf.placeholder(tf.float32, shape=(None, 1), name="y-input")

#定义网络传播过程
a = tf.matmul(x, w1)
y = tf.matmul(a, w2)

#定义损失函数和反向传播算法
'''
    tf.clip_by_value(y, 1e-10, 1.0)函数作用：
        将y中小于1e-10的数值替换为1e-10,大于1.0的数值替换为1.0,防止交叉上计算过程中出现log(0)的没有意义的错误
'''
cross_entropy = -tf.reduce_mean(y_ * tf.log(tf.clip_by_value(y, 1e-10, 1.0)))

train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)

rdm = RandomState(1)
data_set_size = 128
X = rdm.rand(data_set_size, 2)
Y = [[int(x1 + x2)<1] for (x1, x2) in X]

with tf.Session() as sess:
    init_op = tf.initialize_all_variables()
    sess.run(init_op)
    print("训练之前神经网络的权值：\n")
    print(sess.run(w1))
    print(sess.run(w2))

    #训练轮数
    STEP = 10000
    for i in range(STEP):
        start = (i * batch_size) % data_set_size
        end = min(start+batch_size, data_set_size)
        sess.run(train_step, feed_dict={x: X[start:end], y_: Y[start:end]})

        if i % 1000 == 0:
            total_cross_entropy = sess.run(cross_entropy, feed_dict={x: X, y_:Y})
            print("%d 轮训练后,错误率为：%g"%(i, total_cross_entropy))
    print("训练之后的神经网路权值")
    print(sess.run(w1))
    print(sess.run(w2))
