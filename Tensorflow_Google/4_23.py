# -*- coding: utf-8 -*-
"""
    python文件说明：自定义损失函数
"""
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from numpy.random import RandomState

batch_size = 8

x = tf.placeholder(tf.float32, shape=(None, 2), name="x-input")
y_ = tf.placeholder(tf.float32, shape=(None, 1), name="y-input")

w1 = tf.Variable(tf.random_normal([2, 1],stddev=1, seed=1))
y = tf.matmul(x, w1)

loss_less = 10
loss_more = 1

loss = tf.reduce_sum(tf.where(tf.greater(y, y_), loss_more*(y-y_), loss_less*(y_-y)))

train_step = tf.train.AdamOptimizer(0.001).minimize(loss)

rdm = RandomState(1)
dataset_size = 128
X = rdm.rand(dataset_size, 2)
Y = [[x1 + x2 + rdm.rand()/10.0 -0.05] for (x1, x2) in X]

xx = np.zeros([91], dtype=float)
yy = np.ones([91], dtype=float)
with tf.Session() as sess:
    init_op = tf.initialize_all_variables()
    sess.run(init_op)
    STEP = 9000
    for i in range(STEP):
        start = (i*batch_size ) % dataset_size
        end = min(start + batch_size, dataset_size)
        sess.run(train_step, feed_dict={x: X[start:end], y_: Y[start:end]})
        if i % 100==0:
            loss_value = sess.run(loss, feed_dict={x: X[start:end], y_: Y[start:end]})
            print("当前loss值为： %.4f" %loss_value)
            yy[i//100] = loss_value
            print("权值为：\n"+str(sess.run(w1)))
plt.figure()
plt.plot(yy)
plt.show()
