# -*- coding: utf-8 -*-
'''
    python file describe:卷积网络基本操作
'''
# 假设我们输入矩阵[[1,-1,0],[-1, 2, 1],[0,2,-2]]
import tensorflow as tf
import numpy as np

# define a Matrix, it's shape is(3*3*1)
M = np.array([
    [[1], [-1], [0]],
    [[-1], [2], [1]],
    [[0], [2], [-2]]
])

# define a convolution filter, the deepth is 1
# shape = [2,2,1,1] means that the  depth of the current layer is 1,
# the depth of the filter is 1, the filter's size is 2*2
filter_weight = tf.get_variable("weights", [2, 2, 1, 1],
                                initializer=tf.constant_initializer([
                                    [1, -1],
                                    [0, 2]]))

biases = tf.get_variable("biases", [1], initializer=tf.constant_initializer(1))
# adjust the format(shape or dimensionality) of the Matrix  M to meet the requirements of Tensorflow
M = np.asarray(M, dtype='float32')
M = M.reshape(1, 3, 3, 1)

# compute the result
x = tf.placeholder('float32', [1, None, None, 1])
"""
    :param
    input: given an input tensor of shape [batch, in_height, in_weight, in_channels]
    and a filter/kernel tensor of shape [filter_height, filter_weight, in_channels, out_channels]
"""
conv1 = tf.nn.conv2d(x, filter_weight, strides=[1, 2, 2, 1], padding="SAME")
conv2 = tf.nn.conv2d(x, filter_weight, strides=[1, 2, 2, 1], padding="VALID")
bias = tf.nn.bias_add(conv2, biases)
pool = tf.nn.avg_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")
with tf.Session() as sess:
    tf.global_variables_initializer().run()
    convoluted_M = sess.run(bias, feed_dict={x: M})
    pooled_M = sess.run(pool, feed_dict={x: M})
    bias = sess.run(bias, feed_dict={x: M})
    print("bias is:\n", bias.shape)

    print("hello word!")
    # print("convoluted_M:\n", convoluted_M.shape)
    # print("pooled_M:\n", pooled_M)

