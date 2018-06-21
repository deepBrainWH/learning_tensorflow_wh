# -*- coding: utf-8 -*-
"""
    python文件说明：LeNet-5卷积网络模型
"""
import tensorflow as tf
import tensorflow.examples.tutorials.mnist as input_data


INPUT_NODE=784
OUTPUT_NODE=10

IMAGE_SIZE=28
NUM_CHANNELS=1
NUM_LABELS=10

# 第一层卷积层的深度和尺寸
CONV1_SIZE=5
CONV1_DEEP=32

#第二层卷基层的深度和尺寸
CONV2_SIZE=5
CONV2_DEEP=64
#全连接层节点个数
FC_SIZE=512
# 定义卷积神经网络向前传播过程
def inference(input_tensor, train, regularizer):
    with tf.variable_scope('layer1-conv1'):
        conv1_weights = tf.get_variable('weight', [CONV1_SIZE, CONV1_SIZE, NUM_CHANNELS, CONV1_DEEP],
                                        initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv1_biases = tf.get_variable('bias', [CONV1_DEEP], initializer=tf.constant_initializer(0.0))
        #使用边长为5深度为32 的过滤器 过滤器移动步长为1，SAME填充方法
        conv1 = tf.nn.conv2d(input_tensor, conv1_weights, strides=[1, 1, 1, 1], padding='SAME')
        relu1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_biases))
    with tf.name_scope('layer2-pool1'):
        pool1 = tf.nn.max_pool(relu1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    with tf.variable_scope('layer3-conv2'):
        conv2_weights = tf.get_variable('weight', [CONV2_SIZE, CONV2_SIZE, CONV1_DEEP, CONV2_DEEP],
                                        initializer=tf.truncated_normal_initializer(stddev=0.1))
        conv2_biases = tf.get_variable('bias', [CONV2_DEEP], initializer=tf.constant_initializer(0.0))

        #使用过滤器
        conv2 = tf.nn.conv2d(pool1, conv2_weights, strides=[1, 2, 2, 1], padding='SAME')
        relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_biases))
    with tf.name_scope('layer4-pool2'):
        pool2 = tf.nn.max_pool(relu2, ksize=[1, 2, 2, 1], strides=[1, 1, 1, 1], padding='SAME')

    #将第四层池华输出转换为第五层的全连接输入层
    pool_shape = pool2.get_shape().as_list()

    nodes = pool_shape[1] * pool_shape[2] * pool_shape[3]

    reshaped = tf.reshape(pool2, [pool_shape[0], nodes])

    with tf.variable_scope('layer5-fc1'):
        fc1_weights = tf.get_variable('weight', [nodes, FC_SIZE],
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))
        if regularizer != None:
            tf.add_to_collection('losses', regularizer(fc1_weights))
        fc1_biases = tf.get_variable('bias', [FC_SIZE], initializer=tf.constant_initializer(0.1))

        fc1 = tf.nn.relu(tf.matmul(reshaped, fc1_weights) + fc1_biases)
        if train:fc1 = tf.nn.dropout(fc1, 0.5)

    with tf.variable_scope('layer6-fc1'):
        fc2_weights = tf.get_variable('weight', [FC_SIZE, NUM_LABELS],
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))
        fc2_biases = tf.get_variable('bias', [NUM_LABELS], initializer=tf.constant_initializer(0.1))

        logit = tf.matmul(fc1, fc2_weights) + fc2_biases
    return logit

def evaluate(mnist):
    with tf.Graph().as_default() as g:
        x = tf.placeholder(tf.float32, [None,784], name='x-input')
        y_ = tf.placeholder(tf.float32, [None, 10], name='y-input')
        x_image = tf.reshape(x, [-1, 28, 28, 1])
        regularizer = None
        y = inference(x_image, False, regularizer=None)






