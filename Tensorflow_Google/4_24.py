# -*- coding: utf-8 -*-
"""
    python文件说明：引入正则化项
"""
'''
    w = tf.Variable(tf.random_nomal([2, 1], stddev = 1, seed = 1))
    y = tf.matmul(x1, w)
    
    loss = tf.reduce_mean(tf.square(y_ -y)) + tf.contrib.layers.l2_regularizer(lambda)(w)
    
    loss 由两部分组成，第一部分是均方误差，第二部分是是正则化
    
    lambda是正则化的项的权重
    
'''
import tensorflow as tf

def get_weight(shape, my_lambda):
    #生成一个变量
    var = tf.Variable(tf.random_normal(shape, dtype=tf.float32))
    #add_to_collection 函数将生成的这个变量的L2正则化损失项添加到集合中

    tf.add_to_collection('losses', tf.contrib.layers.l2_regularizer(my_lambda)(var))

    #返回生成的变量
    return var

x = tf.placeholder(tf.float32, shape=(None, 32))
y_ = tf.placeholder(tf.float32, shape=(None, 1))
batch_size = 8
#定义每一层网络中的节点个数
layer_dimension = [2, 10, 10, 10, 1]
#层数
n_layer = len(layer_dimension)

#开始的层
cur_layer = x
#当前层的节点数
in_dimension = layer_dimension[0]

#循环生成5层全连接网络
for i in range(1, n_layer):
    #下一层节点个书
    out_dimension = layer_dimension[i]
    #生成当前层中的权重
    weight = get_weight([in_dimension, out_dimension], 0.001)
    bias = tf.Variable(tf.constant(0.1, shape=[out_dimension]))

    cur_layer = tf.nn.relu(tf.matmul(cur_layer, weight) + bias)

    in_dimension = layer_dimension[i]

mse_loss = tf.reduce_mean(tf.square(y_ - cur_layer))

tf.add_to_collection('losses', mse_loss)

loss = tf.add_n(tf.get_collection('losses'))


