# -*- coding: utf-8 -*-
"""
    python文件说明：加载之前保存的模型---1
"""
import tensorflow as tf
v3 = tf.Variable(tf.constant(4.0, shape=[1], name='v3'))
v4 = tf.Variable(tf.constant(5.0, shape=[1], name='v4'))
result = v3+v4
saver = tf.train.Saver()

with tf.Session() as sess:
    saver.restore(sess, 'model/model.ckpt')
    print(sess.run(result))