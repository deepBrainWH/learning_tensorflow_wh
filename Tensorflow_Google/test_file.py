# -*- coding: utf-8 -*-
"""
    python文件说明：tensorflow函数测试
"""
import tensorflow as tf
import numpy as np
value = np.array([1,2,3,4,5,6,7,8,8,10,11,12])
a = tf.get_variable('test', shape=[4], initializer=tf.truncated_normal_initializer(stddev=0.1))

with tf.Session() as sess:
    init = tf.initialize_all_variables()
    sess.run(init)
    print(a.name)