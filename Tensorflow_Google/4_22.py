# -*- coding: utf-8 -*-
"""
    python文件说明：tf.select(where) 函数和tf.greater函数的用法
"""
import tensorflow as tf
v1 = tf.constant([1.0, 2.0, 3.0, 4.0])
v2 = tf.constant([4.0, 3.0, 2.0, 1.0])

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

print(v1.eval())

sess.close()
