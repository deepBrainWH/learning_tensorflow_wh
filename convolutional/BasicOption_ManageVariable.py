# -*- coding: utf-8 -*-
'''
    python file describe: Managing variable in tensorflow
'''
import tensorflow as tf

with tf.variable_scope("foo"):
    v = tf.get_variable("v1", shape=[3, 3, 1, 1],
                        initializer=tf.truncated_normal_initializer(0, 0.1))
with tf.variable_scope("foo", reuse=True):
    v1 = tf.get_variable("v1", shape=[3, 3, 1, 1])

print(v.name)
print(v1.name)

with tf.variable_scope("", reuse=True):
    v2 = tf.get_variable("foo/v1", [3, 3, 1, 1])
    print(v2.name)
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

print(v2.eval())
sess.close()