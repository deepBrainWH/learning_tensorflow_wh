# -*- coding: utf-8 -*-
'''
    python file describe:队列操作
'''
import tensorflow as tf

q = tf.FIFOQueue(5, dtypes='int32')
init = q.enqueue_many(([0, 1, 2, 3, 4],))
x = q.dequeue()
y = x+1
q_inc = q.enqueue([y])

with tf.Session() as sess:
    init.run()
    for _ in range(10):
        print(sess.run([x, q_inc]))


