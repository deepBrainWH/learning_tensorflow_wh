# -*- coding: utf-8 -*-
'''
    python file describe:卷积网络再图像中的应用实际操作
'''
import tensorflow as tf
import numpy as np
import cv2

FILENAME=r"C:\Users\wangheng\Desktop\25359.jpg"
image = cv2.imread(FILENAME, flags=0)
print(type(image))
cv2.imshow("image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
image_array = np.asarray(image, dtype='float32')
input_tensor = image_array.reshape(1, image.shape[0], image.shape[1], 1)

weights = tf.get_variable("weights", shape=[5, 5, 1, 1],
                          initializer=tf.constant_initializer([[0.01, 0.02, 0.01, 0.002, 0.02],
                                                               [0.002, 0.001, 0.01, 0.002, 0],
                                                               [0.03, 0.01, 0.009, 0.8, 0.01],
                                                               [0.03, 0.003, 0.0021, 0, 0.019],
                                                               [0.003, 0.01, 0.002, 0.004, 0.009]]))
biases = tf.get_variable("biases", shape=[1], initializer=tf.constant_initializer(1))

x_ = tf.placeholder('float32', shape=[1, None, None, 1])
conv = tf.nn.conv2d(x_, weights, strides=[1, 1, 1, 1], padding='SAME')
res = tf.nn.bias_add(conv, biases)
pool = tf.nn.max_pool(res, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

with tf.Session() as sess:
    tf.global_variables_initializer().run()
    conv_res = sess.run(pool, feed_dict={x_: input_tensor})
    print(conv_res.shape)
    print(type(conv_res))
    conv_res = conv_res.reshape(conv_res.shape[1], conv_res.shape[2], 1)
    conv_res = np.asarray(conv_res, dtype='uint8')
    print(conv_res)
    for i in range(conv_res.shape[0]):
        for j in range(conv_res.shape[1]):
            if (conv_res[i, j]<=0) | (conv_res[i, j]>=255):
                print("%d and %d is failed!"%(i,j))
    cv2.imshow("image", conv_res)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

