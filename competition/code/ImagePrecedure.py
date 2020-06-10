# -*- coding: utf-8 -*-
'''
    python file describe:图像预处理
'''
import cv2
import os
import numpy as np
import tensorflow as tf

class ImagePre:
    def __init__(self, image_path):
        self.image_path = image_path
        self.path = self._get_all_image_filename(image_path)

    def _get_all_image_filename(self, filepath):
        filename_list = []
        for root, dirs, files in os.walk(filepath):
            for fn in files:
                filename_list.append(str(root)+str("\\")+str(fn))
        return filename_list

    def _get_contour(self, image_path):
        image_ = cv2.imread(image_path, 1)
        image = image_[100:900, 200:1000]
        return image

    def getImageTensor(self):
        input_image_tensor = []
        output_image_tensor = []
        for i in range(40):
            if i%2 == 0:
                output_image_tensor.append(self._get_contour(self.path[i]))
            else:
                input_image_tensor.append(self._get_contour(self.path[i]))
        input_image_tensor = np.asarray(input_image_tensor, dtype=np.float32)
        output_image_tensor = np.asarray(output_image_tensor, dtype=np.float32)
        return input_image_tensor, output_image_tensor

class NeuralNetwork:
    def __init__(self, input_tensor, output_tensor):
        self.input_tensor = input_tensor
        self.output_tensor = output_tensor

    def _get_weights(self, name, shape, dtype=tf.float32):
        return tf.get_variable(name=name, shape=shape, dtype=dtype,
                               initializer=tf.truncated_normal_initializer())

    def _get_biases(self, name, shape, dtype=tf.float32):
        return tf.get_variable(name=name, shape=shape, dtype=dtype,
                               initializer=tf.truncated_normal_initializer())

    def _conv(self, input, weight, biase, name):
        conv = tf.nn.conv2d(input, weight, strides=[1, 1, 1, 1], padding='SAME', name=name)
        biase_add = tf.nn.bias_add(conv, biase, name="biase_add_op")
        relu = tf.nn.relu(biase_add)
        return relu

    def _deconv(self, input, weight, biase, output_shape, name):
        deconv = tf.nn.conv2d_transpose(input, weight, output_shape=output_shape,
                                        strides=[1, 2, 2, 1], padding='SAME', name=name)
        biase_add = tf.nn.bias_add(deconv, biase, name="biase_add_op")
        relu = tf.nn.relu(biase_add)
        return  relu

    def nn(self):
        with tf.name_scope("input_layer"):
            x_ = tf.placeholder(tf.float32, shape=[20, 800, 800, 3], name='x_input')
            y_ = tf.placeholder(tf.float32, shape=[20, 800, 800, 3], name='y_input')

        with tf.name_scope("conv1"):# 400*400
            weight1 = self._get_weights("weight1", [5, 5, 3, 8])
            biase1 = self._get_biases("biase1", [8])
            conv1 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(x_, weight1, strides=[1, 1, 1, 1], padding='SAME', name='convolution1'),
                                              biase1, name='add_biase1'), name='relu1')
            pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool1')

        with tf.name_scope("conv2"):# 200*200
            weight2 = self._get_weights("weight2", [5, 5, 8, 16])
            biase2 = self._get_biases("biase2", [16])
            conv2 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(pool1, weight2, strides=[1, 1, 1, 1], padding='SAME', name='convolution2'),
                                              biase2,name='add_biase2'), name='relu2')
            pool2 = tf.nn.max_pool(conv2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME',name='pool2')

        with tf.name_scope("conv3"):# 100*100
            weight3 = self._get_weights('weight3', shape=[5, 5, 16, 32])
            biase3 = self._get_biases('biase3', shape=[32])
            conv3 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(pool2, weight3, strides=[1, 1, 1, 1], padding='SAME', name='convolution3'),
                               biase3, name='add_biase3'), name='relu3')
            pool3 = tf.nn.max_pool(conv3, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool3')

        with tf.name_scope("conv4"):#50*50
            weight4 = self._get_weights('weight4', shape=[3, 3, 32, 32])
            biase4 = self._get_biases('biase4', shape=[32])
            conv4 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(pool3, weight4, strides=[1, 1, 1, 1], padding='SAME', name='convolution4'),
                               biase4, name='add_biase4'), name='relu4')
            pool4 = tf.nn.max_pool(conv4, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool4')

        with tf.name_scope("conv5"):#25*25
            weight5 = self._get_weights('weight5', shape=[5, 5, 32, 64])
            biase5 = self._get_biases('biase5', shape=[64])
            conv5 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(pool4, weight5, strides=[1, 1, 1, 1], padding='SAME', name='convolution5'),
                               biase5, name='add_biase5'), name='relu5')
            pool5 = tf.nn.max_pool(conv5, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool5')

        """
        :param 卷积层由这五层构成， 下面设计反卷积层
        """
        with tf.name_scope("de-conv1"):
            d_weight1 = self._get_weights("d_weight1", [5, 5, 32, 64])
            d_biase1 = self._get_biases("d_biase1", [32])
            d_conv1 = self._deconv(pool5, d_weight1, d_biase1, output_shape=[20, 50, 50, 32], name="d_conv1")

        with tf.name_scope("de_conv2"):
            d_weight2 = self._get_weights("d_weight2", [5, 5, 16, 32])
            d_biase2 = self._get_biases("d_biase2", [16])
            d_conv2 = self._deconv(d_conv1, d_weight2, d_biase2, output_shape=[20, 100, 100, 16], name="d_conv2")

        with tf.name_scope("de_conv3"):
            d_weight3 = self._get_weights("d_weight3", shape=[5, 5, 8, 16])
            d_biase3 = self._get_biases("d_biase3", shape=[8])
            d_conv3 = self._deconv(d_conv2, d_weight3, d_biase3, [20, 200, 200, 8], name="d_conv3")

        with tf.name_scope("de_conv4"):
            d_weight4 = self._get_weights("d_weight4", shape=[5, 5, 4, 8])
            d_biase4 = self._get_biases("d_biase4", shape=[4])
            d_conv4 = self._deconv(d_conv3, d_weight4, d_biase4, [20, 400, 400, 4], name="d_conv4")

        with tf.name_scope("de_conv5"):
            d_weight5 = self._get_weights("d_weight5", shape=[5, 5, 1, 4])
            d_biase5 = self._get_biases("d_biase5", shape=[1])
            d_conv5 = self._deconv(d_conv4, d_weight5, d_biase5, [20, 800, 800, 1], name="d_conv5")

        """
        卷积层设计完成，接下来设计训练层
        """
        with tf.name_scope("loss"):
            loss = tf.reduce_mean(tf.abs(y_-d_conv5))
            tf.summary.scalar("loss", loss)

        merged = tf.summary.merge_all()
        train = tf.train.AdadeltaOptimizer(0.001).minimize(loss)

        with tf.Session() as sess:
            tf.global_variables_initializer().run()
            train_writer = tf.summary.FileWriter("../train_log", sess.graph)
            for i in range(10):
                loss_value, _ = sess.run([merged, train], feed_dict={x_: self.input_tensor, y_: self.output_tensor})
                train_writer.add_summary(loss_value, i)
                print("epoch %d, the loss value is: %d" % (i, sess.run(loss, feed_dict={x_: self.input_tensor, y_: self.output_tensor})))


image_pre = ImagePre(r"C:\Users\wangheng\Desktop\6.22yc\6.22yc\2cm")
input_tensor, output_tensor = image_pre.getImageTensor()

NN = NeuralNetwork(input_tensor, output_tensor)
NN.nn()

# print(input_tensor.shape)
# image = np.asarray(input_tensor[12, :, :, :], dtype='uint8')
# cv2.imshow("image", image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
