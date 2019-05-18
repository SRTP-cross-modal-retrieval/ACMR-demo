import tensorflow as tf
import os
import numpy as np
import tflearn

slim = tf.contrib.slim
    
def vgg16(input_images):
    with tf.variable_scope('vgg_16', 'vgg_16', [input_images], reuse=None):
        with slim.arg_scope([slim.conv2d, slim.fully_connected],
                          activation_fn=tf.nn.relu,
                          weights_initializer=tf.truncated_normal_initializer(0.0, 0.01),
                          weights_regularizer=slim.l2_regularizer(0.0005)):
            net = slim.repeat(input_images, 2, slim.conv2d, 64, [3, 3], scope='conv1')
            net = slim.max_pool2d(net, [2, 2], scope='pool1')
            net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
            net = slim.max_pool2d(net, [2, 2], scope='pool2')
            net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope='conv3')
            net = slim.max_pool2d(net, [2, 2], scope='pool3')
            net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv4')
            net = slim.max_pool2d(net, [2, 2], scope='pool4')
            net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv5')
            net = slim.max_pool2d(net, [2, 2], scope='pool5')
            net = slim.fully_connected(net, 4096, scope='fc6')
            net = slim.dropout(net, 0.5, scope='dropout6')
            feature = slim.fully_connected(net, 4096, scope='fc7')
            net = slim.dropout(feature, 0.5, scope='dropout7')
            net = slim.fully_connected(net, 10575, activation_fn=None, scope='fc8')#10575 classes
    return net, feature

tf.reset_default_graph()
x = tf.placeholder(tf.float32, [None,112, 96, 3])
net, feature = vgg16(x)
train_writer = tf.summary.FileWriter('board', tf.get_default_graph())
variables_to_restore = slim.get_variables_to_restore()#exclude=exclude_list
slim.assign_from_checkpoint_fn(
            'PreTrainModel/vgg_16.ckpt',
            variables_to_restore,
            ignore_missing_vars=True)




