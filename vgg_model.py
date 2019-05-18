import tensorflow as tf
import os
import numpy as np
import tflearn

slim = tf.contrib.slim
    
def vgg16(input_images):
    with tf.variable_scope('vgg_16', 'vgg_16', [input_images], reuse=None):#第一个参数是scope名 第二个是名字
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
            #以上，卷积池化结束
            
            #标准vgg16这里有没有一个展开的层（就是flatten）我不太清楚  
            
            #以下全连接
            net = slim.fully_connected(net, 4096, scope='fc6')
            #这里我认为不能丢弃了 已经训练好了
            feature = slim.fully_connected(net, 4096, scope='fc7')  #这里是feature
            net = slim.fully_connected(feature, 1000, activation_fn=None, scope='fc8')#1000 classes 这里不重要
    return net, feature

with tf.Graph().as_default() as graph:
    tf.reset_default_graph()


    x = tf.placeholder(tf.float32, [None,112, 96, 3])  #图片形状
    net, feature = vgg16(x)
    train_writer = tf.summary.FileWriter('board', tf.get_default_graph())#可视化网络结构

    variables_to_restore = slim.get_variables_to_restore()#exclude=exclude_list这里可以加不想恢复的某些参数
    saver = tf.train.Saver(variables_to_restore)
    with tf.Session() as sess:
        
        saver.restore(sess, 'PreTrainModel/vgg_16.ckpt')#恢复vgg参数 这个参数我有 我得发给你
        
        #读数据
        

        #run feature 把数据塞到x里


        train_writer.close()



