#coding=utf-8
import pickle
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from models.flip_gradient import flip_gradient

with open('/home/media/Downloads/ACMR_demo_master/data/wikipedia_dataset/train_img_feats.pkl', 'rb') as f:
    train_img_feats = pickle.load(f)
with open('/home/media/Downloads/ACMR_demo_master/data/wikipedia_dataset/train_txt_vecs.pkl', 'rb') as f:
    train_txt_vecs = pickle.load(f)
with open('/home/media/Downloads/ACMR_demo_master/data/wikipedia_dataset/train_labels.pkl', 'rb') as f:
    train_labels = pickle.load(f)
with open('/home/media/Downloads/ACMR_demo_master/data/wikipedia_dataset/test_img_feats.pkl', 'rb') as f:
    test_img_feats = pickle.load(f)
with open('/home/media/Downloads/ACMR_demo_master/data/wikipedia_dataset/test_txt_vecs.pkl', 'rb') as f:
    test_txt_vecs = pickle.load(f)
with open('/home/media/Downloads/ACMR_demo_master/data/wikipedia_dataset/test_labels.pkl', 'rb') as f:
    test_labels = pickle.load(f)

a = np.zeros(6)
a[0] = len(train_img_feats)
a[1] = len(train_txt_vecs)
a[2] = len(train_labels)
a[3] = len(test_img_feats)
a[4] = len(test_txt_vecs)
a[5] = len(test_labels)

print (a)

def visual_feature_embed(X, is_training=True, reuse=False):
    #slim.arg_scope可以定义一些函数的默认参数值，在scope内，我们重复用到这些函数时可以不用把所有参数都写一遍
    with slim.arg_scope([slim.fully_connected], activation_fn=None, reuse=reuse):
        net = tf.nn.tanh(slim.fully_connected(X, 512, scope='vf_fc_0'))
        net = tf.nn.tanh(slim.fully_connected(net, 100, scope='vf_fc_1'))
        net = tf.nn.tanh(slim.fully_connected(net, 40, scope='vf_fc_2'))
    return net

def label_embed(L, is_training=True, reuse=False):
    with slim.arg_scope([slim.fully_connected], activation_fn=None, reuse=reuse):
        net = tf.nn.tanh(slim.fully_connected(L, 40, scope='le_fc_0'))
        net = tf.nn.tanh(slim.fully_connected(net, 100, scope='le_fc_1'))
        net = tf.nn.tanh(slim.fully_connected(net, 40, scope='le_fc_2'))
    return net

def domain_classifier( E, l, is_training=True, reuse=False):
    with slim.arg_scope([slim.fully_connected], activation_fn=None, reuse=reuse):
        E = flip_gradient(E, l)
        net = slim.fully_connected(E, 20, scope='dc_fc_0')
        net = slim.fully_connected(net, 10, scope='dc_fc_1')
        net = slim.fully_connected(net, 2, scope='dc_fc_2')
    return net



visual_feats = tf.placeholder(tf.float32, [None, 4096])
word_vecs = tf.placeholder(tf.float32, [None, 5000])
y_single = tf.placeholder(tf.int32, [3,1])
emb_v = visual_feature_embed(visual_feats)
emb_w = label_embed(word_vecs)
emb_v_ = tf.reduce_sum(emb_v, axis=1, keep_dims=True)
emb_w_ = tf.reduce_sum(emb_w, axis=1, keep_dims=True)
p = float(1) / 200
l = 2. / (1. + np.exp(-10. * p)) - 1
emb_v_class = domain_classifier(emb_v, l)
emb_w_class = domain_classifier(emb_w, l, reuse=True)


distance_map = tf.matmul(emb_v_,tf.ones([1,3])) - tf.matmul(emb_v,tf.transpose(emb_w))+ \
    tf.matmul(tf.ones([3,1]),tf.transpose(emb_w_))
mask_initial = tf.to_float(tf.matmul(y_single,tf.ones([1,3],dtype=tf.int32)) - \
    tf.matmul(tf.ones([3,1],dtype=tf.int32),tf.transpose(y_single)))
#返回一个布尔型变量,数组中的每个元素都进行比较，与0相等则返回0，与0不等则返回1 ，tf.zeros_like()新建一个与给定的tensor类型大小一致的tensor，其所有元素为0
mask = tf.to_float(tf.not_equal(mask_initial, tf.zeros_like(mask_initial)))
masked_dissimilar_loss = tf.multiply(distance_map,mask)
dissimilar_loss_sample = 0.1*tf.ones_like(mask) - masked_dissimilar_loss
dissimilar_loss = tf.reduce_mean(tf.maximum(0., 0.1*tf.ones_like(mask)-masked_dissimilar_loss))

with tf.Session() as sess:
    tf.global_variables_initializer().run()
    validate_feed = {visual_feats:train_img_feats[0:3], word_vecs:train_txt_vecs[0:3], y_single:np.transpose([train_labels[0:3]])}
    k = sess.run(emb_v, feed_dict=validate_feed)
    l = sess.run(emb_w, feed_dict=validate_feed)
    m = sess.run(emb_v_, feed_dict=validate_feed)
    n = sess.run(emb_w_, feed_dict=validate_feed)
    o = sess.run(distance_map, feed_dict=validate_feed)
    p = sess.run(mask_initial, feed_dict=validate_feed)
    q = sess.run(mask, feed_dict=validate_feed)
    r = sess.run(masked_dissimilar_loss, feed_dict=validate_feed)
    s = sess.run(dissimilar_loss_sample, feed_dict=validate_feed)
    t = sess.run(dissimilar_loss, feed_dict=validate_feed)
    u = sess.run(emb_v_class, feed_dict=validate_feed)
    v = sess.run(emb_w_class, feed_dict=validate_feed)

    print (k, '\n', l, '\n', m, '\n', n, '\n', o, '\n', p, '\n', q, '\n', r, '\n', s, '\n', t, '\n', u, '\n', v)
