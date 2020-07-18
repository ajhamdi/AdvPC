import os
import imp
import tensorflow as tf
import numpy as np
import math
import sys
print('GCN')
#del sys.path[0:-1]
BASE_DIR = os.path.dirname(__file__)
# sys.path.append(BASE_DIR)
# sys.path.append(os.path.join(BASE_DIR, '../utils'))
tf_util = imp.load_source(
    'tf_util', os.path.join(BASE_DIR, '../utils', "tf_util.py"))

transform_nets = imp.load_source(
    'transform_nets', os.path.join(BASE_DIR,  "transform_nets.py"))
#sys.path.append(os.path.join(BASE_DIR, '../../utils'))
import tf_util
# print(tf_util.__file__)
import transform_nets
from transform_nets import input_transform_net
# print(transform_nets.__file__)
#print sys.path

def placeholder_inputs(batch_size, num_point):
  pointclouds_pl = tf.placeholder(tf.float32, shape=(batch_size, num_point, 3))
  labels_pl = tf.placeholder(tf.int32, shape=(batch_size))
  return pointclouds_pl, labels_pl


def get_model(point_cloud, is_training, bn_decay=None):
  """ Classification PointNet, input is BxNx3, output Bx40 """
  batch_size = point_cloud.get_shape()[0].value
  num_point = point_cloud.get_shape()[1].value
  end_points = {}
  k = 20

  adj_matrix = tf_util.pairwise_distance(point_cloud)
  nn_idx = tf_util.knn(adj_matrix, k=k)
  edge_feature = tf_util.get_edge_feature(point_cloud, nn_idx=nn_idx, k=k)
  print(adj_matrix, nn_idx, edge_feature)  
  with tf.variable_scope('transform_net1') as sc:
    transform = input_transform_net(edge_feature, is_training, bn_decay, K=3)

  point_cloud_transformed = tf.matmul(point_cloud, transform)
  adj_matrix = tf_util.pairwise_distance(point_cloud_transformed)
  nn_idx = tf_util.knn(adj_matrix, k=k)
  edge_feature = tf_util.get_edge_feature(point_cloud_transformed, nn_idx=nn_idx, k=k)

  net = tf_util.conv2d(edge_feature, 64, [1,1],
                       padding='VALID', stride=[1,1],
                       bn=True, is_training=is_training,
                       scope='dgcnn1', bn_decay=bn_decay)
  net = tf.reduce_max(net, axis=-2, keep_dims=True)
  net1 = net

  adj_matrix = tf_util.pairwise_distance(net)
  nn_idx = tf_util.knn(adj_matrix, k=k)
  edge_feature = tf_util.get_edge_feature(net, nn_idx=nn_idx, k=k)

  net = tf_util.conv2d(edge_feature, 64, [1,1],
                       padding='VALID', stride=[1,1],
                       bn=True, is_training=is_training,
                       scope='dgcnn2', bn_decay=bn_decay)
  net = tf.reduce_max(net, axis=-2, keep_dims=True)
  net2 = net
 
  adj_matrix = tf_util.pairwise_distance(net)
  nn_idx = tf_util.knn(adj_matrix, k=k)
  edge_feature = tf_util.get_edge_feature(net, nn_idx=nn_idx, k=k)  

  net = tf_util.conv2d(edge_feature, 64, [1,1],
                       padding='VALID', stride=[1,1],
                       bn=True, is_training=is_training,
                       scope='dgcnn3', bn_decay=bn_decay)
  net = tf.reduce_max(net, axis=-2, keep_dims=True)
  net3 = net

  adj_matrix = tf_util.pairwise_distance(net)
  nn_idx = tf_util.knn(adj_matrix, k=k)
  edge_feature = tf_util.get_edge_feature(net, nn_idx=nn_idx, k=k)  
  
  net = tf_util.conv2d(edge_feature, 128, [1,1],
                       padding='VALID', stride=[1,1],
                       bn=True, is_training=is_training,
                       scope='dgcnn4', bn_decay=bn_decay)
  net = tf.reduce_max(net, axis=-2, keep_dims=True)
  net4 = net

  net = tf_util.conv2d(tf.concat([net1, net2, net3, net4], axis=-1), 1024, [1, 1], 
                       padding='VALID', stride=[1,1],
                       bn=True, is_training=is_training,
                       scope='agg', bn_decay=bn_decay)
 
  net = tf.reduce_max(net, axis=1, keep_dims=True) 

  # MLP on global point cloud vector
  end_points['post_max'] = net
  net = tf.reshape(net, [batch_size, -1]) 
  net = tf_util.fully_connected(net, 512, bn=True, is_training=is_training,
                                scope='fc1', bn_decay=bn_decay)
  net = tf_util.dropout(net, keep_prob=0.5, is_training=is_training,
                         scope='dp1')
  net = tf_util.fully_connected(net, 256, bn=True, is_training=is_training,
                                scope='fc2', bn_decay=bn_decay)
  net = tf_util.dropout(net, keep_prob=0.5, is_training=is_training,
                        scope='dp2')
  end_points['final'] = net
  net = tf_util.fully_connected(net, 40, activation_fn=None, scope='fc3')

  return net, end_points


def get_adv_loss(unscaled_logits,targets,kappa=0):
    
    with tf.variable_scope('adv_loss'):
        unscaled_logits_shape = tf.shape(unscaled_logits)
        #import pdb; pdb.set_trace()
        B = unscaled_logits_shape[0]
        K = unscaled_logits_shape[1]

        tlab=tf.one_hot(targets,depth=K,on_value=1.,off_value=0.)
        tlab=tf.expand_dims(tlab,0)
        tlab=tf.tile(tlab,[B,1])
        real = tf.reduce_sum((tlab) * unscaled_logits, 1)
        other = tf.reduce_max((1 - tlab) * unscaled_logits -
                              (tlab * 10000), 1)
        loss1 = tf.maximum(np.asarray(0., dtype=np.dtype('float32')), other - real + kappa)
        
        return tf.reduce_mean(loss1)


if __name__=='__main__':
  batch_size = 2
  num_pt = 124
  pos_dim = 3

  input_feed = np.random.rand(batch_size, num_pt, pos_dim)
  label_feed = np.random.rand(batch_size)
  label_feed[label_feed>=0.5] = 1
  label_feed[label_feed<0.5] = 0
  label_feed = label_feed.astype(np.int32)

  # # np.save('./debug/input_feed.npy', input_feed)
  # input_feed = np.load('./debug/input_feed.npy')
  # print input_feed

  with tf.Graph().as_default():
    input_pl, label_pl = placeholder_inputs(batch_size, num_pt)
    pos, ftr = get_model(input_pl, tf.constant(True))
    # loss = get_loss(logits, label_pl, None)

    #with tf.Session() as sess:
    #  sess.run(tf.global_variables_initializer())
    #  feed_dict = {input_pl: input_feed, label_pl: label_feed}
    #  res1, res2 = sess.run([pos, ftr], feed_dict=feed_dict)
    #  print res1.shape
    #  print res1

    #  print res2.shape
    #  print res2












