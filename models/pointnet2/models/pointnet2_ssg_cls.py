import os
import sys
BASE_DIR = os.path.dirname(__file__)
# sys.path.append(BASE_DIR)
import imp
# sys.path.append(os.path.join(BASE_DIR, '../utils'))
import tensorflow as tf
import numpy as np
tf_util = imp.load_source(
    'tf_util', os.path.join(BASE_DIR, '../utils', "tf_util.py"))
pointnet_util = imp.load_source(
    'pointnet_util', os.path.join(BASE_DIR, '../utils', "pointnet_util.py"))

from pointnet_util import pointnet_sa_module
#import tf_nndistance

def placeholder_inputs(batch_size, num_point):
    pointclouds_pl = tf.placeholder(tf.float32, shape=(batch_size, num_point, 3))
    labels_pl = tf.placeholder(tf.int32, shape=(batch_size))
    return pointclouds_pl, labels_pl


def get_model(point_cloud, is_training, bn_decay=None):
    """" Classification PointNet, input is BxNx3, output Bx40 """
    batch_size = point_cloud.get_shape()[0].value
    num_point = point_cloud.get_shape()[1].value
    end_points = {}
    l0_xyz = point_cloud
    l0_points = None
    end_points['l0_xyz'] = l0_xyz

    # Set abstraction layers
    # Note: When using NCHW for layer 2, we see increased GPU memory usage (in TF1.4).
    # So we only use NCHW for layer 1 until this issue can be resolved.
    l1_xyz, l1_points, l1_indices = pointnet_sa_module(l0_xyz, l0_points, npoint=512, radius=0.2, nsample=32, mlp=[64,64,128], mlp2=None, group_all=False, is_training=is_training, bn_decay=bn_decay, scope='layer1', use_nchw=True)
    l2_xyz, l2_points, l2_indices = pointnet_sa_module(l1_xyz, l1_points, npoint=128, radius=0.4, nsample=64, mlp=[128,128,256], mlp2=None, group_all=False, is_training=is_training, bn_decay=bn_decay, scope='layer2')
    l3_xyz, l3_points, l3_indices = pointnet_sa_module(l2_xyz, l2_points, npoint=None, radius=None, nsample=None, mlp=[256,512,1024], mlp2=None, group_all=True, is_training=is_training, bn_decay=bn_decay, scope='layer3')

    # Fully connected layers
    end_points['post_max'] = l3_points
    net = tf.reshape(l3_points, [batch_size, -1])
    net = tf_util.fully_connected(net, 512, bn=True, is_training=is_training, scope='fc1', bn_decay=bn_decay)
    net = tf_util.dropout(net, keep_prob=0.5, is_training=is_training, scope='dp1')
    net = tf_util.fully_connected(net, 256, bn=True, is_training=is_training, scope='fc2', bn_decay=bn_decay)
    net = tf_util.dropout(net, keep_prob=0.5, is_training=is_training, scope='dp2')
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

def get_critical_points(sess,ops,data,BATCH_SIZE,NUM_ADD,NUM_POINT=1024):

    ####################################################
    ### get the critical point of the given point clouds
    ### data shape: BATCH_SIZE*NUM_POINT*3
    ### return : BATCH_SIZE*NUM_ADD*3
    #####################################################
    sess.run(tf.assign(ops['pert'],tf.zeros([BATCH_SIZE,NUM_ADD,3])))
    is_training=False

    #to make sure init_points is in shape of BATCH_SIZE*NUM_ADD*3 so that it can be fed to initial_point_pl
    if NUM_ADD > NUM_POINT:
        init_points=np.tile(data[:,:2,:],[1,NUM_ADD/2,1]) ## due to the max pooling operation of PointNet, 
                                                          ## duplicated points would not affect the global feature vector   
    else:
        init_points=data[:, :NUM_ADD, :]
    feed_dict = {ops['pointclouds_pl']: data,
                 ops['is_training_pl']: is_training,
                 ops['initial_point_pl']:init_points}
    pre_max_val,post_max_val=sess.run([ops['pre_max'],ops['post_max']],feed_dict=feed_dict)
    pre_max_val = pre_max_val[:,:NUM_POINT,...]
    pre_max_val=np.reshape(pre_max_val,[BATCH_SIZE,NUM_POINT,1024])#1024 is the dimension of PointNet's global feature vector
    
    critical_points=[]
    for i in range(len(pre_max_val)):
        #get the most important critical points if NUM_ADD < number of critical points
        #the importance is demtermined by counting how many elements in the global featrue vector is 
        #contributed by one specific point 
        idx,counts=np.unique(np.argmax(pre_max_val[i],axis=0),return_counts=True)
        idx_idx=np.argsort(counts)
        if len(counts) > NUM_ADD:
            points = data[i][idx[idx_idx[-NUM_ADD:]]]
        else:
            points = data[i][idx]
            tmp_num = NUM_ADD - len(counts)
            while(tmp_num > len(counts)):
                points = np.concatenate([points,data[i][idx]])
                tmp_num-=len(counts)
            points = np.concatenate([points,data[i][-tmp_num:]])
        
        critical_points.append(points)
    critical_points=np.stack(critical_points)
    return critical_points

if __name__=='__main__':
    with tf.Graph().as_default():
        inputs = tf.zeros((32,1024,3))
        inputs2 = tf.zeros((32,122,3))
        outputs = get_model(inputs, tf.constant(True))
        print(outputs)
        dists_forward,_,dists_backward,_ = tf_nndistance.nn_distance(inputs2,inputs)
