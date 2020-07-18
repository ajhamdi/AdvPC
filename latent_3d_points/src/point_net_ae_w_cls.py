'''
Created on January 26, 2017

@author: optas
'''

import sys
import os
import imp
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# print(20*"@", os.path.join(BASE_DIR))
sys.path.append(BASE_DIR)
# sys.path.append(os.path.join(BASE_DIR, '..', '..', 'models'))
sys.path.append("/home/hamdiaj/notebooks/learning_torch/data/3d-adv-pc/utils")
sys.path.append("/home/hamdiaj/pointcloudattacks/utils")

# from transform_nets import input_transform_net, feature_transform_net
import os.path as osp
import time
from tflearn.layers.conv import conv_1d
from tflearn.layers.core import fully_connected
import tf_util

import tensorflow as tf
import numpy as np
import math


# import tf_nndistance   ############################################################# HERER UNCOMMENT

from . in_out import create_dir
from . autoencoder_w_cls import AutoEncoder
from . general_utils import apply_augmentations

try:    
    print("HERE###############")
    from .. external.structural_losses.tf_nndistance import nn_distance
    from .. external.structural_losses.tf_approxmatch import approx_match, match_cost
except:
    print('External Losses (Chamfer-EMD) cannot be loaded. Please install them first.')

class PointNetAutoEncoderWithClassifier(AutoEncoder):
    '''
    An Auto-Encoder for point-clouds.
    '''

    def __init__(self, name, configuration, graph=None):
        c = configuration
        self.configuration = c
        self.models = {}
        AutoEncoder.__init__(self, name, graph, configuration)
        # print(20*"#", c.hard_bound)
        with tf.variable_scope(name):
            self.bound_ball_infty = tf.placeholder(shape=[c.batch_size, c.n_input[0],3], dtype=tf.float32)
            self.bound_ball_two = tf.placeholder(
                shape=[c.batch_size, c.n_input[0], 3], dtype=tf.float32)

            self.pert = tf.get_variable(name='pert', shape=[
                                        c.batch_size, c.n_input[0], 3], initializer=tf.truncated_normal_initializer(stddev=0.01))
            if c.hard_bound_mode == 1:
                print("bound L infty")
                self.pert_ = tf_util.tf_norm_projection(self.pert, norm=c.u_infty, norm_type="linfty")
                # self.pert_ = tf.clip_by_value(self.pert, clip_value_min=-c.u_infty, clip_value_max=c.u_infty)
            elif c.hard_bound_mode == 2:
                print("bound L 2")
                self.pert_ = tf_util.tf_norm_projection(
                    self.pert, norm=self.bound_ball_two, norm_type="l2")
            else:
                self.pert_ = self.pert
            if c.dyn_bound_mode == 1:
                self.pert_ = tf_util.tf_norm_projection(
                    self.pert_, norm=self.bound_ball_infty, norm_type="linfty")
                # self.pert_ = tf.clip_by_value(
                #     self.pert_, clip_value_min=-self.bound_ball_infty, clip_value_max=self.bound_ball_infty)
            elif c.dyn_bound_mode == 2:
                self.pert_ = tf_util.tf_norm_projection(
                    self.pert_, norm=self.bound_ball_two, norm_type="l2")

            self.x_h = self.x+self.pert_
            self.z = c.encoder(self.x_h, **c.encoder_args)
            self.bottleneck_size = int(self.z.get_shape()[1])
            if c.use_ae:
                layer = c.decoder(self.z, **c.decoder_args)
                
                if c.exists_and_is_not_none('close_with_tanh'):
                    layer = tf.nn.tanh(layer)
                

                self.x_reconstr = tf.reshape(layer, [-1, self.n_output[0], self.n_output[1]])
                # print("@"*40, [x.name for x in tf.global_variables()
                #                if x.name != 'single_class_ae/pert:0'])
                self.saver = tf.train.Saver(
                    [x for x in tf.global_variables()
                    if 'pert' not in x.name and "single_class_ae" in x.name and "clip_by_value" not in x.name], max_to_keep=c.saver_max_to_keep)
            # self.pred, self.end_points = self.get_model_w_ae(
            #     is_training=tf.constant(False, dtype=tf.bool))

                self._create_loss()
                self._setup_optimizer()

            # GPU configuration
            if hasattr(c, 'allow_gpu_growth'):
                growth = c.allow_gpu_growth
            else:
                growth = True

            config = tf.ConfigProto()
            config.gpu_options.allow_growth = growth

            # Summaries
            self.merged_summaries = tf.summary.merge_all()
            self.train_writer = tf.summary.FileWriter(osp.join(configuration.train_dir, 'summaries'), self.graph)

            # Initializing the tensor flow variables
            self.init = tf.global_variables_initializer()

            # Launch the session
            self.sess = tf.Session(config=config)
            self.sess.run(self.init)

    def _create_loss(self):
        c = self.configuration

        if c.loss == 'chamfer':
            cost_p1_p2, _, cost_p2_p1, _ = nn_distance(self.x_reconstr, self.gt)
            self.loss = tf.reduce_mean(cost_p1_p2) + tf.reduce_mean(cost_p2_p1)
        elif c.loss == 'emd':
            match = approx_match(self.x_reconstr, self.gt)
            self.loss = tf.reduce_mean(match_cost(self.x_reconstr, self.gt, match))

        reg_losses = self.graph.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        if c.exists_and_is_not_none('w_reg_alpha'):
            w_reg_alpha = c.w_reg_alpha
        else:
            w_reg_alpha = 1.0

        for rl in reg_losses:
            self.loss += (w_reg_alpha * rl)

    def chamfer_distance(self,set_1,set_2):
        cost_p1_p2, _, cost_p2_p1, _ = nn_distance(set_1, set_2)
        return tf.reduce_mean(cost_p1_p2, axis=1) + tf.reduce_mean(cost_p2_p1, axis=1)


    def emd_distance(self, set_1, set_2):
        match = approx_match(set_1, set_2)
        return match_cost(set_1, set_2, match)


    def knn_loss(self,input_point_cloud, K=10):
        adj_matrix = tf_util.pairwise_distance(input_point_cloud)
        nn_idx = tf_util.knn(adj_matrix, k=K)
        knn_distances = tf_util.get_neighbor_distances(
            input_point_cloud, nn_idx=nn_idx, k=K)
        knn_loss_val = tf.reduce_mean(knn_distances, [1, 2])
        return knn_loss_val

    def _setup_optimizer(self):
        c = self.configuration
        self.lr = c.learning_rate
        if hasattr(c, 'exponential_decay'):
            self.lr = tf.train.exponential_decay(c.learning_rate, self.epoch, c.decay_steps, decay_rate=0.5, staircase=True, name="learning_rate_decay")
            self.lr = tf.maximum(self.lr, 1e-5)
            tf.summary.scalar('learning_rate', self.lr)

        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr)
        self.train_step = self.optimizer.minimize(self.loss)

    def _single_epoch_train(self, train_data, configuration, only_fw=False):
        n_examples = train_data.num_examples
        epoch_loss = 0.
        batch_size = configuration.batch_size
        n_batches = int(n_examples / batch_size)
        start_time = time.time()

        if only_fw:
            fit = self.reconstruct
        else:
            fit = self.partial_fit

        # Loop over all batches
        for _ in xrange(n_batches):

            if self.is_denoising:
                original_data, _, batch_i = train_data.next_batch(batch_size)
                if batch_i is None:  # In this case the denoising concern only the augmentation.
                    batch_i = original_data
            else:
                batch_i, _, _ = train_data.next_batch(batch_size)

            batch_i = apply_augmentations(batch_i, configuration)   # This is a new copy of the batch.

            if self.is_denoising:
                _, loss = fit(batch_i, original_data)
            else:
                _, loss = fit(batch_i)

            # Compute average loss
            epoch_loss += loss
        epoch_loss /= n_batches
        duration = time.time() - start_time
        
        if configuration.loss == 'emd':
            epoch_loss /= len(train_data.point_clouds[0])
        
        return epoch_loss, duration

    def gradient_of_input_wrt_loss(self, in_points, gt_points=None):
        if gt_points is None:
            gt_points = in_points
        return self.sess.run(tf.gradients(self.loss, self.x), feed_dict={self.x: in_points, self.gt: gt_points})

    def get_model_w_ae_gcn(self,point_cloud, is_training, bn_decay=None):
        """ Classification PointNet, input is BxNx3, output Bx40 """
        tf_util = imp.load_source('tf_util', os.path.join(os.path.dirname(self.models["test"]), '../utils', "tf_util.py"))
        transform_nets = imp.load_source('transform_nets', os.path.join(os.path.dirname(self.models["test"]), "transform_nets.py"))
        import tf_util
        from transform_nets import input_transform_net
        batch_size = self.configuration.batch_size
        num_point = self.configuration.n_input[0]
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
        net = tf.reshape(net, [batch_size, -1]) 
        end_points['post_max'] = net
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
    
    def get_model_w_ae_pp(self,point_cloud, is_training, bn_decay=None):
        """" Classification PointNet, input is BxNx3, output Bx40 """
        pointnet_util = imp.load_source('pointnet_util', os.path.join(
            os.path.dirname(self.models["test"]), '../utils', "pointnet_util.py"))
        tf_util = imp.load_source('tf_util', os.path.join(os.path.dirname(self.models["test"]), '../utils', "tf_util.py"))
        from pointnet_util import pointnet_sa_module
        batch_size = self.configuration.batch_size
        num_point = self.configuration.n_input[0]
        end_points = {}
        l0_xyz = point_cloud
        l0_points = None
        end_points['l0_xyz'] = l0_xyz

        # Set abstraction layers
        # Note: When using NCHW for layer 2, we see increased GPU memory usage (in TF1.4).
        # So we only use NCHW for layer 1 until this issue can be resolved.
        l1_xyz, l1_points, l1_indices = pointnet_sa_module(l0_xyz, l0_points, npoint=512, radius=0.2, nsample=32, mlp=[
                                                        64, 64, 128], mlp2=None, group_all=False, is_training=is_training, bn_decay=bn_decay, scope='layer1', use_nchw=True)
        l2_xyz, l2_points, l2_indices = pointnet_sa_module(l1_xyz, l1_points, npoint=128, radius=0.4, nsample=64, mlp=[
                                                        128, 128, 256], mlp2=None, group_all=False, is_training=is_training, bn_decay=bn_decay, scope='layer2')
        l3_xyz, l3_points, l3_indices = pointnet_sa_module(l2_xyz, l2_points, npoint=None, radius=None, nsample=None, mlp=[
                                                        256, 512, 1024], mlp2=None, group_all=True, is_training=is_training, bn_decay=bn_decay, scope='layer3')

        # Fully connected layers
        net = tf.reshape(l3_points, [batch_size, -1])
        end_points['post_max'] = net
        net = tf_util.fully_connected(
            net, 512, bn=True, is_training=is_training, scope='fc1', bn_decay=bn_decay)
        net = tf_util.dropout(net, keep_prob=0.5,
                            is_training=is_training, scope='dp1')
        net = tf_util.fully_connected(
            net, 256, bn=True, is_training=is_training, scope='fc2', bn_decay=bn_decay)
        net = tf_util.dropout(net, keep_prob=0.5,
                            is_training=is_training, scope='dp2')
        end_points['final'] = net
        net = tf_util.fully_connected(net, 40, activation_fn=None, scope='fc3')

        return net, end_points

    def get_model_w_ae_p(self, point_cloud, is_training, bn_decay=None):
        """" Classification PointNet, input is BxNx3, output Bx40 """
        pointnet_util = imp.load_source('pointnet_util', os.path.join(os.path.dirname(self.models["test"]),'../utils', "pointnet_util.py"))
        tf_util = imp.load_source('tf_util', os.path.join(
            os.path.dirname(self.models["test"]), '../utils', "tf_util.py"))
        from pointnet_util import pointnet_sa_module, pointnet_sa_module_msg
        batch_size = self.configuration.batch_size
        num_point = self.configuration.n_input[0]
        end_points = {}
        l0_xyz = point_cloud
        l0_points = None

    # Set abstraction layers
        l1_xyz, l1_points = pointnet_sa_module_msg(l0_xyz, l0_points, 512, [0.1, 0.2, 0.4], [16, 32, 128], [
                                                [32, 32, 64], [64, 64, 128], [64, 96, 128]], is_training, bn_decay, scope='layer1', use_nchw=True)
        l2_xyz, l2_points = pointnet_sa_module_msg(l1_xyz, l1_points, 128, [0.2, 0.4, 0.8], [32, 64, 128], [
                                                [64, 64, 128], [128, 128, 256], [128, 128, 256]], is_training, bn_decay, scope='layer2')
        l3_xyz, l3_points, _ = pointnet_sa_module(l2_xyz, l2_points, npoint=None, radius=None, nsample=None, mlp=[
                                                256, 512, 1024], mlp2=None, group_all=True, is_training=is_training, bn_decay=bn_decay, scope='layer3')

        # Fully connected layers
        net = tf.reshape(l3_points, [batch_size, -1])
        end_points['post_max'] = net
        net = tf_util.fully_connected(
            net, 512, bn=True, is_training=is_training, scope='fc1', bn_decay=bn_decay)
        net = tf_util.dropout(net, keep_prob=0.4,
                            is_training=is_training, scope='dp1')
        net = tf_util.fully_connected(
            net, 256, bn=True, is_training=is_training, scope='fc2', bn_decay=bn_decay)
        net = tf_util.dropout(net, keep_prob=0.4,
                            is_training=is_training, scope='dp2')
        end_points['final'] = net
        net = tf_util.fully_connected(net, 40, activation_fn=None, scope='fc3')

        return net, end_points


    def get_model_w_ae(self, input_x, is_training, reuse=False, bn_decay=None):
        """
        Classification PointNet, input is BxNx3, output Bx40
        """
        batch_size = self.configuration.batch_size
        num_point = self.configuration.n_input[0]
        end_points = {}
        # with tf.variable_scope('Classifier', reuse=tf.AUTO_REUSE) as scope:

        with tf.variable_scope('transform_net1') as sc:
            transform = self.input_transform_net(
                input_x,is_training, bn_decay, K=3)
        point_cloud_transformed = tf.matmul(input_x, transform)
        end_points['first'] = point_cloud_transformed

        input_image = tf.expand_dims(point_cloud_transformed, -1)

        net = tf_util.conv2d(input_image, 64, [1, 3],
                            padding='VALID', stride=[1, 1],
                            bn=True, is_training=is_training,
                            scope='conv1', bn_decay=bn_decay)
        end_points['second'] = net
        net = tf_util.conv2d(net, 64, [1, 1],
                            padding='VALID', stride=[1, 1],
                            bn=True, is_training=is_training,
                            scope='conv2', bn_decay=bn_decay)

        with tf.variable_scope('transform_net2') as sc:
            transform = self.feature_transform_net(net, is_training, bn_decay, K=64)
        end_points['transform'] = transform
        net_transformed = tf.matmul(tf.squeeze(net, axis=[2]), transform)
        net_transformed = tf.expand_dims(net_transformed, [2])

        net = tf_util.conv2d(net_transformed, 64, [1, 1],
                            padding='VALID', stride=[1, 1],
                            bn=True, is_training=is_training,
                            scope='conv3', bn_decay=bn_decay)
        net = tf_util.conv2d(net, 128, [1, 1],
                            padding='VALID', stride=[1, 1],
                            bn=True, is_training=is_training,
                            scope='conv4', bn_decay=bn_decay)
        net = tf_util.conv2d(net, 1024, [1, 1],
                            padding='VALID', stride=[1, 1],
                            bn=True, is_training=is_training,
                            scope='conv5', bn_decay=bn_decay)

        #print("before maxpool")
        #print(net.get_shape())
        end_points['pre_max'] = net
        # Symmetric function: max pooling
        net = tf_util.max_pool2d(net, [num_point, 1],
                                padding='VALID', scope='maxpool')
        #print("after maxpool")
        #print(net.get_shape())
        net = tf.reshape(net, [batch_size, -1])
        end_points['post_max'] = net

        #print("after reshape")
        #print(net.get_shape())
        net = tf_util.fully_connected(net, 512, bn=True, is_training=is_training,
                                    scope='fc1', bn_decay=bn_decay)
        net = tf_util.dropout(net, keep_prob=0.7, is_training=is_training,
                            scope='dp1')
        net = tf_util.fully_connected(net, 256, bn=True, is_training=is_training,
                                    scope='fc2', bn_decay=bn_decay)
        net = tf_util.dropout(net, keep_prob=0.7, is_training=is_training,
                            scope='dp2')
        end_points['final'] = net
        net = tf_util.fully_connected(net, 40, activation_fn=None, scope='fc3')

        #print(end_points['pre_max'].get_shape())
        return net, end_points


    def get_adv_loss(self,unscaled_logits, targets, kappa=0):

        with tf.variable_scope('adv_loss'):
            unscaled_logits_shape = tf.shape(unscaled_logits)

            B = unscaled_logits_shape[0]
            K = unscaled_logits_shape[1]

            tlab = tf.one_hot(targets, depth=K, on_value=1., off_value=0.)
            tlab = tf.expand_dims(tlab, 0)
            tlab = tf.tile(tlab, [B, 1])
            real = tf.reduce_sum((tlab) * unscaled_logits, 1)
            other = tf.reduce_max((1 - tlab) * unscaled_logits -
                                (tlab * 10000), 1)
            loss1 = tf.maximum(np.asarray(
                0., dtype=np.dtype('float32')), other - real + kappa)
            return tf.reduce_mean(loss1)
    
    def get_untargeted_adv_loss(self, unscaled_logits, victim, kappa=0):

        with tf.variable_scope('adv_loss'):
            unscaled_logits_shape = tf.shape(unscaled_logits)

            B = unscaled_logits_shape[0]
            K = unscaled_logits_shape[1]

            tlab = tf.one_hot(victim, depth=K, on_value=1., off_value=0.)
            tlab = tf.expand_dims(tlab, 0)
            tlab = tf.tile(tlab, [B, 1])
            # c_targets = tf.reduce_max((1 - tlab) * unscaled_logits - (tlab * 10000), 1)
            c_targets = tf.argmax(
                (1 - tlab) * unscaled_logits - (tlab * 10000), 1)

            c_tlab = tf.one_hot(c_targets, depth=K, on_value=1., off_value=0.)

            real = tf.reduce_sum((c_tlab) * unscaled_logits, 1)
            other = tf.reduce_max((1 - c_tlab) * unscaled_logits -
                                  (c_tlab * 10000), 1)
            loss1 = tf.maximum(np.asarray(
                0., dtype=np.dtype('float32')), other - real + kappa)
            return tf.reduce_mean(loss1)

    def get_untargeted_adv_loss_batch(self, unscaled_logits, victim, kappa=0):

        with tf.variable_scope('adv_loss'):
            unscaled_logits_shape = tf.shape(unscaled_logits)

            B = unscaled_logits_shape[0]
            K = unscaled_logits_shape[1]

            tlab = tf.one_hot(victim, depth=K, on_value=1., off_value=0.)
            # tlab = tf.expand_dims(tlab, 0)
            # tlab = tf.tile(tlab, [B, 1])
            # c_targets = tf.reduce_max((1 - tlab) * unscaled_logits - (tlab * 10000), 1)
            c_targets = tf.argmax(
                (1 - tlab) * unscaled_logits - (tlab * 10000), 1)

            c_tlab = tf.one_hot(c_targets, depth=K, on_value=1., off_value=0.)

            real = tf.reduce_sum((c_tlab) * unscaled_logits, 1)
            other = tf.reduce_max((1 - c_tlab) * unscaled_logits -
                                  (c_tlab * 10000), 1)
            loss1 = tf.maximum(np.asarray(
                0., dtype=np.dtype('float32')), other - real + kappa)
            return tf.reduce_mean(loss1)

    def get_adv_loss_batch(self, unscaled_logits, targets, kappa=0):

        with tf.variable_scope('adv_loss'):
            unscaled_logits_shape = tf.shape(unscaled_logits)

            B = unscaled_logits_shape[0]
            K = unscaled_logits_shape[1]

            tlab = tf.one_hot(targets, depth=K, on_value=1., off_value=0.,axis=-1)
            # tlab = tf.expand_dims(tlab, 0)
            # tlab = tf.tile(tlab, [B, 1])
            real = tf.reduce_sum((tlab) * unscaled_logits, 1)
            other = tf.reduce_max((1 - tlab) * unscaled_logits -
                                  (tlab * 10000), 1)
            loss1 = tf.maximum(np.asarray(
                0., dtype=np.dtype('float32')), other - real + kappa)
            return tf.reduce_mean(loss1)

    def input_transform_net(self,inputs, is_training, bn_decay=None, K=3):
        """ Input (XYZ) Transform Net, input is BxNx3 gray image
            Return:
                Transformation matrix of size 3xK """
        batch_size = self.configuration.batch_size
        num_point = self.configuration.n_input[0]

        input_image = tf.expand_dims(inputs, -1)
        net = tf_util.conv2d(input_image, 64, [1, 3],
                            padding='VALID', stride=[1, 1],
                            bn=True, is_training=is_training,
                            scope='tconv1', bn_decay=bn_decay)
        net = tf_util.conv2d(net, 128, [1, 1],
                            padding='VALID', stride=[1, 1],
                            bn=True, is_training=is_training,
                            scope='tconv2', bn_decay=bn_decay)
        net = tf_util.conv2d(net, 1024, [1, 1],
                            padding='VALID', stride=[1, 1],
                            bn=True, is_training=is_training,
                            scope='tconv3', bn_decay=bn_decay)
        net = tf_util.max_pool2d(net, [num_point, 1],
                                padding='VALID', scope='tmaxpool')

        net = tf.reshape(net, [batch_size, -1])
        net = tf_util.fully_connected(net, 512, bn=True, is_training=is_training,
                                    scope='tfc1', bn_decay=bn_decay)
        net = tf_util.fully_connected(net, 256, bn=True, is_training=is_training,
                                    scope='tfc2', bn_decay=bn_decay)

        with tf.variable_scope('transform_XYZ') as sc:
            assert(K == 3)
            weights = tf.get_variable('weights', [256, 3*K],
                                    initializer=tf.constant_initializer(0.0),
                                    dtype=tf.float32)
            biases = tf.get_variable('biases', [3*K],
                                    initializer=tf.constant_initializer(0.0),
                                    dtype=tf.float32)
            biases += tf.constant([1, 0, 0, 0, 1, 0, 0, 0, 1], dtype=tf.float32)
            transform = tf.matmul(net, weights)
            transform = tf.nn.bias_add(transform, biases)

        transform = tf.reshape(transform, [batch_size, 3, K])
        return transform


    def feature_transform_net(self,inputs, is_training, bn_decay=None, K=64):
        """ Feature Transform Net, input is BxNx1xK
            Return:
                Transformation matrix of size KxK """
        batch_size = self.configuration.batch_size
        num_point = self.configuration.n_input[0]

        net = tf_util.conv2d(inputs, 64, [1, 1],
                            padding='VALID', stride=[1, 1],
                            bn=True, is_training=is_training,
                            scope='tconv1', bn_decay=bn_decay)
        net = tf_util.conv2d(net, 128, [1, 1],
                            padding='VALID', stride=[1, 1],
                            bn=True, is_training=is_training,
                            scope='tconv2', bn_decay=bn_decay)
        net = tf_util.conv2d(net, 1024, [1, 1],
                            padding='VALID', stride=[1, 1],
                            bn=True, is_training=is_training,
                            scope='tconv3', bn_decay=bn_decay)
        net = tf_util.max_pool2d(net, [num_point, 1],
                                padding='VALID', scope='tmaxpool')

        net = tf.reshape(net, [batch_size, -1])
        net = tf_util.fully_connected(net, 512, bn=True, is_training=is_training,
                                    scope='tfc1', bn_decay=bn_decay)
        net = tf_util.fully_connected(net, 256, bn=True, is_training=is_training,
                                    scope='tfc2', bn_decay=bn_decay)

        with tf.variable_scope('transform_feat') as sc:
            weights = tf.get_variable('weights', [256, K*K],
                                    initializer=tf.constant_initializer(0.0),
                                    dtype=tf.float32)
            biases = tf.get_variable('biases', [K*K],
                                    initializer=tf.constant_initializer(0.0),
                                    dtype=tf.float32)
            biases += tf.constant(np.eye(K).flatten(), dtype=tf.float32)
            transform = tf.matmul(net, weights)
            transform = tf.nn.bias_add(transform, biases)

        transform = tf.reshape(transform, [batch_size, K, K])
        return transform
