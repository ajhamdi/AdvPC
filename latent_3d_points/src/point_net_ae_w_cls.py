'''
Created on January 26, 2017

@author: optas
'''

import sys
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# print(20*"@", os.path.join(BASE_DIR))
sys.path.append(BASE_DIR)
# sys.path.append(os.path.join(BASE_DIR, '..', '..', 'models'))
sys.path.append("/home/hamdiaj/notebooks/learning_torch/data/3d-adv-pc/utils")

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

        AutoEncoder.__init__(self, name, graph, configuration)
        # print(20*"#", c.hard_bound)
        with tf.variable_scope(name):
            self.bound_ball = tf.placeholder(shape=[c.batch_size, c.n_input[0],3], dtype=tf.float32)
            self.pert = tf.get_variable(name='pert', shape=[
                                        c.batch_size, c.n_input[0], 3], initializer=tf.truncated_normal_initializer(stddev=0.01))
            if c.hard_bound:
                self.pert_ = tf.clip_by_value(
                    self.pert, clip_value_min=-self.bound_ball, clip_value_max=self.bound_ball)
            else:
                self.pert_ = self.pert
            self.x_h = self.x+self.pert_
            self.z = c.encoder(self.x_h, **c.encoder_args)
            self.bottleneck_size = int(self.z.get_shape()[1])
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
        input_image = tf.expand_dims(point_cloud_transformed, -1)

        net = tf_util.conv2d(input_image, 64, [1, 3],
                            padding='VALID', stride=[1, 1],
                            bn=True, is_training=is_training,
                            scope='conv1', bn_decay=bn_decay)
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
        end_points['post_max'] = net
        #print("after maxpool")
        #print(net.get_shape())
        net = tf.reshape(net, [batch_size, -1])
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
