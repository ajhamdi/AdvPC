from scipy.spatial import distance as set_distance
import os
import imageio
import numpy as np
import pptk
import imp
import glob
import tensorflow as tf 
from sklearn.utils.extmath import softmax
from latent_3d_points.external.structural_losses.tf_nndistance import nn_distance
from latent_3d_points.external.structural_losses.tf_approxmatch import approx_match, match_cost


def tf_norm_projection(x, norm, norm_type="l2", reduction_axis=[1, 2]):
    """
    performs norm projection according to norm and norm type [ l2 or linfty] using tensor flow
    """
    if norm_type == "l2":
        c_norm = tf.sqrt(tf.reduce_sum(tf.square(x),reduction_axis))
        condition = tf.greater(c_norm , tf.reduce_mean(norm,reduction_axis))
        x_normalized = tf.where(condition, norm * tf.nn.l2_normalize(x, axis=reduction_axis) , x)
    elif norm_type == "linfty":
        x_normalized =  tf.clip_by_value(x, clip_value_min=-norm, clip_value_max=norm)
    return x_normalized 


def SRS(points, percentage=0.1):
    new_batch = np.zeros(points.shape)
    for j in range(points.shape[0]):
        new = None
        n = int(round(points.shape[1] * percentage))
        idx = np.arange(points.shape[1])
        np.random.shuffle(idx)
        new = np.delete(points[j], idx[:n], 0)
        s = points.shape[1]-n
        if n == 0:
            new_batch[j] = new
            continue
        n_inx = np.random.randint(low=0, high=s, size=n)
        #print(len(n_inx))
        for i in range(len(n_inx)):
            new = np.append(new, [new[n_inx[i]]], axis=0)
        new_batch[j] = new
    return new_batch
def SOR(points, alpha = 1.1, k = 3):
    new_batch = np.zeros(points.shape)
    for j in range(points.shape[0]):
        # Distances ||Xi - Xj||
        dist = set_distance.squareform(set_distance.pdist(points[j]))
        # Closest points
        closest = np.argsort(dist, axis=1)
        # Choose k neighbors
        dist_k = [sum(dist[i, closest[i,1:k]])/(k-1) for i in range(points.shape[1])]
        # Mean and standard deviation
        di = np.mean(dist_k) + alpha * np.std(dist_k)
        # Only points that have lower distance than di
        list_idx = [i for i in range(len(dist_k)) if dist_k[i] < di]
        if len(list_idx) > 0 :
        # Concatenate the new and the old indexes
            idx = np.concatenate((np.random.choice(list_idx, (points.shape[1]-np.unique(list_idx).shape[0])), list_idx))
            # New points
            new = np.array([points[j][idx[i]] for i in range(len(idx))])
        else :
            new = points[j]
        new_batch[j] = new
    return new_batch
def chamfer_distance(set_1,set_2):
        with tf.Graph().as_default():
            sess = tf.Session()
            pointclouds_input_1 = tf.placeholder(tf.float32,shape=(set_1.shape[0],set_1.shape[1],set_1.shape[2]))
            pointclouds_input_2 = tf.placeholder(tf.float32,shape=(set_2.shape[0],set_1.shape[1],set_1.shape[2]))
            cost_p1_p2, _, cost_p2_p1, _ = nn_distance(pointclouds_input_1, pointclouds_input_2)
            distances = tf.reduce_mean(cost_p1_p2,axis=1) + tf.reduce_mean(cost_p2_p1,axis=1)
        return sess.run(distances,{pointclouds_input_1:set_1,pointclouds_input_2:set_2})

def emd_distance(set_1, set_2):
        with tf.Graph().as_default():
            sess = tf.Session()
            pointclouds_input_1 = tf.placeholder(tf.float32,shape=(set_1.shape[0],set_1.shape[1],set_1.shape[2]))
            pointclouds_input_2 = tf.placeholder(tf.float32,shape=(set_2.shape[0],set_1.shape[1],set_1.shape[2]))
            match = approx_match(pointclouds_input_1, pointclouds_input_1)
            distances = match_cost(pointclouds_input_1,pointclouds_input_2, match)
        return sess.run(distances,{pointclouds_input_1:set_1,pointclouds_input_2:set_2})

def evaluate_ptc(attacked_data,model,model_path,verbose=True):
    GPU_INDEX = 0
#     vl=tf.global_variables()
#     print(20*"B",vl)
    tf.reset_default_graph()
#     vl=tf.global_variables()
#     print(20*"A",vl)
#     MODEL = importlib.import_module(model) # import network module
    MODEL = imp.load_source(model,model) # import network module
    MODEL_PATH = model_path
    with tf.Graph().as_default() as g:
        with tf.device('/gpu:'+str(GPU_INDEX)):
            pointclouds_input, labels_pl = MODEL.placeholder_inputs(attacked_data.shape[0], attacked_data.shape[1])
            is_training_pl = tf.constant(False,tf.bool, shape=())
         
            pred, end_points = MODEL.get_model(pointclouds_input, is_training_pl)
           
            vl=tf.global_variables()
            saver = tf.train.Saver(vl)

        # Create a session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        #config.log_device_placement = True
        sess = tf.Session(graph=g,config=config)
        sess.run(tf.global_variables_initializer())


        ops = {'pointclouds_input': pointclouds_input,
               'pred': pred,
               }

        saver.restore(sess,MODEL_PATH)

    feed_dict = {ops['pointclouds_input']: attacked_data}

    pred_val = sess.run(ops['pred'], feed_dict=feed_dict)
    pred_cls = np.argmax(pred_val, 1)
    pred_scr = np.max(softmax(pred_val), 1)

    if verbose:
        print('model restored!')
        print("the predicted classes and scores:  ",pred_cls.squeeze(), pred_val)
    return pred_cls
# def evaluate_one_batch(sess,ops,attacked_data):
    # is_training = False



def down_sample_ptc(points,target):
    if target > points.shape[0]:
        return points
    import random
    return np.array(random.sample(list(points),target))
def down_sample_ptc_batch(points_batch,target):
    down_sampled_batch = []
    for ii in range(points_batch.shape[0]):
        down_sampled_batch.append(down_sample_ptc(points_batch[ii,...],target))
    return np.array(down_sampled_batch)
def up_sample_ptc(points,target):
    return np.repeat(points,target,axis=0)
def up_sample_ptc_batch(points_batch,target):
    up_sampled_batch = []
    for ii in range(points_batch.shape[0]):
        up_sampled_batch.append(up_sample_ptc(points_batch[ii,...],target))
    return np.array(up_sampled_batch)
