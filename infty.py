# print("111111111111111111")
import tensorflow as tf
# print("2222222222222222222")
import numpy as np
import argparse
import importlib
import os
import sys
from latent_3d_points.src.ae_templates import mlp_architecture_ala_iclr_18, default_train_params
from latent_3d_points.src.autoencoder import Configuration as Conf
from latent_3d_points.src.point_net_ae import PointNetAutoEncoder

from latent_3d_points.src.in_out import snc_category_to_synth_id, create_dir, PointCloudDataSet, \
    load_all_point_clouds_under_folder

import os.path as osp
from latent_3d_points.src.tf_utils import reset_tf_graph

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'models'))
# sys.path.append(os.path.join(BASE_DIR, 'utils'))
# Use to save Neural-Net check-points etc.
# Use to save Neural-Net check-points etc.
top_out_dir = osp.join(BASE_DIR, "latent_3d_points", "data")
# Top-dir of where point-clouds are stored.
top_in_dir = osp.join(BASE_DIR, "latent_3d_points", "data",
                      "shape_net_core_uniform_samples_2048")


experiment_name = 'single_class_ae'
n_pc_points = 1024  # 2048                # Number of points per model.
bneck_size = 128                  # Bottleneck-AE size
# Loss to optimize: 'emd' or 'chamfer'             # Bottleneck-AE size
ae_loss = 'chamfer'
train_params = default_train_params()
encoder, decoder, enc_args, dec_args = mlp_architecture_ala_iclr_18(
    n_pc_points, bneck_size)
train_dir = create_dir(osp.join(top_out_dir, experiment_name))
conf = Conf(n_input=[n_pc_points, 3],
            loss=ae_loss,
            training_epochs=train_params['training_epochs'],
            batch_size=train_params['batch_size'],
            denoising=train_params['denoising'],
            learning_rate=train_params['learning_rate'],
            train_dir=train_dir,
            loss_display_step=train_params['loss_display_step'],
            saver_step=train_params['saver_step'],
            z_rotate=train_params['z_rotate'],
            encoder=encoder,
            decoder=decoder,
            encoder_args=enc_args,
            decoder_args=dec_args
            )
conf.experiment_name = experiment_name
conf.held_out_step = 5   # How often to evaluate/print out loss on
# held_out data (if they are provided in ae.train() ).
conf.save(osp.join(train_dir, 'configuration'))

# import tf_nndistance
import joblib

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--model', default='pointnet_cls', help='Model name: pointnet_cls or pointnet_cls_basic [default: pointnet_cls]')
parser.add_argument('--log_dir', default='log', help='Log dir [default: log]')
parser.add_argument('--batch_size', type=int, default=5, help='Batch Size for attack [default: 5]')
parser.add_argument('--num_point', type=int, default=1024, help='Point Number [256/512/1024/2048] [default: 1024]')
parser.add_argument('--data_dir', default='data', help='data folder path [data]')
parser.add_argument('--dump_dir', default='infty', help='dump folder path [perturbation]')

parser.add_argument('--target', type=int, default=5, help='target class index')
parser.add_argument('--lr_attack', type=float, default=0.005, help='learning rate for optimization based attack')
parser.add_argument('--initial_alpha', type=float, default=0.3,help=' natural factor')
parser.add_argument('--gamma', type=float, default=0.2,help='natural factor growth/depreciation rate')
parser.add_argument('--initial_weight', type=float, default=10, help='initial value for the parameter lambda')
parser.add_argument('--upper_bound_weight', type=float, default=40, help='upper_bound value for the parameter lambda')
parser.add_argument('--step', type=int, default=10, help='binary search step')
parser.add_argument('--num_iter', type=int, default=500, help='number of iterations for each binary search step')

FLAGS = parser.parse_args()

load_pre_trained_ae = True
restore_epoch = 500
if load_pre_trained_ae:
    conf = Conf.load(train_dir + '/configuration')
    reset_tf_graph()
    ae = PointNetAutoEncoder(conf.experiment_name, conf)
    ae.restore_model(conf.train_dir, epoch=restore_epoch,verbose=True)

BATCH_SIZE = FLAGS.batch_size
NUM_POINT = FLAGS.num_point
MODEL_PATH = os.path.join(FLAGS.log_dir, "model.ckpt")
GPU_INDEX = FLAGS.gpu
MODEL = importlib.import_module(FLAGS.model) # import network module
DUMP_DIR = FLAGS.dump_dir
if not os.path.exists(DUMP_DIR): os.mkdir(DUMP_DIR)
DATA_DIR = FLAGS.data_dir

TARGET=FLAGS.target
LR_ATTACK=FLAGS.lr_attack
INITIAL_ALPHA = FLAGS.initial_alpha
GAMMA = FLAGS.gamma
#WEIGHT=FLAGS.weight

attacked_data_all=joblib.load(os.path.join(DATA_DIR,'attacked_data.z'))

INITIAL_WEIGHT=FLAGS.initial_weight
UPPER_BOUND_WEIGHT=FLAGS.upper_bound_weight
#ABORT_EARLY=False
BINARY_SEARCH_STEP=FLAGS.step
NUM_ITERATIONS=FLAGS.num_iter

def attack():
    is_training = False
    with tf.Graph().as_default():
        # print("3333333333333333333")
        with tf.device('/gpu:'+str(GPU_INDEX)):
            pointclouds_pl, labels_pl = MODEL.placeholder_inputs(BATCH_SIZE, NUM_POINT)
            is_training_pl = tf.placeholder(tf.bool, shape=())
            # is_projection = tf.placeholder(tf.bool, shape=())

            pert=tf.get_variable(name='pert',shape=[BATCH_SIZE,NUM_POINT,3],initializer=tf.truncated_normal_initializer(stddev=0.01))

            pointclouds_input=pointclouds_pl+pert
            
            pred, end_points = MODEL.get_model(pointclouds_input, is_training_pl)

            #adv loss
            adv_loss=MODEL.get_adv_loss(pred,TARGET)
            
            #perturbation l2 constraint
            pert_norm=tf.sqrt(tf.reduce_sum(tf.square(pert),[1,2]))
            #perturbation l1 constraint
            # pert_norm = tf.reduce_sum(tf.abs(pert), [1, 2])
            #perturbation l_infty constraint
            pert_bound = tf.norm(
                tf.nn.relu(pert-GAMMA), ord=1, axis=(1,2))


            dist_weight=tf.placeholder(shape=[BATCH_SIZE],dtype=tf.float32)
            lr_attack=tf.placeholder(dtype=tf.float32)
            attack_optimizer = tf.train.AdamOptimizer(lr_attack)
            l_2_loss = tf.reduce_mean(tf.multiply(dist_weight, pert_norm))
            l_infty_loss = tf.reduce_mean(tf.multiply(dist_weight, pert_bound))
            total_loss = adv_loss + l_2_loss + (500*GAMMA) *  l_infty_loss
            attack_op = attack_optimizer.minimize(total_loss,var_list=[pert])
            
            vl=tf.global_variables()
            vl=[x for x in vl if 'pert' not in x.name]
            saver = tf.train.Saver(vl)

        # Create a session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        #config.log_device_placement = True
        sess = tf.Session(config=config)
        sess.run(tf.global_variables_initializer())


        ops = {'pointclouds_pl': pointclouds_pl,
               'labels_pl': labels_pl,
               'is_training_pl': is_training_pl,
               'pointclouds_input':pointclouds_input,
               'dist_weight':dist_weight,
               'pert': pert,
               'pre_max':end_points['pre_max'],
               'post_max':end_points['post_max'],
               'pred': pred,
               'adv_loss': adv_loss,
               'pert_norm':pert_norm,
               "pert_bound": pert_bound,
               #'total_loss':tf.reduce_mean(tf.multiply(dist_weight,pert_norm))+adv_loss,
               'lr_attack':lr_attack,
               'attack_op':attack_op
               }

        saver.restore(sess,MODEL_PATH)
        print('model restored!')

        dist_list=[]
        # the class index of selected 10 largest classed in ModelNet40
        for victim in [0,5, 35, 2, 8, 33, 22, 37, 4, 30]:
            if victim == TARGET:
                continue
            attacked_data=attacked_data_all[victim]#attacked_data shape:25*1024*3
            for j in range(25//BATCH_SIZE):
                dist, img = attack_one_batch(
                    sess, ops, attacked_data[j*BATCH_SIZE:(j+1)*BATCH_SIZE], victim)
                dist_list.append(dist)
                np.save(os.path.join('.',DUMP_DIR,'{}_{}_{}_adv.npy' .format(victim,TARGET,j)),img)
                np.save(os.path.join('.',DUMP_DIR,'{}_{}_{}_orig.npy' .format(victim,TARGET,j)),attacked_data[j*BATCH_SIZE:(j+1)*BATCH_SIZE])#dump originial example for comparison
        #joblib.dump(dist_list,os.path.join('.',DUMP_DIR,'dist_{}.z' .format(TARGET)))#log distance information for performation evaluation


def attack_one_batch(sess, ops, attacked_data, victim):
    c_NUM_ITERATIONS = NUM_ITERATIONS
    ###############################################################
    ### a simple implementation
    ### Attack all the data in variable 'attacked_data' into the same target class (specified by TARGET)
    ### binary search is used to find the near-optimal results
    ### part of the code is adpated from https://github.com/tensorflow/cleverhans/blob/master/cleverhans/attacks/carlini_wagner_l2.py
    ###############################################################

    is_training = False

    attacked_label=np.ones(shape=(len(attacked_data)),dtype=int) * TARGET #the target label for adv pcs
    attacked_label=np.squeeze(attacked_label)

    #the bound for the binary search
    lower_bound=np.zeros(BATCH_SIZE)
    WEIGHT = np.ones(BATCH_SIZE) * INITIAL_WEIGHT
    ALPHA = np.ones((BATCH_SIZE, NUM_POINT,3)) * INITIAL_ALPHA

    upper_bound=np.ones(BATCH_SIZE) * UPPER_BOUND_WEIGHT

   
    o_bestdist = [1e10] * BATCH_SIZE
    # o_bestnat = [1e10] * BATCH_SIZE
    o_bestinfty = [1e10] * BATCH_SIZE
    o_bestscore = [-1] * BATCH_SIZE
    o_bestattack = np.ones(shape=(BATCH_SIZE,NUM_POINT,3))

    feed_dict = {ops['pointclouds_pl']: attacked_data,
         ops['is_training_pl']: is_training,
         ops['lr_attack']:LR_ATTACK,
         ops['dist_weight']:WEIGHT}


    for out_step in range(BINARY_SEARCH_STEP):
        if out_step == BINARY_SEARCH_STEP-1:
            c_NUM_ITERATIONS = c_NUM_ITERATIONS * 5


        feed_dict[ops['dist_weight']]=WEIGHT

        sess.run(tf.assign(ops['pert'],tf.truncated_normal([BATCH_SIZE,NUM_POINT,3], mean=0, stddev=0.0000001)))

        bestdist = [1e10] * BATCH_SIZE
        # bestnat = [1e10] * BATCH_SIZE
        bestinfty = [1e10] * BATCH_SIZE

        bestscore = [-1] * BATCH_SIZE  

        prev = 1e6      

        for iteration in range(c_NUM_ITERATIONS):
            # feed_dict.update({ops['pointclouds_pl']: attacked_data})
            _= sess.run([ops['attack_op']], feed_dict=feed_dict)


            # adv_loss_val,dist_val,pred_val,input_val,cur_pert = sess.run([ops['adv_loss'],
            #                                                           ops['pert_norm'], ops['pred'], ops['pointclouds_input'], ops["pert"]], feed_dict=feed_dict)
            # pred_val = np.argmax(pred_val, 1)
            # loss=adv_loss_val+np.average(dist_val*WEIGHT)
            cur_pert = sess.run(ops["pert"], feed_dict=feed_dict)
                        ###########################################################################################3
            mid_val = ae.reconstruct(attacked_data + cur_pert)[0]
            dist_nat = np.linalg.norm(mid_val - (attacked_data + cur_pert),axis=(1,2))
            input_val = ALPHA * mid_val + \
                (1-ALPHA) * (attacked_data + cur_pert)
            # feed_dict.update({ops['pointclouds_pl']: input_val - cur_pert})
            sess.run(tf.assign(ops['pert'], input_val - attacked_data))

#########################################################################################
            adv_loss_val,dist_val,pred_val,input_val,penality_infty = sess.run([ops['adv_loss'],
                                                                            ops['pert_norm'], ops['pred'], ops['pointclouds_input'], ops["pert_bound"]], feed_dict=feed_dict)
            pred_val = np.argmax(pred_val, 1)
            loss=adv_loss_val+np.average(dist_val*WEIGHT)
            dist_infty = np.amax(
                np.abs(input_val - attacked_data), axis=(1, 2))

            if iteration % ((NUM_ITERATIONS // 10) or 1) == 0:
                print((" Iter {:3d} of {}: loss={:.2f} adv:{:.2f} infty={:.2f}" +
                       "  perecl={:.2f} perinf={:.2f} ")
                              .format(iteration, NUM_ITERATIONS,
                                      loss, adv_loss_val, np.mean(penality_infty), np.mean(dist_val), np.mean(dist_infty)))
            # check if we should abort search if we're getting nowhere.
            '''
            if ABORT_EARLY and iteration % ((MAX_ITERATIONS // 10) or 1) == 0:
                
                if loss > prev * .9999999:
                    msg = "    Failed to make progress; stop early"
                    print(msg)
                    break
                prev = loss
            '''

            # for e, (dist, pred, ii,nat) in enumerate(zip(dist_val, pred_val, input_val, dist_nat)):
            #     if dist < bestdist[e] and pred == TARGET: # and nat < bestdist[e]:
            #         bestdist[e] = dist
            #         bestscore[e] = pred
            #         bestnat[e] = nat

            #     if dist < o_bestdist[e] and pred == TARGET: # and nat < o_bestnat[e]:
            #         o_bestdist[e]=dist
            #         o_bestscore[e]=pred
            #         o_bestattack[e] = ii
            #         o_bestnat[e] = nat
            #         # print(str(e)*10)
            for e, (dist, pred, ii, linfty) in enumerate(zip(dist_val, pred_val, input_val, dist_infty)):
                # and nat < bestdist[e]:
                if  pred == TARGET and linfty < bestinfty[e]:
                    bestdist[e] = dist
                    bestscore[e] = pred
                    bestinfty[e] = linfty

                # and nat < o_bestnat[e]:
                if  pred == TARGET and linfty < o_bestinfty[e]:
                    o_bestdist[e] = dist
                    o_bestscore[e] = pred
                    o_bestattack[e] = ii
                    o_bestinfty[e] = linfty
                    # print(str(e)*10)

        # # adjust the constant as needed
        for e in range(BATCH_SIZE):
            if bestscore[e]==TARGET and bestscore[e] != -1 and bestdist[e] <= o_bestdist[e] :
                # success
                lower_bound[e] = max(lower_bound[e], WEIGHT[e])
                WEIGHT[e] = (lower_bound[e] + upper_bound[e]) / 2
                #print('new result found!')
            else:
                # failure
                upper_bound[e] = min(upper_bound[e], WEIGHT[e])
                WEIGHT[e] = (lower_bound[e] + upper_bound[e]) / 2
        # #bestdist_prev=deepcopy(bestdist)

                # adjust the constant ALPHA as needed
        # for e in range(BATCH_SIZE):
        #     if bestscore[e] == TARGET and bestscore[e] != -1 and bestnat[e] <= o_bestnat[e]:
        #         # success
        #         ALPHA[e] = (1+GAMMA)*ALPHA[e]
        #         #print('new result found!')
        #     else:
        #         # failure
        #         ALPHA[e] = (1-GAMMA)*ALPHA[e]

    print(" Successfully generated adversarial examples on {} of {} instances of class {}." .format(
        sum(lower_bound > 0), BATCH_SIZE, victim))
    print("best L2 distance  :{:.2f}  best L_infty :{:.2f}".format(
        np.mean(o_bestdist), np.mean(o_bestinfty)))
    print("\n \n ----------------------------------------- \n \n")
    return o_bestdist,o_bestattack


if __name__=='__main__':
    attack()
