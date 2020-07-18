from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file
import tensorflow as tf
import numpy as np
import argparse
import importlib
import os
import pandas as pd
import sys
import copy
from collections import OrderedDict
from scipy import stats
from latent_3d_points.src.ae_templates import mlp_architecture_ala_iclr_18, default_train_params
from latent_3d_points.src.autoencoder_w_cls import Configuration as Conf
from latent_3d_points.src.point_net_ae_w_cls import PointNetAutoEncoderWithClassifier

from latent_3d_points.src.in_out import snc_category_to_synth_id, create_dir, PointCloudDataSet, \
    load_all_point_clouds_under_folder

import os.path as osp
from latent_3d_points.src.tf_utils import reset_tf_graph

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, 'models'))
sys.path.append(os.path.join(BASE_DIR, 'utils'))
from my_utils import *
from evaluate import evaluate
# Use to save Neural-Net check-points etc.
# Use to save Neural-Net check-points etc.


# import tf_nndistance
import joblib

parser = argparse.ArgumentParser()
parser.add_argument('--phase', type=str, default='attack', choices=['attack', 'evaluate','all'], help='perform attack or evaluate performed attack or both')
parser.add_argument('--exp_id', type=str, default='random',help='pick ')
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
# parser.add_argument('--model', default='pointnet_cls_w_ae', help='Model name: pointnet_cls or pointnet_cls_basic [default: pointnet_cls]')
parser.add_argument('--network', default='PN', choices=['PN', 'PN1', 'PN2',"GCN"], 
    help='the network used to perform the attack , PN1: POintNet , PN2 : POinNEt ++ , PN3: POneNet++ w/ 2 scales , GCN : DCGCN network   ')
# parser.add_argument('--log_dir', default='log', help='Log dir [default: log]')
parser.add_argument('--batch_size', type=int, default=5, help='Batch Size for attack [default: 5]')
parser.add_argument('--num_point', type=int, default=1024, help='Point Number [256/512/1024/2048] [default: 1024]')
parser.add_argument('--data_dir', default='data', help='data folder path [data]')
parser.add_argument('--dump_dir', default='results', help='dump folder path [perturbation]')

parser.add_argument('--evaluation_mode', type=int, default=0,
                    help='the int type of evaluation mode : 0: regukar targeted attack  .. 1: relativisitic loss targeteed attack   2:regular untargeted attack  3: relativisitic untargeted attack   ')
parser.add_argument('--srs', type=float, default=0.1,
                    help='SRS percentage of the SRS defense if --evaluation_mode== 1')
parser.add_argument('--sor', type=float, default=1.1,
                    help='SOR percentage of the SOR defense if --evaluation_mode== 2')

parser.add_argument('--hard_bound_mode', type=int, default=1,
   help='the int type of hard bound mode : 0: no upper hard bound... 1: u_infty upper hard bound on L infty 2: u_two upper hard bound on L 2')
parser.add_argument('--dyn_bound_mode', type=int, default=1,
                    help='the int type of dyn bound mode : 0: no upper dyn bound... 1: b_infty upper hard bound on L infty 2: b_two upper hard bound on L 2')
parser.add_argument('--target', type=int, default=5, help='target class index')
parser.add_argument('--unnecessary', type=int, default=0, help='target class index')
parser.add_argument('--victim', type=int, default=0, help='target class index')
parser.add_argument('--lr_attack', type=float, default=0.005, help='learning rate for optimization based attack')
parser.add_argument('--initial_alpha', type=float, default=0,help=' natural factor')

parser.add_argument('--gamma', type=float, default=0.2,help='natural factor ')
parser.add_argument('--kappa', type=float, default=0, help='margin of attack ')
parser.add_argument('--kappa_ae', type=float,
                    default=0, help='margin of attack on the AE output ')


parser.add_argument('--u_infty', type=float, default=0.362,
                    help='hard_upper_bound on L infty')
parser.add_argument('--u_two', type=float, default=0.362,
                    help='hard_upper_bound on L two')

parser.add_argument('--beta_two', type=float, default=0.0,
                    help='L 2 factor')
parser.add_argument('--beta_infty', type=float, default=0.0,
                    help='L infty factor')
parser.add_argument('--beta_cham', type=float, default=0.0,
                    help='L Chamfer factor')
parser.add_argument('--beta_emd', type=float, default=0.0,
                    help='L EMD factor')

parser.add_argument('--b_infty', type=float, default=0.2,
                    help='L infty dynamic hard bound b infty')
parser.add_argument('--s_infty', type=float, default=0.2,
                    help='L infty soft bound s_infty')
parser.add_argument('--b_two', type=float, default=0.2,
                    help='L 2 dynamic hard bound b_2 ')

parser.add_argument('--initial_weight', type=float, default=10, help='initial value for the parameter lambda')
parser.add_argument('--upper_bound_weight', type=float, default=80, help='upper_bound value for the parameter lambda')
parser.add_argument('--step', type=int, default=10, help='binary search step')
parser.add_argument('--num_iter', type=int, default=500, help='number of iterations for each binary search step')
parser.add_argument('--cluster_nb', type=int, default=0,
                    help='number of the exp in a cluster array ')
parser.add_argument('--dyn_freq', type=int, default=10,
                    help='the frequency at which to comute the dynamic target in untargeted AE loss and in untargeted attack  ')


# parser.add_argument("--set",metavar="KEY=VALUE",nargs='+',help="Set a number of key-value pairs "
#                              "(do not put spaces before or after the = sign). "
#                              "If a value contains spaces, you should define "
#                              "it with double quotes: "
#                              'foo="this is a sentence". Note that '
#                              "values are always treated as strings.")
FLAGS = parser.parse_args()


KAPPA = FLAGS.kappa
KAPPA_AE = FLAGS.kappa_ae
BATCH_SIZE = FLAGS.batch_size
NUM_POINT = FLAGS.num_point
GPU_INDEX = FLAGS.gpu
# MODEL_PATH = os.path.join("log", FLAGS.network, "model.ckpt")

# if FLAGS.network == "PN":
#     model = "pointnet_cls_w_ae"
# elif FLAGS.network == "PN2":
#     model = "pointnet_cls_w_ae_pp"
# elif FLAGS.network == "PN1":
#     model = "pointnet_cls_w_ae_p"
# else :
#     model = "gcn_cls_w_ae"
# MODEL = importlib.import_module(model) # import network module
if FLAGS.exp_id == "random":
    EXP_ID = random_id()
    FLAGS.exp_id = EXP_ID
else :
    EXP_ID = FLAGS.exp_id

DUMP_DIR = os.path.join(FLAGS.dump_dir,EXP_ID)
FLAGS.dump_dir = DUMP_DIR
DATA_DIR = FLAGS.data_dir

check_folder(DUMP_DIR)
# TARGET=FLAGS.target
LR_ATTACK=FLAGS.lr_attack
INITIAL_ALPHA = FLAGS.initial_alpha
B_INFTY = FLAGS.b_infty
B_TWO = FLAGS.b_two
U_INFTY = FLAGS.u_infty
U_TWO = FLAGS.u_two
S_INFTY = FLAGS.s_infty
GAMMA = FLAGS.gamma

BETA_TWO = FLAGS.beta_two
BETA_INFTY = FLAGS.beta_infty
BETA_CHAM = FLAGS.beta_cham
BETA_EMD = FLAGS.beta_emd

#WEIGHT=FLAGS.weight

attacked_data_all=joblib.load(os.path.join(DATA_DIR,'attacked_data.z'))

INITIAL_WEIGHT=FLAGS.initial_weight
UPPER_BOUND_WEIGHT=FLAGS.upper_bound_weight
#ABORT_EARLY=False
BINARY_SEARCH_STEP=FLAGS.step
NUM_ITERATIONS=FLAGS.num_iter
DYN_FREQ = FLAGS.dyn_freq
# top_out_dir = osp.join(BASE_DIR, "latent_3d_points", "data")
# # Top-dir of where point-clouds are stored.
# top_in_dir = osp.join(BASE_DIR, "latent_3d_points", "data",
#                       "shape_net_core_uniform_samples_2048")


# experiment_name = 'single_class_ae'
# n_pc_points = 1024  # 2048                # Number of points per model.
# bneck_size = 128                  # Bottleneck-AE size
# # Loss to optimize: 'emd' or 'chamfer'             # Bottleneck-AE size
# ae_loss = 'chamfer'
# train_params = default_train_params()
# encoder, decoder, enc_args, dec_args = mlp_architecture_ala_iclr_18(
#     n_pc_points, bneck_size)
# train_dir = create_dir(osp.join(top_out_dir, experiment_name))
# conf = Conf(n_input=[n_pc_points, 3],
#             loss=ae_loss,
#             training_epochs=train_params['training_epochs'],
#             batch_size=BATCH_SIZE,
#             denoising=train_params['denoising'],
#             learning_rate=train_params['learning_rate'],
#             train_dir=train_dir,
#             hard_bound_mode=FLAGS.hard_bound_mode,
#             dyn_bound_mode=FLAGS.dyn_bound_mode,
#             b_infty=FLAGS.b_infty,
#             b_two=FLAGS.b_two,
#             u_infty=FLAGS.u_infty,
#             u_two=FLAGS.u_two,
#             loss_display_step=train_params['loss_display_step'],
#             saver_step=train_params['saver_step'],
#             z_rotate=train_params['z_rotate'],
#             encoder=encoder,
#             decoder=decoder,
#             encoder_args=enc_args,
#             decoder_args=dec_args
#             )
# conf.experiment_name = experiment_name
# conf.held_out_step = 5   # How often to evaluate/print out loss on
# # held_out data (if they are provided in ae.train() ).
# conf.save(osp.join(train_dir, 'configuration'))



def attack(setup,models,targets_list,victims_list):
    top_out_dir = osp.join(BASE_DIR, "latent_3d_points", "data")


    # Top-dir of where point-clouds are stored.
    top_in_dir = osp.join(BASE_DIR, "latent_3d_points", "data",
                        "shape_net_core_uniform_samples_2048")


    experiment_name = 'single_class_ae'
    n_pc_points = 1024  # 2048                # Number of points per model.
    bneck_size = 128
    NB_PER_VICTIM = 25                   # nb of point clouds per class
    # Loss to optimize: 'emd' or 'chamfer'             # Bottleneck-AE size
    ae_loss = 'chamfer'
    train_params = default_train_params()
    encoder, decoder, enc_args, dec_args = mlp_architecture_ala_iclr_18(
        n_pc_points, bneck_size)
    train_dir = create_dir(osp.join(top_out_dir, experiment_name))
    conf = Conf(n_input=[n_pc_points, 3],
                loss=ae_loss,
                training_epochs=train_params['training_epochs'],
                batch_size=setup["batch_size"],
                denoising=train_params['denoising'],
                learning_rate=train_params['learning_rate'],
                train_dir=train_dir,
                hard_bound_mode=setup["hard_bound_mode"],
                dyn_bound_mode=setup["dyn_bound_mode"],
                b_infty=setup["b_infty"],
                b_two=setup["b_two"],
                u_infty=setup["u_infty"],
                u_two=setup["u_two"],
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
    # conf.save(osp.join(train_dir, 'configuration'))
    is_training = False
    with tf.Graph().as_default():
        # with tf.device('/gpu:'+str(GPU_INDEX)):

        # print("3333333333333333333")
        load_pre_trained_ae = True
        restore_epoch = 500
        if load_pre_trained_ae:
            # conf = Conf.load(train_dir + '/configuration')
            # reset_tf_graph()
            ae = PointNetAutoEncoderWithClassifier(conf.experiment_name, conf)
            ae.restore_model(conf.train_dir, epoch=restore_epoch, verbose=True)
            # pointclouds_pl, labels_pl = MODEL.placeholder_inputs(BATCH_SIZE, NUM_POINT)
        ae.models = models
        is_training_pl = tf.placeholder(tf.bool, shape=())
        # is_projection = tf.placeholder(tf.bool, shape=())

        # pert=tf.get_variable(name='pert',shape=[BATCH_SIZE,NUM_POINT,3],initializer=tf.truncated_normal_initializer(stddev=0.01))
        target = tf.placeholder(tf.int32, shape=(None))
        victim_label = tf.placeholder(tf.int32, shape=(None))
        pert = ae.pert_
        pointclouds_pl = ae.x
        pointclouds_input=ae.x_h
        # with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
        if setup["network"] == "PN":
            with tf.variable_scope(tf.get_variable_scope(), reuse=False):
                early_pred, end_points = ae.get_model_w_ae(
                    pointclouds_input, is_training_pl)
            with tf.variable_scope("QQ", reuse=False):
                late_pred, end_points_late = ae.get_model_w_ae(ae.x_reconstr, is_training_pl)
        elif setup["network"] == "PN1":
            with tf.variable_scope(tf.get_variable_scope(), reuse=False):
                early_pred, end_points = ae.get_model_w_ae_p(
                    pointclouds_input, is_training_pl)
            with tf.variable_scope("QQ", reuse=False):
                late_pred, end_points_late = ae.get_model_w_ae_p(
                    ae.x_reconstr, is_training_pl)
        elif setup["network"] == "PN2":
            with tf.variable_scope(tf.get_variable_scope(), reuse=False):
                early_pred, end_points = ae.get_model_w_ae_pp(
                    pointclouds_input, is_training_pl)
            with tf.variable_scope("QQ", reuse=False):
                late_pred, end_points_late = ae.get_model_w_ae_pp(
                    ae.x_reconstr, is_training_pl)
        elif setup["network"] == "GCN":
            with tf.variable_scope(tf.get_variable_scope(), reuse=False):
                early_pred, end_points = ae.get_model_w_ae_gcn(
                    pointclouds_input, is_training_pl)
            with tf.variable_scope("QQ", reuse=False):
                late_pred, end_points_late = ae.get_model_w_ae_gcn(
                    ae.x_reconstr, is_training_pl)
        else :
            print("network not known")

        #adv loss targeted /relativistic targeted /  untargeted 
        if setup["evaluation_mode"] == 0:
            early_adv_loss = ae.get_adv_loss(early_pred, target)
        elif setup["evaluation_mode"] == 1:
            early_adv_loss = ae.get_adv_loss(
                early_pred, target) + ae.get_adv_loss(pointclouds_pl, victim_label)
        elif setup["evaluation_mode"] == 2 or setup["evaluation_mode"] == 3:
            early_adv_loss = ae.get_untargeted_adv_loss(early_pred, victim_label, KAPPA)

        dyn_target = tf.placeholder(tf.int32, shape=(None))
        # late_adv_loss = ae.get_adv_loss_batch(late_pred, dyn_target)
        late_adv_loss = ae.get_untargeted_adv_loss(
            late_pred, victim_label, KAPPA_AE)
        # nat_norm = tf.sqrt(tf.reduce_sum(
        #     tf.square(ae.x_reconstr - ae.x_h), [1, 2]))
        nat_norm = 1000*ae.chamfer_distance(ae.x_reconstr, ae.x_h)

        
        #perturbation l2 constraint
        pert_norm=tf.sqrt(tf.reduce_sum(tf.square(pert),[1,2]))
        #perturbation l1 constraint
        # pert_norm = tf.reduce_sum(tf.abs(pert), [1, 2])
        #perturbation l_infty constraint
        pert_bound = tf.norm(
            tf.nn.relu(pert-S_INFTY), ord=1, axis=(1, 2))
        pert_cham = 1000*ae.chamfer_distance(pointclouds_input, pointclouds_pl)
        pert_emd = ae.emd_distance(pointclouds_input, pointclouds_pl)


        dist_weight=tf.placeholder(shape=[BATCH_SIZE],dtype=tf.float32)
        nat_weight = tf.placeholder(shape=[BATCH_SIZE], dtype=tf.float32)
        cham_weight = tf.placeholder(shape=[BATCH_SIZE], dtype=tf.float32)
        emd_weight = tf.placeholder(shape=[BATCH_SIZE], dtype=tf.float32)
        infty_weight = tf.placeholder(shape=[BATCH_SIZE], dtype=tf.float32)
        lr_attack=tf.placeholder(dtype=tf.float32)
        attack_optimizer = tf.train.AdamOptimizer(lr_attack)
        l_2_loss = tf.reduce_mean(tf.multiply(dist_weight, pert_norm))
        l_cham_loss = tf.reduce_mean(tf.multiply(cham_weight, pert_cham))
        l_emd_loss = tf.reduce_mean(tf.multiply(emd_weight, pert_emd))

        nat_loss = tf.reduce_mean(tf.multiply(nat_weight, nat_norm))

        l_infty_loss = tf.reduce_mean(tf.multiply(infty_weight, pert_bound))
        adv_loss = (1-GAMMA)*early_adv_loss + (GAMMA)* late_adv_loss
        distance_loss = l_2_loss + nat_loss + l_infty_loss + l_cham_loss + l_emd_loss
        total_loss = adv_loss + distance_loss
        attack_op = attack_optimizer.minimize(total_loss,var_list=[ae.pert])
        
        vl=tf.global_variables()
        vl = [x for x in vl if "single_class_ae" not in x.name ]
        vl_1 = [x for x in vl if "QQ" not in x.name]
        # vl_2 = [x for x in vl if "PP" not in x.name]
        vl_2 = {x.name.replace("QQ/", "").replace(":0", ""): x for x in vl}
        # vl_2 = [x for x in vl if "Classifier_1/" in x.name]

        # vl = [x for x in vl if  "single_class_ae" not in x.name]
        # print(20*"#", vl_1)
        # print(20*"#", vl_2)
        # saver = tf.train.Saver(
        #     {x.name.replace("PP/", "").replace("QQ/", ""): x for x in vl})
        # saver = tf.train.Saver(vl)
        saver_1 = tf.train.Saver(vl_1)
        saver_2 = tf.train.Saver(vl_2)



        # Create a session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        #config.log_device_placement = True
        # sess = tf.Session(config=config)
        sess = ae.sess
        sess.run(tf.global_variables_initializer())
        ae.restore_model(conf.train_dir, epoch=restore_epoch, verbose=True)



        ops = {"ae":ae,
            'pointclouds_pl': pointclouds_pl,
            #    'labels_pl': labels_pl,
               'is_training_pl': is_training_pl,
               'pointclouds_input': pointclouds_input,
               'dist_weight':dist_weight,
               "nat_weight": nat_weight,
               "infty_weight": infty_weight,
               "target":target,
               "victim_label": victim_label,
               "cham_weight": cham_weight,
               "emd_weight": emd_weight,
               'pert': ae.pert,
               "dyn_target": dyn_target,
            #    'pre_max':end_points['pre_max'],
            #    'post_max':end_points['post_max'],
               'early_pred': early_pred,
               "late_pred": late_pred,
               'early_adv_loss': early_adv_loss,
               'adv_loss': adv_loss,
            #    "late_adv_loss": late_adv_loss,
               'pert_norm':pert_norm,
               'nat_norm': nat_norm,
               "pert_bound": pert_bound,
               "bound_ball_infty": ae.bound_ball_infty,
               "bound_ball_two": ae.bound_ball_two,
               "pert_cham": pert_cham,
               "pert_emd": pert_emd,
               'total_loss': total_loss,
               'lr_attack':lr_attack,
               "x_m": ae.x_reconstr,
               'attack_op':attack_op
               }

        # print_tensors_in_checkpoint_file(
        #     file_name=MODEL_PATH, tensor_name='beta1_power', all_tensors=True)
        # saver.restore(sess, MODEL_PATH)
        saver_1.restore(sess, models["test_path"])
        saver_2.restore(sess, models["test_path"])
        print('model restored!')

        norms_names = ["L_2_norm_adv",
            "L_infty_norm_adv",
            "L_cham_norm_adv",
            "L_emd_norm_adv",
            "natural_L_cham_norm_adv"]
        
        # the class index of selected 10 largest classed in ModelNet40
        results = ListDict(norms_names)
        setups = ListDict(setup.keys())
        save_results(setup["save_file"], results+setups)
        for target in targets_list:
            setup["target"] = target
            for victim in victims_list:
                if victim == setup["target"]:
                    continue
                setup["victim"] = victim
                attacked_data=attacked_data_all[victim]#attacked_data shape:25*1024*3
                for j in range(NB_PER_VICTIM//BATCH_SIZE):
                    norms, img = attack_one_batch(
                        sess, ops, attacked_data[j*BATCH_SIZE:(j+1)*BATCH_SIZE], setup)
                    np.save(os.path.join('.',DUMP_DIR,'{}_{}_{}_adv.npy' .format(victim,setup["target"],j)),img)
                    [setups.append(setup) for ii in range(setup["batch_size"])]
                    results.extend(ListDict(norms))
                    # compiled_results.chek_error()
                    save_results(setup["save_file"], results+setups)
                    # np.save(os.path.join('.',DUMP_DIR,'{}_{}_{}_mxadv.npy' .format(victim,setup["target"],j)),img)
                    np.save(os.path.join('.',DUMP_DIR,'{}_{}_{}_orig.npy' .format(victim,setup["target"],j)),attacked_data[j*BATCH_SIZE:(j+1)*BATCH_SIZE])#dump originial example for comparison
        #joblib.dump(dist_list,os.path.join('.',DUMP_DIR,'dist_{}.z' .format(setup["target"])))#log distance information for performation evaluation
        save_results(setup["save_file"], results+setups)
        return results


def attack_one_batch(sess, ops, attacked_data, setup):
    c_NUM_ITERATIONS = NUM_ITERATIONS
    attacked_data = copy.deepcopy(attacked_data)
    ###############################################################
    ### a simple implementation
    ### Attack all the data in variable 'attacked_data' into the same target class (specified by TARGET)
    ### binary search is used to find the near-optimal results
    ### part of the code is adpated from https://github.com/tensorflow/cleverhans/blob/master/cleverhans/attacks/carlini_wagner_l2.py
    ###############################################################

    is_training = False

    attacked_label=np.ones(shape=(len(attacked_data)),dtype=int) * setup["target"] #the target label for adv pcs
    attacked_label=np.squeeze(attacked_label)

    #the bound for the binary search
    lower_bound=np.zeros(BATCH_SIZE)
    i_infty = np.zeros(BATCH_SIZE)
    i_two = np.zeros(BATCH_SIZE)

    WEIGHT = np.ones(BATCH_SIZE) * INITIAL_WEIGHT
    ALPHA = np.ones(BATCH_SIZE) * INITIAL_ALPHA
    BOUND_BALL_infty = np.ones((BATCH_SIZE,NUM_POINT,3)) * B_INFTY
    BOUND_BALL_TWO = np.ones((BATCH_SIZE, NUM_POINT, 3)) * B_TWO

    upper_bound=np.ones(BATCH_SIZE) * UPPER_BOUND_WEIGHT
    u_infty=np.ones(BATCH_SIZE) * U_INFTY#0.362
    u_two = np.ones(BATCH_SIZE) * U_TWO  # 0.362


    o_bestdist = [1e10] * BATCH_SIZE
    o_besttwo = [1e10] * BATCH_SIZE
    o_bestnat = [1e10] * BATCH_SIZE
    o_bestinfty = [1e10] * BATCH_SIZE
    o_bestcham = [1e10] * BATCH_SIZE
    o_bestemd = [1e10] * BATCH_SIZE
    o_bestscore = [-1] * BATCH_SIZE
    o_bestattack = copy.deepcopy(attacked_data)

    feed_dict = {ops['pointclouds_pl']: attacked_data,
         ops['is_training_pl']: is_training,
         ops['lr_attack']:LR_ATTACK,
        ops['target']: setup["target"],
        ops['victim_label']: setup["victim"],
        ops["bound_ball_infty"]: BOUND_BALL_infty,
        ops["bound_ball_two"]: BOUND_BALL_TWO,
        ops['dist_weight']: WEIGHT * BETA_TWO,
        ops['infty_weight']: WEIGHT * BETA_INFTY,
        ops['cham_weight']: WEIGHT * BETA_CHAM,
        ops['emd_weight']: WEIGHT * BETA_EMD,
        ops["nat_weight"]: WEIGHT*ALPHA}


    for out_step in range(BINARY_SEARCH_STEP):
        if out_step == BINARY_SEARCH_STEP-1:
            c_NUM_ITERATIONS = c_NUM_ITERATIONS * 1


        feed_dict[ops['dist_weight']]= WEIGHT
        feed_dict[ops['nat_weight']] =  WEIGHT*ALPHA
        feed_dict[ops["bound_ball_infty"]] =  BOUND_BALL_infty
        feed_dict[ops["bound_ball_two"]] =  BOUND_BALL_TWO



        sess.run(tf.assign(ops['pert'],tf.truncated_normal([BATCH_SIZE,NUM_POINT,3], mean=0, stddev=0.00001)))

        bestdist = [1e10] * BATCH_SIZE
        besttwo = [1e10] * BATCH_SIZE
        bestnat = [1e10] * BATCH_SIZE
        bestinfty = [1e10] * BATCH_SIZE

        bestscore = [-1] * BATCH_SIZE  

        prev = 1e6      

        # early_pred_val,late_pred_val = sess.run([ops['early_pred'],ops['late_pred']], feed_dict=feed_dict)


        def untargeted_attack(all_values, victim):
            all_choices = []
            for ii in range(all_values.shape[0]):
                _, largest_index = np.unique(all_values[ii], return_index=True)
                if largest_index[-1] != victim:
                    all_choices.append(largest_index[-1])
                else :
                    all_choices.append(largest_index[-2])

            return np.array(all_choices)
        
        # c_late_targets = untargeted_attack(late_pred_val,setup["victim"])
        # feed_dict[ops['dyn_target']] = c_late_targets

        # if setup["evaluation_mode"] == 2 or setup["evaluation_mode"] == 3 :
        #     c_early_targets = untargeted_attack(early_pred_val, setup["victim"])
        #     feed_dict[ops['target']] = c_early_targets



        for iteration in range(c_NUM_ITERATIONS):
            # feed_dict.update({ops['pointclouds_pl']: attacked_data})
            
            _ = sess.run([ops['attack_op']], feed_dict=feed_dict)
            # _,largest_index = np.unique(late_pred_val[0],return_index=True)
            # # print(20*"#", late_pred_val[late_pred_val != victim])
            # if len(late_pred_val[late_pred_val != victim]) == 0:
            #     c_target = setup["target"]
            # else :
            #     c_target = late_pred_val[late_pred_val != victim][0]
            #     # print("@@@@@@@@@@@@@@@@@@@@")

            # feed_dict[ops['dyn_target']] = c_target
            # adv_loss_val,dist_two,pred_val,input_val,cur_pert = sess.run([ops['adv_loss'],
            #                                                           ops['pert_norm'], ops['pred'], ops['pointclouds_input'], ops["pert"]], feed_dict=feed_dict)
            # pred_val = np.argmax(pred_val, 1)
            # loss=adv_loss_val+np.average(dist_two*WEIGHT)
            # cur_pert = sess.run(ops["pert"], feed_dict=feed_dict)
                        ###########################################################################################3
            # mid_val = sess.run(ops["x_m"], feed_dict=feed_dict)
            # dist_nat = np.linalg.norm(mid_val - (attacked_data + cur_pert),axis=(1,2))
            # input_val = ALPHA * mid_val + \
            #     (1-ALPHA) * (attacked_data + cur_pert)
            # feed_dict.update({ops['pointclouds_pl']: input_val - cur_pert})
            # sess.run(tf.assign(ops['pert'], input_val - attacked_data))

#########################################################################################
            # adv_loss_val, dist_two, early_pred_val,late_pred_val, input_val, dist_nat = sess.run([ops['adv_loss'],
            #     ops['pert_norm'], ops['early_pred'], ops['late_pred'], ops['pointclouds_input'], ops['nat_norm']], feed_dict=feed_dict)
            total_loss_val, adv_loss_val, dist_two, early_pred_val, late_pred_val, input_val, dist_nat, dist_cham, dist_emd = sess.run([ops["total_loss"], ops['adv_loss'],
             ops['pert_norm'], ops['early_pred'], ops['late_pred'],
              ops['pointclouds_input'], ops['nat_norm'], ops["pert_cham"],
               ops["pert_emd"]], feed_dict=feed_dict)
            # if ((iteration+1) % DYN_FREQ) == 0:
            #     c_late_targets = untargeted_attack(late_pred_val, setup["victim"])
            #     feed_dict[ops['dyn_target']] = c_late_targets
            #     if setup["evaluation_mode"] == 2 or setup["evaluation_mode"] == 3 :
            #         c_early_targets = untargeted_attack(early_pred_val, setup["victim"])
            #         feed_dict[ops['target']] = c_early_targets
            early_pred_val = np.argmax(early_pred_val, 1)
            late_pred_val = np.argmax(late_pred_val, 1)
            # print(20*"#", early_pred_val)
            # loss = adv_loss_val + \
            #     np.average(dist_two*WEIGHT) + \
            #     np.average(dist_nat*WEIGHT*ALPHA)
            dist_infty = np.amax(
                np.abs(input_val - attacked_data), axis=(1, 2))

            if iteration % ((NUM_ITERATIONS // 10) or 1) == 0:
                print((" Iter {:3d} of {}: loss={:.2f} adv:{:.2f} " +
                       " TWO={:.3f} infty={:.3f} Cham={:.2f} NAT={:.2f} ")
                              .format(iteration, NUM_ITERATIONS,
                                      total_loss_val, adv_loss_val, np.mean(dist_two), np.mean(dist_infty), np.mean(dist_cham), np.mean(dist_nat)))
            # check if we should abort search if we're getting nowhere.
            '''
            if ABORT_EARLY and iteration % ((MAX_ITERATIONS // 10) or 1) == 0:
                
                if loss > prev * .9999999:
                    msg = "    Failed to make progress; stop early"
                    print(msg)
                    break
                prev = loss
            '''

            for e, (two, pred, d_pred, ii, linfty,nat,cham,emd) in enumerate(zip(dist_two, early_pred_val, late_pred_val, input_val, dist_infty, dist_nat,dist_cham,dist_emd)):
                dist = (two, linfty, two)[setup["dyn_bound_mode"]]  # according to the norm mode we picked we pick the best distance 
                # if nat < besttwo[e] and pred == setup["target"]:
                adverserial_cond = (pred == setup["target"]) if (
                    setup["evaluation_mode"] == 0 or setup["evaluation_mode"] == 1) else (pred != setup["victim"])
                ae_cond = (d_pred != setup["victim"] or ((GAMMA < 0.001)and not bool(setup["cluster_nb"])) or bool(setup["unnecessary"]))
                if (dist < bestdist[e] or bool(setup["cluster_nb"])) and adverserial_cond and ae_cond:
                # if two < besttwo[e] and pred == setup["target"] :
                    # if emd < besttwo[e] and pred == setup["target"] and d_pred != victim:
                    besttwo[e] = two
                    bestscore[e] = pred
                    bestnat[e] = nat
                    bestinfty[e] = linfty
                    bestdist[e] = dist


                # if nat < o_besttwo[e] and pred == setup["target"]:
                if (dist < o_bestdist[e] or bool(setup["cluster_nb"])) and adverserial_cond and ae_cond:
                    # if two < o_besttwo[e] and pred == setup["target"] :
                # if emd < o_besttwo[e] and pred == setup["target"] and d_pred != victim:
                    o_besttwo[e] = two
                    o_bestscore[e]=pred
                    o_bestattack[e] = ii  # ii
                    o_bestnat[e] = nat
                    o_bestinfty[e] = linfty
                    o_bestcham[e] = cham
                    o_bestemd[e] = emd
                    o_bestdist[e] = dist



                    # print(str(e)*10)
            # for e, (two, pred, ii, linfty) in enumerate(zip(dist_two, pred_val, input_val, dist_infty)):
            #     # and nat < besttwo[e]:
            #     if  pred == setup["target"] and linfty < bestinfty[e]:
            #         besttwo[e] = two
            #         bestscore[e] = pred
            #         bestinfty[e] = linfty

            #     # and nat < o_bestnat[e]:
            #     if  pred == setup["target"] and linfty < o_bestinfty[e]:
            #         o_besttwo[e] = two
            #         o_bestscore[e] = pred
            #         o_bestattack[e] = ii
            #         o_bestinfty[e] = linfty
                    # print(str(e)*10)

        # # adjust the constant as needed
        if not bool(setup["cluster_nb"]):
            for e in range(BATCH_SIZE):
                if bestscore[e]==setup["target"] and bestscore[e] != -1 and besttwo[e] <= o_besttwo[e] :
                    # success
                    lower_bound[e] = max(lower_bound[e], WEIGHT[e])
                    WEIGHT[e] = (lower_bound[e] + upper_bound[e]) / 2
                    #print('new result found!')
                else:
                    # failure
                    upper_bound[e] = min(upper_bound[e], WEIGHT[e])
                    WEIGHT[e] = (lower_bound[e] + upper_bound[e]) / 2
            if setup["dyn_bound_mode"] == 1:
                for e in range(BATCH_SIZE):
                    if bestscore[e] == setup["target"] and bestscore[e] != -1 and bestinfty[e] <= o_bestinfty[e]:
                        # success
                        u_infty[e] = min(u_infty[e], BOUND_BALL_infty[e][0][0])
                        BOUND_BALL_infty[e] = np.ones_like(BOUND_BALL_infty[e]) *( (i_infty[e] + u_infty[e]) / 2)
                        # print('new result found!')
                    else:
                        # failure
                        i_infty[e] = max(i_infty[e], BOUND_BALL_infty[e][0][0])
                        BOUND_BALL_infty[e] = np.ones_like(BOUND_BALL_infty[e])* ((i_infty[e] + u_infty[e]) / 2)

            elif setup["dyn_bound_mode"] == 2:
                for e in range(BATCH_SIZE):
                    if bestscore[e] == setup["target"] and bestscore[e] != -1 and bestinfty[e] <= o_bestinfty[e]:
                        # success
                        u_two[e] = min(u_two[e], BOUND_BALL_TWO[e][0][0])
                        BOUND_BALL_TWO[e] = np.ones_like(
                            BOUND_BALL_TWO[e]) * ((i_two[e] + u_two[e]) / 2)
                        # print('new result found!')
                    else:
                        # failure
                        i_two[e] = max(i_two[e], BOUND_BALL_TWO[e][0][0])
                        BOUND_BALL_TWO[e] = np.ones_like(
                            BOUND_BALL_TWO[e]) * ((i_two[e] + u_two[e]) / 2)
        # #besttwo_prev=deepcopy(besttwo)

                # adjust the constant ALPHA as needed
        # for e in range(BATCH_SIZE):
        #     if bestscore[e] == setup["target"] and bestscore[e] != -1 and bestnat[e] <= o_bestnat[e]:
        #         # success
        #         ALPHA[e] = (1+GAMMA)*ALPHA[e]
        #         #print('new result found!')
        #     else:
        #         # failure
        #         ALPHA[e] = (1-GAMMA)*ALPHA[e]

    print(" Successfully generated adversarial examples on {} of {} instances of class {}." .format(
        sum(o_bestscore == attacked_label), BATCH_SIZE, setup["victim"]))
    print("best L 2 distance  :{:.2f}  best L_infty :{:.3f}   best L_nat :{:.3f}".format(
        np.mean(o_besttwo), np.mean(o_bestinfty), np.mean(o_bestnat)))
    print("\n \n ----------------------------------------- \n \n")
    
    best_norms = {"L_2_norm_adv": o_besttwo, "L_infty_norm_adv": o_bestinfty,
        "L_cham_norm_adv": o_bestcham,"L_emd_norm_adv": o_bestemd,
                  "natural_L_cham_norm_adv": o_bestnat}
    
    return best_norms,o_bestattack

def initialize(setup,models):
        setup["results_file"] = os.path.join(
        setup["dump_dir"], "" + setup["exp_id"]+"_full.csv")
        setup["save_file"] = os.path.join(setup["dump_dir"], setup["exp_id"]+".csv")
        setup["setups_file"] = os.path.join(setup["dump_dir"], setup["exp_id"]+"_setup.csv")
        setup["load_file"] = os.path.join(setup["dump_dir"],setup["exp_id"] +".csv")
        pn1_dir = os.path.join(BASE_DIR, "..", "pointnet2")
        pn2_dir = os.path.join(BASE_DIR, "..", "pointnet2")
        gcn_dir = os.path.join(BASE_DIR, "..", "dgcnn", "tensorflow")

        models["PN_PATH"] = os.path.join(BASE_DIR, "log", "PN", "model.ckpt")
        models["PN1_PATH"] = os.path.join(BASE_DIR, "log", "PN1", "model.ckpt")
        models["PN2_PATH"] = os.path.join(BASE_DIR, "log", "PN2", "model.ckpt")
        models["GCN_PATH"] = os.path.join(BASE_DIR, "log", "GCN", "model.ckpt")
        models["PN_PATH_ROBUST"] = os.path.join(BASE_DIR, "Adv_Training", "PN", "model.ckpt")
        models["PN1_PATH_ROBUST"] = os.path.join(BASE_DIR, "Adv_Training", "PN1", "model.ckpt")
        models["PN2_PATH_ROBUST"] = os.path.join(BASE_DIR, "Adv_Training", "PN2", "model.ckpt")
        models["GCN_PATH_ROBUST"] = os.path.join(BASE_DIR, "Adv_Training", "GCN", "model.ckpt")
        models["PN"] = os.path.join(BASE_DIR, 'models', "pointnet_cls.py")
        models["PN1"] = os.path.join(pn1_dir, 'models', "pointnet2_cls_msg.py")
        models["PN2"] = os.path.join(pn2_dir, 'models', "pointnet2_ssg_cls.py")
        models["GCN"] = os.path.join(gcn_dir, 'models', "dgcnn.py")
        models["test"] = copy.deepcopy(models[setup["network"]])
        models["test_path"] = copy.deepcopy(models[setup["network"]+"_PATH"])
        models["test_path_robust"] = copy.deepcopy(models[setup["network"]+"_PATH_ROBUST"])

if __name__=='__main__':
    setup = vars(FLAGS)
    models = {}
    initialize(setup, models)
    victims_list = [0, 5, 35, 2, 8, 33, 22, 37, 4, 30]
    # victims_list = [35]
    # targets_list = [0, 5, 35, 2, 8, 33, 22, 37, 4, 30]
    targets_list = [0]
    # targets_list = [5,0,35]
    if setup["evaluation_mode"] == 2 or setup["evaluation_mode"] == 3 :
        targets_list = [100]
        


    
    if setup["phase"] == "attack":
        log_setup(setup, setup["setups_file"])
        results = attack(setup,models,targets_list,victims_list)
    
    elif setup["phase"] == "evaluate":
        results = load_results(setup["load_file"])
        ev_results = evaluate(setup, results,models, targets_list, victims_list)
    
    elif setup["phase"] == "all":
        log_setup(setup, setup["setups_file"])
        results = attack(setup,models, targets_list, victims_list)
        ev_results = evaluate(setup, results,models, targets_list, victims_list)
    else :
        print("unkown phase")
