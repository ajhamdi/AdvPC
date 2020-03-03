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
from my_utils import *



def train(setup,models,targets_list,victims_list):
    top_out_dir = osp.join(setup["base_dir"], "latent_3d_points", "data")


    # Top-dir of where point-clouds are stored.
    top_in_dir = osp.join(setup["base_dir"], "latent_3d_points", "data",
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
                decoder_args=dec_args,
                use_ae=False
                )
    conf.experiment_name = experiment_name
    conf.held_out_step = 5   # How often to evaluate/print out loss on
    # held_out data (if they are provided in ae.train() ).
    # conf.save(osp.join(train_dir, 'configuration'))
    is_training = False
    with tf.Graph().as_default():

        ae = PointNetAutoEncoderWithClassifier(conf.experiment_name, conf)
        ae.models = models
        is_training_pl = tf.placeholder(tf.bool, shape=())
        # is_projection = tf.placeholder(tf.bool, shape=())

        # pert=tf.get_variable(name='pert',shape=[setup["batch_size"],setup["num_point"],3],initializer=tf.truncated_normal_initializer(stddev=0.01))
        target = tf.placeholder(tf.int32, shape=(None))
        victim_label = tf.placeholder(tf.int32, shape=(None))
        pert = ae.pert_
        pointclouds_pl = ae.x
        pointclouds_input=ae.x_h
        # with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
        if setup["network"] == "PN":
            with tf.variable_scope(tf.get_variable_scope(), reuse=False):
                early_pred, end_points = ae.get_model_w_ae(pointclouds_input, is_training_pl)
            # with tf.variable_scope("QQ", reuse=False):
            #     late_pred, late_end_points = ae.get_model_w_ae(ae.x_reconstr, is_training_pl)
        elif setup["network"] == "PN1":
            with tf.variable_scope(tf.get_variable_scope(), reuse=False):
                early_pred, end_points = ae.get_model_w_ae_p(pointclouds_input, is_training_pl)
            # with tf.variable_scope("QQ", reuse=False):
            #     late_pred, late_end_points = ae.get_model_w_ae_p(
            #         ae.x_reconstr, is_training_pl)
        elif setup["network"] == "PN2":
            with tf.variable_scope(tf.get_variable_scope(), reuse=False):
                early_pred, end_points = ae.get_model_w_ae_pp(
                    pointclouds_input, is_training_pl)
            # with tf.variable_scope("QQ", reuse=False):
            #     late_pred, late_end_points = ae.get_model_w_ae_pp(
            #         ae.x_reconstr, is_training_pl)
        elif setup["network"] == "GCN":
            with tf.variable_scope(tf.get_variable_scope(), reuse=False):
                early_pred, end_points = ae.get_model_w_ae_gcn(
                    pointclouds_input, is_training_pl)
            # with tf.variable_scope("QQ", reuse=False):
            #     late_pred, late_end_points = ae.get_model_w_ae_gcn(
            #         ae.x_reconstr, is_training_pl)
        else :
            print("network not known")

        #adv loss targeted /relativistic targeted /  untargeted 
        if setup["evaluation_mode"] == 0:
            early_adv_loss = ae.get_adv_loss(early_pred, target)
        elif setup["evaluation_mode"] == 1:
            early_adv_loss = ae.get_adv_loss(
                early_pred, target) + ae.get_adv_loss(pointclouds_pl, victim_label)
        elif setup["evaluation_mode"] == 2 or setup["evaluation_mode"] == 3:
            early_adv_loss = ae.get_adv_loss_batch(early_pred, target)

        # dyn_target = tf.placeholder(tf.int32, shape=(None))
        # late_adv_loss = ae.get_adv_loss_batch(late_pred, dyn_target)
        # nat_norm = tf.sqrt(tf.reduce_sum(
        #     tf.square(ae.x_reconstr - ae.x_h), [1, 2]))
        # nat_norm = 1000*ae.chamfer_distance(ae.x_reconstr, ae.x_h)

        
        #perturbation l2 constraint
        pert_norm=tf.sqrt(tf.reduce_sum(tf.square(pointclouds_input- pointclouds_pl),[1,2]))
        #perturbation l1 constraint
        # pert_norm = tf.reduce_sum(tf.abs(pert), [1, 2])
        #perturbation l_infty constraint
        pert_bound = tf.norm(
            tf.nn.relu(pert-setup["s_infty"]), ord=1, axis=(1, 2))
        pert_cham = 1000*ae.chamfer_distance(pointclouds_input, pointclouds_pl)
        pert_emd = ae.emd_distance(pointclouds_input, pointclouds_pl)


        dist_weight=tf.placeholder(shape=[setup["batch_size"]],dtype=tf.float32)
        # nat_weight = tf.placeholder(shape=[setup["batch_size"]], dtype=tf.float32)
        cham_weight = tf.placeholder(shape=[setup["batch_size"]], dtype=tf.float32)
        emd_weight = tf.placeholder(shape=[setup["batch_size"]], dtype=tf.float32)
        infty_weight = tf.placeholder(shape=[setup["batch_size"]], dtype=tf.float32)
        loaded_features = tf.placeholder(shape=setup["features_size"], dtype=tf.float32)
        lr_attack=tf.placeholder(dtype=tf.float32)
        attack_optimizer = tf.train.AdamOptimizer(lr_attack)
        l_2_loss = tf.reduce_mean(tf.multiply(dist_weight, pert_norm))
        l_cham_loss = tf.reduce_mean(tf.multiply(cham_weight, pert_cham))
        l_emd_loss = tf.reduce_mean(tf.multiply(emd_weight, pert_emd))

        # nat_loss = tf.reduce_mean(tf.multiply(nat_weight, nat_norm))

        l_infty_loss = tf.reduce_mean(tf.multiply(infty_weight, pert_bound))
        
        if setup["features_layer"] == "2":
            features_loss = tf.reduce_sum(tf.square(end_points["post_max"] - loaded_features), 1)

        elif setup["features_layer"] == "1":
            features_loss = tf.reduce_sum(tf.square(end_points["final"] - loaded_features), 1)
        elif setup["features_layer"] == "3":
            features_loss = tf.reduce_sum(tf.square(end_points["pre_max"] - loaded_features[:,:,None,...]), [1,2,3])
        elif setup["features_layer"] == "5":
            features_loss = tf.reduce_sum(tf.square(end_points["second"] - loaded_features[:,:,None,...]), [1,2,3])
        elif setup["features_layer"] == "6":
            features_loss = tf.reduce_sum(tf.square(end_points["first"] - loaded_features), [1,2])
        adv_loss = (1-setup["gamma"])*early_adv_loss #+ (setup["gamma"]) * late_adv_loss 
        distance_loss = l_2_loss + l_infty_loss + l_cham_loss + l_emd_loss # +nat_loss
        total_loss = setup["sigma"] * adv_loss + distance_loss # + tf.reduce_mean(features_loss)
        attack_op = attack_optimizer.minimize(total_loss,var_list=[ae.pert])

        
        vl=tf.global_variables()
        vl = [x for x in vl if "single_class_ae" not in x.name and 'pert' not in x.name]
        train_op = attack_optimizer.minimize(total_loss, var_list=vl)

        # vl_1 = [x for x in vl if "QQ" not in x.name]
        # vl_2 = [x for x in vl if "PP" not in x.name]
        # vl_2 = {x.name.replace("QQ/", "").replace(":0", ""): x for x in vl}
        # vl_2 = [x for x in vl if "Classifier_1/" in x.name]

        # vl = [x for x in vl if  "single_class_ae" not in x.name]
        # print(20*"#", vl_1)
        # print(20*"#", vl_2)
        # saver = tf.train.Saver(
        #     {x.name.replace("PP/", "").replace("QQ/", ""): x for x in vl})
        saver = tf.train.Saver(vl)
        # saver_1 = tf.train.Saver(vl_1)
        # saver_2 = tf.train.Saver(vl_2)



        # Create a session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        #config.log_device_placement = True
        # sess = tf.Session(config=config)
        sess = ae.sess
        sess.run(tf.global_variables_initializer())
        # ae.restore_model(conf.train_dir, epoch=restore_epoch, verbose=True)



        ops = {"ae":ae,
            'pointclouds_pl': pointclouds_pl,
            #    'labels_pl': labels_pl,
               'is_training_pl': is_training_pl,
               'pointclouds_input': pointclouds_input,
               'dist_weight':dist_weight,
            #    "nat_weight": nat_weight,
               "infty_weight": infty_weight,
               "target":target,
               "victim_label": victim_label,
               "cham_weight": cham_weight,
               "emd_weight": emd_weight,
               'pert': ae.pert,
            #    "dyn_target": dyn_target,
            #    'pre_max':end_points['pre_max'],
               "loaded_features": loaded_features,
            #    'post_max':end_points['post_max'],
            #    'first': end_points['first'],
            #    'final': end_points['final'],
                'early_pred': early_pred,
            #    "late_pred": late_pred,
               'early_adv_loss': early_adv_loss,
               'adv_loss': adv_loss,
            #    "late_adv_loss": late_adv_loss,
               'pert_norm':pert_norm,
            #    'nat_norm': nat_norm,
               "pert_bound": pert_bound,
               "bound_ball_infty": ae.bound_ball_infty,
               "bound_ball_two": ae.bound_ball_two,
               "pert_cham": pert_cham,
               "pert_emd": pert_emd,
               'total_loss': total_loss,
               "features_loss": features_loss,
               'lr_attack':lr_attack,
            #    "x_m": ae.x_reconstr,
               'attack_op':attack_op,
               "train_op": train_op
               }

        # print_tensors_in_checkpoint_file(
        #     file_name=MODEL_PATH, tensor_name='beta1_power', all_tensors=True)
        saver.restore(sess, models["test_path"])
        # saver_1.restore(sess, models["test_path"])
        # saver_2.restore(sess, models["test_path"])
        print('model restored!')

        norms_names = ["L_2_norm_adv",
            "L_infty_norm_adv",
            "L_cham_norm_adv",
            "L_emd_norm_adv",
            "natural_L_cham_norm_adv"]
        
        # the class index of selected 10 largest classed in ModelNet40
        # results = ListDict(norms_names)
        # setups = ListDict(setup.keys())
        # save_results(setup["save_file"], results+setups)
        for target in targets_list:
            setup["target"] = target
            for victim in victims_list:
                setup["victim"] = victim
                attacked_data=models["attacked_data_all"][victim]#attacked_data shape:25*1024*3
                for j in range(NB_PER_VICTIM//setup["batch_size"]):
                    if setup["init"] == 0:
                        c_attacked_data = np.random.normal(0, setup["std_dev"], (setup["batch_size"], setup["num_point"], 3))
                    elif setup["init"] == 1:
                        c_attacked_data = attacked_data[j*setup["batch_size"]:(j+1)*setup["batch_size"]]
                    elif setup["init"] == 2:
                        m = j-1 if j>0 else j+1 
                        c_attacked_data = attacked_data[m*setup["batch_size"]:(m+1)*setup["batch_size"]]
                    elif setup["init"] == 3:
                        m = [0, 5, 35, 2, 8, 33, 22, 37, 4, 30]
                        idx= m.index(victim)
                        m = m[idx-1] if idx > 0 else m[idx+1]
                        attacked_data = models["attacked_data_all"][m]
                        c_attacked_data = attacked_data[j*setup["batch_size"]:(j+1)*setup["batch_size"]]
                    setup["batch"] = j
                    norms, img = attack_one_batch(sess, ops, c_attacked_data, setup)
                    train_one_batch(sess, ops, img, setup)

                    # np.save(os.path.join('.', setup["dump_dir"], '{}_{}_{}_{}_{}.npy' .format(victim, setup["target"], j, setup["features_type"], setup["features_layer"])), img)
                    # [setups.append(setup) for ii in range(setup["batch_size"])]
                    # results.extend(ListDict(norms))
                    # compiled_results.chek_error()
                    # save_results(setup["save_file"], results+setups)
                    # np.save(os.path.join('.',setup["dump_dir"],'{}_{}_{}_mxadv.npy' .format(victim,setup["target"],j)),img)
                    np.save(os.path.join('.',setup["dump_dir"],'{}_{}_{}_orig.npy' .format(victim,setup["target"],j)),c_attacked_data)#dump originial example for comparison
        #joblib.dump(dist_list,os.path.join('.',setup["dump_dir"],'dist_{}.z' .format(setup["target"])))#log distance information for performation evaluation
        save_results(setup["save_file"], results+setups)
        return results


def attack_one_batch(sess, ops, attacked_data, setup):
    c_NUM_ITERATIONS = setup["num_iter"]
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
    save_file = os.path.join(setup["features_dir"], "{}_{}_{}_features.pkl".format(
        setup["victim"], 5, setup["batch"])) if (setup["victim"] == 0) else os.path.join(setup["features_dir"], "{}_{}_{}_features.pkl".format(
            setup["victim"], 0, setup["batch"]))

    actual_loaded_features = load_obj(save_file)[setup["features_type"]]


    #the bound for the binary search
    lower_bound=np.zeros(setup["batch_size"])
    i_infty = np.zeros(setup["batch_size"])
    i_two = np.zeros(setup["batch_size"])

    WEIGHT = np.ones(setup["batch_size"]) * setup["initial_weight"]
    ALPHA = np.ones(setup["batch_size"]) * setup["initial_alpha"]
    BOUND_BALL_infty = np.ones((setup["batch_size"],setup["num_point"],3)) * setup["b_infty"]
    BOUND_BALL_TWO = np.ones((setup["batch_size"], setup["num_point"], 3)) * setup["b_two"]

    upper_bound=np.ones(setup["batch_size"]) * setup["upper_bound_weight"]
    u_infty=np.ones(setup["batch_size"]) * setup["u_infty"]#0.362
    u_two = np.ones(setup["batch_size"]) * setup["u_two"]  # 0.362


    o_bestdist = [1e10] * setup["batch_size"]
    o_besttwo = [1e10] * setup["batch_size"]
    # o_bestnat = [1e10] * setup["batch_size"]
    o_bestinfty = [1e10] * setup["batch_size"]
    o_bestcham = [1e10] * setup["batch_size"]
    o_bestemd = [1e10] * setup["batch_size"]
    o_bestscore = [-1] * setup["batch_size"]
    o_bestattack = copy.deepcopy(attacked_data)

    feed_dict = {ops['pointclouds_pl']: attacked_data,
         ops['is_training_pl']: is_training,
         ops['lr_attack']:setup["lr_attack"],
        ops['target']: setup["victim"],
        ops['victim_label']: setup["victim"],
        ops["loaded_features"]: actual_loaded_features,
        ops["bound_ball_infty"]: BOUND_BALL_infty,
        ops["bound_ball_two"]: BOUND_BALL_TWO,
        ops['dist_weight']: WEIGHT * setup["beta_two"],
        ops['infty_weight']: WEIGHT * setup["beta_infty"],
        ops['cham_weight']: WEIGHT * setup["beta_cham"],
        ops['emd_weight']: WEIGHT * setup["beta_emd"]
        }
        # ops["nat_weight"]: WEIGHT*ALPHA}


    for out_step in range(setup["step"]):
        if out_step == setup["step"]-1:
            c_NUM_ITERATIONS = c_NUM_ITERATIONS * 1


        feed_dict[ops['dist_weight']]= WEIGHT
        # feed_dict[ops['nat_weight']] =  WEIGHT*ALPHA
        feed_dict[ops["bound_ball_infty"]] =  BOUND_BALL_infty
        feed_dict[ops["bound_ball_two"]] =  BOUND_BALL_TWO

        # sess.run(tf.assign(ops['pert'], tf.zeros([setup["batch_size"], setup["num_point"], 3])))
        # first = sess.run(ops['first'], feed_dict=feed_dict)
        # print(np.linalg.norm(first-actual_loaded_features))

        sess.run(tf.assign(ops['pert'],tf.truncated_normal([setup["batch_size"],setup["num_point"],3], mean=0, stddev=0.00000001)))


        bestdist = [1e10] * setup["batch_size"]
        besttwo = [1e10] * setup["batch_size"]
        # bestnat = [1e10] * setup["batch_size"]
        bestinfty = [1e10] * setup["batch_size"]

        bestscore = [-1] * setup["batch_size"]  

        prev = 1e6      

        # early_pred_val,late_pred_val = sess.run([ops['early_pred'],ops['late_pred']], feed_dict=feed_dict)
        early_pred_val = sess.run(ops['early_pred'], feed_dict=feed_dict)



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

        if setup["evaluation_mode"] == 2 or setup["evaluation_mode"] == 3 :
            c_early_targets = untargeted_attack(
                early_pred_val, setup["victim"])
            feed_dict[ops['target']] = c_early_targets



        for iteration in range(c_NUM_ITERATIONS):
            
            _ = sess.run([ops['attack_op']], feed_dict=feed_dict)

            loss, features_loss, adv_loss_val,  dist_two, early_pred_val, input_val,  dist_cham, dist_emd = sess.run([ops["total_loss"], ops["features_loss"], ops['adv_loss'],
             ops['pert_norm'], ops['early_pred'],
              ops['pointclouds_input'], ops["pert_cham"],ops["pert_emd"]], feed_dict=feed_dict)
            # print(40*"$",actual_loaded_features.shape,features_loss.shape)
            if ((iteration+1) % setup["dyn_freq"]) == 0:
                # c_late_targets = untargeted_attack(late_pred_val, setup["victim"])
                # feed_dict[ops['dyn_target']] = c_late_targets
                if setup["evaluation_mode"] == 2 or setup["evaluation_mode"] == 3 :
                    c_early_targets = untargeted_attack(early_pred_val, setup["victim"])
                    feed_dict[ops['target']] = c_early_targets
            early_pred_val = np.argmax(early_pred_val, 1)
            # late_pred_val = np.argmax(late_pred_val, 1)
            # print(20*"#", early_pred_val)
            # loss = adv_loss_val + np.average(dist_two*WEIGHT) + np.average(dist_nat*WEIGHT*ALPHA)
            dist_infty = np.amax(np.abs(input_val - attacked_data), axis=(1, 2))

            if iteration % ((setup["num_iter"] // 10) or 1) == 0:
                print((" Iter {:3d} of {}: loss={:.2f} adv:{:.2f} " +
                       " TWO={:.3f} infty={:.3f} Cham={:.2f}")
                              .format(iteration, setup["num_iter"],
                                      float(loss), adv_loss_val, np.mean(dist_two), np.mean(dist_infty), np.mean(dist_cham)))
            # check if we should abort search if we're getting nowhere.

            for e, (c_loss, two, pred, ii, linfty, cham, emd) in enumerate(zip(features_loss, dist_two, early_pred_val, input_val, dist_infty, dist_cham, dist_emd)):
                dist = c_loss
                if dist < bestdist[e] :
                # if two < besttwo[e] and pred == setup["target"] :
                    # if emd < besttwo[e] and pred == setup["target"] and d_pred != victim:
                    besttwo[e] = two
                    bestscore[e] = pred
                    # bestnat[e] = nat
                    bestinfty[e] = linfty
                    bestdist[e] = dist


                if dist < o_bestdist[e] : # and adverserial_cond and ae_cond:
                    o_besttwo[e] = two
                    o_bestscore[e]=pred
                    o_bestattack[e] = ii  # ii
                    # o_bestnat[e] = nat
                    o_bestinfty[e] = linfty
                    o_bestcham[e] = cham
                    o_bestemd[e] = emd
                    o_bestdist[e] = dist



    print(" Successfully completed batch:{} of class {}." .format(
        setup["batch"], setup["victim"]))
                                                      #     sum(o_bestscore == attacked_label), setup["batch_size"], setup["victim"]))
    # print("best L 2 distance  :{:.2f}  best L_infty :{:.3f}   best L_nat :{:.3f}".format(
    #     np.mean(o_besttwo), np.mean(o_bestinfty), np.mean(o_bestnat)))
    # print("\n \n ----------------------------------------- \n \n")
    
    best_norms = {"L_2_norm_adv": o_besttwo, "L_infty_norm_adv": o_bestinfty,
        "L_cham_norm_adv": o_bestcham,"L_emd_norm_adv": o_bestemd}
    return best_norms, o_bestattack


def train_one_batch(sess, ops, attacked_data, setup):
    c_NUM_ITERATIONS = setup["num_iter"]
    attacked_data = copy.deepcopy(attacked_data)
    ###############################################################
    ### a simple implementation
    ### Attack all the data in variable 'attacked_data' into the same target class (specified by TARGET)
    ### binary search is used to find the near-optimal results
    ### part of the code is adpated from https://github.com/tensorflow/cleverhans/blob/master/cleverhans/attacks/carlini_wagner_l2.py
    ###############################################################

    is_training = True

    # the target label for adv pcs
    attacked_label = np.ones(shape=(len(attacked_data)),
                             dtype=int) * setup["target"]
    attacked_label = np.squeeze(attacked_label)
    save_file = os.path.join(setup["features_dir"], "{}_{}_{}_features.pkl".format(
        setup["victim"], 5, setup["batch"])) if (setup["victim"] == 0) else os.path.join(setup["features_dir"], "{}_{}_{}_features.pkl".format(
            setup["victim"], 0, setup["batch"]))

    actual_loaded_features = load_obj(save_file)[setup["features_type"]]

    #the bound for the binary search
    lower_bound = np.zeros(setup["batch_size"])
    i_infty = np.zeros(setup["batch_size"])
    i_two = np.zeros(setup["batch_size"])

    WEIGHT = np.ones(setup["batch_size"]) * setup["initial_weight"]
    ALPHA = np.ones(setup["batch_size"]) * setup["initial_alpha"]
    BOUND_BALL_infty = np.ones(
        (setup["batch_size"], setup["num_point"], 3)) * setup["b_infty"]
    BOUND_BALL_TWO = np.ones(
        (setup["batch_size"], setup["num_point"], 3)) * setup["b_two"]

    upper_bound = np.ones(setup["batch_size"]) * setup["upper_bound_weight"]
    u_infty = np.ones(setup["batch_size"]) * setup["u_infty"]  # 0.362
    u_two = np.ones(setup["batch_size"]) * setup["u_two"]  # 0.362

    o_bestdist = [1e10] * setup["batch_size"]
    o_besttwo = [1e10] * setup["batch_size"]
    # o_bestnat = [1e10] * setup["batch_size"]
    o_bestinfty = [1e10] * setup["batch_size"]
    o_bestcham = [1e10] * setup["batch_size"]
    o_bestemd = [1e10] * setup["batch_size"]
    o_bestscore = [-1] * setup["batch_size"]
    o_bestattack = copy.deepcopy(attacked_data)

    feed_dict = {ops['pointclouds_pl']: attacked_data,
                 ops['is_training_pl']: is_training,
                 ops['lr_attack']: setup["lr_attack"],
                 ops['target']: setup["victim"],
                 ops['victim_label']: setup["victim"],
                 ops["loaded_features"]: actual_loaded_features,
                 ops["bound_ball_infty"]: BOUND_BALL_infty,
                 ops["bound_ball_two"]: BOUND_BALL_TWO,
                 ops['dist_weight']: WEIGHT * setup["beta_two"],
                 ops['infty_weight']: WEIGHT * setup["beta_infty"],
                 ops['cham_weight']: WEIGHT * setup["beta_cham"],
                 ops['emd_weight']: WEIGHT * setup["beta_emd"]
                 }
    # ops["nat_weight"]: WEIGHT*ALPHA}

    for out_step in range(setup["step"]):
        if out_step == setup["step"]-1:
            c_NUM_ITERATIONS = c_NUM_ITERATIONS * 1

        feed_dict[ops['dist_weight']] = WEIGHT
        # feed_dict[ops['nat_weight']] =  WEIGHT*ALPHA
        feed_dict[ops["bound_ball_infty"]] = BOUND_BALL_infty
        feed_dict[ops["bound_ball_two"]] = BOUND_BALL_TWO



        
        bestdist = [1e10] * setup["batch_size"]
        besttwo = [1e10] * setup["batch_size"]
        # bestnat = [1e10] * setup["batch_size"]
        bestinfty = [1e10] * setup["batch_size"]

        bestscore = [-1] * setup["batch_size"]

        prev = 1e6

        # early_pred_val,late_pred_val = sess.run([ops['early_pred'],ops['late_pred']], feed_dict=feed_dict)
        early_pred_val = sess.run(ops['early_pred'], feed_dict=feed_dict)

        def untargeted_attack(all_values, victim):
            all_choices = []
            for ii in range(all_values.shape[0]):
                _, largest_index = np.unique(all_values[ii], return_index=True)
                if largest_index[-1] != victim:
                    all_choices.append(largest_index[-1])
                else:
                    all_choices.append(largest_index[-2])

            return np.array(all_choices)

        # c_late_targets = untargeted_attack(late_pred_val,setup["victim"])
        # feed_dict[ops['dyn_target']] = c_late_targets

        if setup["evaluation_mode"] == 2 or setup["evaluation_mode"] == 3:
            c_early_targets = untargeted_attack(
                early_pred_val, setup["victim"])
            feed_dict[ops['target']] = c_early_targets
        print("start TRAINING : \n \n ")
        for iteration in range(c_NUM_ITERATIONS):

            _ = sess.run([ops['train_op']], feed_dict=feed_dict)

            loss, features_loss, adv_loss_val,  dist_two, early_pred_val, input_val,  dist_cham, dist_emd = sess.run([ops["total_loss"], ops["features_loss"], ops['adv_loss'],
                                                                                                                      ops['pert_norm'], ops[
                                                                                                                          'early_pred'],
                                                                                                                      ops['pointclouds_input'], ops["pert_cham"], ops["pert_emd"]], feed_dict=feed_dict)
            # print(40*"$",actual_loaded_features.shape,features_loss.shape)
            if ((iteration+1) % setup["dyn_freq"]) == 0:
                # c_late_targets = untargeted_attack(late_pred_val, setup["victim"])
                # feed_dict[ops['dyn_target']] = c_late_targets
                if setup["evaluation_mode"] == 2 or setup["evaluation_mode"] == 3:
                    c_early_targets = untargeted_attack(
                        early_pred_val, setup["victim"])
                    feed_dict[ops['target']] = c_early_targets
            early_pred_val = np.argmax(early_pred_val, 1)
            # late_pred_val = np.argmax(late_pred_val, 1)
            # print(20*"#", early_pred_val)
            # loss = adv_loss_val + np.average(dist_two*WEIGHT) + np.average(dist_nat*WEIGHT*ALPHA)
            dist_infty = np.amax(
                np.abs(input_val - attacked_data), axis=(1, 2))

            if iteration % ((setup["num_iter"] // 10) or 1) == 0:
                print((" Iter {:3d} of {}: loss={:.2f} adv:{:.2f} " +
                       " TWO={:.3f} infty={:.3f} Cham={:.2f}")
                      .format(iteration, setup["num_iter"],
                              float(loss), adv_loss_val, np.mean(dist_two), np.mean(dist_infty), np.mean(dist_cham)))
            # check if we should abort search if we're getting nowhere.

            for e, (c_loss, two, pred, ii, linfty, cham, emd) in enumerate(zip(features_loss, dist_two, early_pred_val, input_val, dist_infty, dist_cham, dist_emd)):
                dist = c_loss
                if dist < bestdist[e]:
                    # if two < besttwo[e] and pred == setup["target"] :
                    # if emd < besttwo[e] and pred == setup["target"] and d_pred != victim:
                    besttwo[e] = two
                    bestscore[e] = pred
                    # bestnat[e] = nat
                    bestinfty[e] = linfty
                    bestdist[e] = dist

                if dist < o_bestdist[e]:  # and adverserial_cond and ae_cond:
                    o_besttwo[e] = two
                    o_bestscore[e] = pred
                    o_bestattack[e] = ii  # ii
                    # o_bestnat[e] = nat
                    o_bestinfty[e] = linfty
                    o_bestcham[e] = cham
                    o_bestemd[e] = emd
                    o_bestdist[e] = dist

    print(" Successfully completed batch:{} of class {}." .format(
        setup["batch"], setup["victim"]))
    #     sum(o_bestscore == attacked_label), setup["batch_size"], setup["victim"]))
    # print("best L 2 distance  :{:.2f}  best L_infty :{:.3f}   best L_nat :{:.3f}".format(
    #     np.mean(o_besttwo), np.mean(o_bestinfty), np.mean(o_bestnat)))
    # print("\n \n ----------------------------------------- \n \n")

    best_norms = {"L_2_norm_adv": o_besttwo, "L_infty_norm_adv": o_bestinfty,
                  "L_cham_norm_adv": o_bestcham, "L_emd_norm_adv": o_bestemd}
    return best_norms, o_bestattack
