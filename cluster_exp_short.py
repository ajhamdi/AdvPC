import os 
import sys 
import argparse
import numpy as np

parser = argparse.ArgumentParser()

parser.add_argument('--gpu', type=int, default=0,
                    help='GPU to use [default: GPU 0]')

parser.add_argument('--gamma', type=float, default=0.2, help='natural factor ')

parser.add_argument('--cluster_nb', type=int, default=0,
                    help='number of the exp in a cluster array ')
parser.add_argument('--step', type=int, default=2,
                    help='number of the hyper steps done ')


def generate_setup_short(cluster_nb=0):
    cluster_setup = []
    two_epsilons = [0.1, 0.22, 0.48, 0.72, 1.0, 1.5, 1.8,  2.8, 4.0, 7.0]
    infty_epsilons = [0.01, 0.04, 0.05, 0.1, 0.18, 0.28, 0.35, 0.45, 0.6, 0.75]
    networks_list = ["PN", "PN1", "PN2", "GCN"]
    iterations_list = [500, 500, 500, 700]
    for hard_bound_mode in [2, 1]:
        if hard_bound_mode == 1:
            for network, iteration in zip(networks_list, iterations_list):
                for epsilon_infty in infty_epsilons:
                    if epsilon_infty == 0.0:
                        cluster_setup.append(
                            (hard_bound_mode, network, epsilon_infty, 20, 10))
                    else:
                        cluster_setup.append(
                            (hard_bound_mode, network, epsilon_infty, 20, iteration))
        elif hard_bound_mode == 2:
            for network, iteration in zip(networks_list, iterations_list):
                for epsilon_two in two_epsilons:
                    if epsilon_two == 0.0:
                        cluster_setup.append(
                            (hard_bound_mode, network, 20, epsilon_two, 10))
                    else:
                        cluster_setup.append(
                            (hard_bound_mode, network, 20, epsilon_two, iteration))
    return cluster_setup[cluster_nb]  # len(cluster_setup) = 80

if __name__ == '__main__':
    FLAGS = parser.parse_args()
    (hard_bound_mode, network, u_infty, u_two,
     iteration) = generate_setup_short(FLAGS.cluster_nb)
    command = "python natural.py --phase all --cluster_nb={} --gpu={} --network {} --evaluation_mode=0 --unnecessary=0 --initial_alpha=0 --step={} --batch_size=5 --num_iter={} --lr_attack=0.01 --initial_weight=10 --gamma={} --beta_infty=0 --beta_cham=0 --beta_emd=0 --beta_two=0 --hard_bound_mode={} --dyn_bound_mode={} --b_two={} --b_infty={} --s_infty=0.2 --u_two={} --u_infty={}".format(
        FLAGS.cluster_nb, FLAGS.gpu, network, FLAGS.step, iteration,
        FLAGS.gamma, hard_bound_mode, hard_bound_mode, u_two, u_infty, u_two, u_infty)
    os.system(command)
