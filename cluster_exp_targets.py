import os 
import sys 
import argparse
import numpy as np

parser = argparse.ArgumentParser()

parser.add_argument('--gpu', type=int, default=0,
                    help='GPU to use [default: GPU 0]')

parser.add_argument('--gamma', type=float, default=0.2, help='natural factor ')
parser.add_argument('--epsilon_infty', type=float,
                    default=0.22, help='natural factor ')
parser.add_argument('--epsilon_two', type=float,
                    default=1.8, help='natural factor ')

parser.add_argument('--cluster_nb', type=int, default=0,
                    help='number of the exp in a cluster array ')
parser.add_argument('--step', type=int, default=2,
                    help='number of the hyper steps done ')


def generate_targets_setup(cluster_nb=0, epsilon_infty=0.22, epsilon_two=1.8):
    cluster_setup = []
    networks_list = ["PN", "PN1", "PN2", "GCN"]
    iterations_list = [500, 500, 500, 700]
    targets_list = [0, 5, 35, 2, 8, 33, 22, 37, 4, 30]
    for hard_bound_mode in [1, 2]:
        if hard_bound_mode == 1:
            for network, iteration in zip(networks_list, iterations_list):
                for target in targets_list:
                    cluster_setup.append(
                        (hard_bound_mode, network, epsilon_infty, 20, iteration, target))
        elif hard_bound_mode == 2:
            for network, iteration in zip(networks_list, iterations_list):
                for target in targets_list:
                    cluster_setup.append(
                        (hard_bound_mode, network, 20, epsilon_two, iteration, target))
    return cluster_setup[cluster_nb]  # len(cluster_setup) = 160


if __name__ == '__main__':
    FLAGS = parser.parse_args()
    (hard_bound_mode, network, u_infty, u_two,
     iteration, target) = generate_targets_setup(FLAGS.cluster_nb, epsilon_infty=FLAGS.epsilon_infty, epsilon_two=FLAGS.epsilon_two)
    command = "python fix.py --phase all --exp_id={} --target={} --cluster_nb={} --gpu={} --network {} --evaluation_mode=0 --unnecessary=0 --initial_alpha=0 --step={} --batch_size=5 --num_iter={} --lr_attack=0.01 --initial_weight=10 --gamma={} --beta_infty=0 --beta_cham=0 --beta_emd=0 --beta_two=0 --hard_bound_mode={} --dyn_bound_mode={} --b_two={} --b_infty={} --s_infty=0.2 --u_two={} --u_infty={}".format(
        str(FLAGS.cluster_nb), target, FLAGS.cluster_nb, FLAGS.gpu, network, FLAGS.step, iteration,
        FLAGS.gamma, hard_bound_mode, hard_bound_mode, u_two, u_infty, u_two, u_infty)
    os.system(command)
