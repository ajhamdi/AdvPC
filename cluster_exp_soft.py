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
parser.add_argument('--soft', type=int, default=1,
                    help='tthe type of teh exp : 0-> hard constrant 1-> soft constraint  ')


def generate_soft_setup(cluster_nb=0):
    cluster_setup = []
    networks_list = ["PN", "PN1", "PN2", "GCN"]
    iterations_list = [500, 500, 500, 700]
    for network, iteration in zip(networks_list, iterations_list):
        for ii in [(1, 0, 0), (0, 1, 0), (0, 0, 1)]:
            beta_cham = ii[0]
            beta_two = ii[1]
            beta_emd = ii[2]
            cluster_setup.append(
                (network, beta_cham, beta_two, beta_emd, iteration))
    return cluster_setup[cluster_nb]  # len(cluster_setup) = 160


if __name__ == '__main__':
    FLAGS = parser.parse_args()
    (network, beta_cham, beta_two, beta_emd,
     iteration) = generate_soft_setup(FLAGS.cluster_nb)
    command = "python natural.py --phase all --cluster_nb={} --gpu={} --network {} --evaluation_mode=0 --unnecessary=1 --initial_alpha=0 --step={} --batch_size=5 --num_iter={} --lr_attack=0.01 --initial_weight=10 --gamma={} --beta_infty=0 --beta_cham={} --beta_emd={} --beta_two={} --hard_bound_mode=0 --dyn_bound_mode=0 --b_two=20 --b_infty=2 --s_infty=0.2 --u_two=20 --u_infty=2".format(
        FLAGS.cluster_nb, FLAGS.gpu, network, FLAGS.step, iteration,
        FLAGS.gamma, beta_cham,beta_emd,beta_two)
    os.system(command)
