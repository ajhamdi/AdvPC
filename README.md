# AdvPC: Transferable Adversarial Perturbations on 3D Point Clouds (ECCV 2020)
By [Abdullah Hamdi](https://abdullahamdi.com/), Sara Rojas , Ali Thabet, [Bernard Ghanem](http://www.bernardghanem.com/)

official code of ECCV 2020 paper "[AdvPC: Transferable Adversarial Perturbations on 3D Point Clouds](https://arxiv.org/abs/1912.00461)"

<img src="https://github.com/ajhamdi/AdvPC/blob/master/doc/pipeline.png" width="80%" alt="attack pipeline" align=center>


## Citation
If you find our work useful in your research, please consider citing:
```
@article{hamdi2020advpc,
  title={AdvPC: Transferable Adversarial Perturbations on 3D Point Clouds},
  author={Abdullah Hamdi and Sara Rojas and Ali Thabet and Bernard Ghanem},
  journal={ECCV},
  year={2020}
}
```


## Requirement
This code is tested with Python 2.7 and Tensorflow 1.9/1.10

Other required packages include numpy, joblib, sklearn, etc.( see [environment.yml](https://github.com/ajhamdi/AdvPC/blob/master/environment.yml))

## creating conda environment and compiling tf_ops C++ libraries 
- conda create -n NAME python=2.7 anaconda
- conda activate NAME
- conda install tensorflow-gpu=1.10.0
- conda install -c anaconda cudatoolkit==9
-  make sure everything is there nvcc --version / gcc --version / whereis nvcc
- look for TF paths `python -c 'import tensorflow as tf; print(tf.sysconfig.get_lib() + "/python/_pywrap_tensorflow_internal.so")' | xargs ldd`
- change TF_PATHS in the **makefile** file in `latent_3d_points/external/structural_losses/makefile`
- run `make` inside the above the directory

## Usage
There are two main Python scripts in the root directorty: 
- `attack.py` -- AdvPC Adversarial Point Pertubations
- `evaluate.py` -- code to evaluate the atcked point clouds under different networks and defeneses

To run AdPV to attack network `NETWORK` and also evaluate the the attack, please use the following command under  :
```
python attack.py --phase all --network NETWORK --step=1 --batch_size=5 --num_iter=100 --lr_attack=0.01 --gamma=0.25 --b_infty=0.1 --u_infty=0.1
```
- `NETWORK` is one of four networks : **PN**: [PointNet](https://arxiv.org/abs/1612.00593), **PN1**:[PointNet++ (MSG)](https://github.com/charlesq34/pointnet2) , **PN2**: [PointNet++ (SSG)](https://github.com/charlesq34/pointnet2),  **GCN**: [DGCNN](https://liuziwei7.github.io/projects/DGCNN)
- `b_infty` , `u_infty` is the L_infty norm budget used in the experiments.
- `step` is the number of different initilizations for the attack.
- `lr_attack` is the learning rate of the attack.
- `gamma` is the main hyper parameter of **AdvPC** (that trades-off success with transferablity).
- `num_iter` is the number of iterations in the optimzation.

Other parameters can be founded in the script, or run `python attack.py -h`. The default parameters are the ones used in the paper.

The results will be saved in `results/exp0/` with the original point cloud and attacked point cloud saved as `V_T_B_orig.npy` and `V_T_B_adv.npy` respectively. `V` is the victim class of the expirements (out of ModelNet 40 classes ) and `T` is the target class (100 if untargeted attack) , and `B` is the batch number. 


## Other files
- log/`NETWORK`/model.ckpt -- the victims models (trained on ModelNet40) used in the paper, where `NETWORK` is one of four networks **PN**: [PointNet](https://arxiv.org/abs/1612.00593), **PN1**:[PointNet++ (MSG)](https://github.com/charlesq34/pointnet2) , **PN2**: [PointNet++ (SSG)](https://github.com/charlesq34/pointnet2),  **GCN**: [DGCNN](https://liuziwei7.github.io/projects/DGCNN)
- data/attacked_data.z -- the victim data used in the paper. It can be loaded with `joblib.load`, resulting in a Python list whose element is a numpy array (shape: 25\*1024\*3; 25 objects of the same class, each object is represented by 1024 points)
- utils/tf_nndistance -- a self-defined tensorlfow op used for Chamfer/Hausdorff distance calculation. Use tf_nndistance_compile.sh to compile the op. The bash code may need modification according to the version and installtion path of CUDA. Note that it should be OK to directly calculate Chamfer/Hausdorff distance with available tf ops instead of tf_nndistance.

## Misc
- The sample adversarial point clouds can be downloaded [here](https://drive.google.com/open?id=1KLtJXFpq70YkB2DAxfUYyrWcv8kbkUJd)
- The aligned version of ModelNet40 data (in point cloud data format) can be downloaded [here](https://drive.google.com/open?id=1m7BmdtX1vWrpl9WRX5Ds2qnIeJHKmE36).
- The visulization in the paper is rendered with [pptk](https://github.com/heremaps/pptk)
- Please open an issue or contact Abdullah Hamdi (abdullah.hamdi@kaust.edu.sa) if there is any question.

## Acknoledgements
This paper and repo borrows codes and ideas from several great github repos:
[latent 3D point clouds](https://github.com/optas/latent_3d_points) , [3d-adv-pc](https://github.com/xiangchong1/3d-adv-pc), [Dynamic Graph CNN for Learning on Point Clouds](https://liuziwei7.github.io/projects/DGCNN), [PointNet ++](https://github.com/charlesq34/pointnet2)

## License
The code is released under MIT License (see LICENSE file for details).
