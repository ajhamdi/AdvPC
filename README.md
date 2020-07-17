# AdvPC: Transferable Adversarial Perturbations on 3D Point Clouds (ECCV 2020)
By [Abdullah Hamdi](https://abdullahamdi.com/), Sara Rojas , Ali Thabet, [Bernard Ghanem](http://www.bernardghanem.com/)

official code of ECCV 2020 paper "[AdvPC: Transferable Adversarial Perturbations on 3D Point Clouds](https://arxiv.org/abs/1912.00461)"

<img src="https://github.com/ajhamdi/AdvPC/blob/master/doc/pipeline.png" width="50%" alt="attack pipeline" align=center>

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
This code is tested with Python 2.7 and Tensorflow 1.10.0

Other required packages include numpy, joblib, sklearn, etc. 

## Usage
There are two main Python scripts in the root directorty: 
- attack.py -- AdvPC Adversarial Point Pertubations
- evaluate.py -- code to evaluate the atcked point clouds under different networks and defeneses

The code logics of these four scripts are similar; they attack the victim objects into the specified target class.
The basic usage is `python perturbation.py --target=5`. 

Other parameters can be founded in the script, or run `python perturbation.py -h`. The default parameters are the ones used in the paper.



## Other files
- log/NETWORK/model.ckpt -- the victims models (trained on ModelNet40) used in the paper, where NETWORK is one of four networks are `PN`: PointNet, `PN1`:[PointNet++ (MSG)](https://github.com/charlesq34/pointnet2) , `PN2`: [PointNet++ (SSG)](https://github.com/charlesq34/pointnet2),  `GCN`: [DGCNN](https://liuziwei7.github.io/projects/DGCNN)
- data/attacked_data.z -- the victim data used in the paper. It can be loaded with `joblib.load`, resulting in a Python list whose element is a numpy array (shape: 25\*1024\*3; 25 objects of the same class, each object is represented by 1024 points)
- **gen_initial.py** -- used to generate initial points for adversarial cluster/object. The script uses DBSCAN to cluster the generated critical points.
- critical -- the default directory to dump the generated initial points
- data/airplane.py -- the airplane object used in the paper as a uav for the adversarial object. can be loaded with ```np.load```.
- utils/tf_nndistance -- a self-defined tensorlfow op used for Chamfer/Hausdorff distance calculation. Use tf_nndistance_compile.sh to compile the op. The bash code may need modification according to the version and installtion path of CUDA. Note that it should be OK to directly calculate Chamfer/Hausdorff distance with available tf ops instead of tf_nndistance.

## compiling tf_ops C++ libraries 
- conda create -n NAME python=2.7 anaconda
- conda activate NAME
- conda install tensorflow-gpu=1.10.0
- conda install -c anaconda cudatoolkit==9
-  make sure everything is there nvcc --version / gcc --version / whereis nvcc
- look for TF paths python -c 'import tensorflow as tf; print(tf.sysconfig.get_lib() + "/python/_pywrap_tensorflow_internal.so")' | xargs lddpython -c 'import tensorflow as tf; print(tf.sysconfig.get_lib() + "/python/_pywrap_tensorflow_internal.so")' | xargs ldd
- change paths in the makefile file
- run `make` 

## Misc
- The sample adversarial point clouds can be downloaded [here](https://drive.google.com/open?id=1KLtJXFpq70YkB2DAxfUYyrWcv8kbkUJd). The targeted model is log/model.ckpt
- The aligned version of ModelNet40 data (in point cloud data format) can be downloaded [here](https://drive.google.com/open?id=1m7BmdtX1vWrpl9WRX5Ds2qnIeJHKmE36).
- The visulization in the paper is rendered with MeshLab
- Please open an issue or contact Abdullah Hamdi (abdullah.hamdi@kaust.edu.sa) if there is any question.

## Acknoledgements
This paper and repo borrows codes and ideas from several great github repos:
- [latent 3D point clouds](https://github.com/optas/latent_3d_points) 
- [3d-adv-pc](https://github.com/xiangchong1/3d-adv-pc)
- [Dynamic Graph CNN for Learning on Point Clouds](https://liuziwei7.github.io/projects/DGCNN)
- [PointNet ++](https://github.com/charlesq34/pointnet2)

## License
The code is released under MIT License (see LICENSE file for details).
