# Semi-supervised Implicit Scene Completion from Sparse LiDAR

![demo](doc/demo.gif)

![teaser](doc/qualitative.png)

![sup0](doc/qualitative_0.png)

![sup1](doc/qualitative_1.png)

![sup2](doc/qualitative_2.png)

![sup3](doc/qualitative_3.png)

![sup4](doc/qualitative_4.png)


## Introduction

Recent advances show that semi-supervised implicit representation learning can be achieved through physical constraints like Eikonal equations. However, this scheme has not yet been successfully used for LiDAR point cloud data, due to its spatially varying sparsity. 

In this repository, we develop a novel formulation that conditions the semi-supervised implicit function on localized shape embeddings. It exploits the strong representation learning power of sparse convolutional networks to generate shape-aware dense feature volumes, while still allows semi-supervised signed distance function learning without knowing its exact values at free space. With extensive quantitative and qualitative results, we demonstrate intrinsic properties of this new learning system and its usefulness in real-world road scenes. Notably, we improve IoU from 26.3\% to 51.0\% on SemanticKITTI. Moreover, we explore two paradigms to integrate semantic label predictions, achieving implicit semantic completion. Codes and data are publicly available.


## Installation

### Requirements
    
    CUDA=11.1
    python>=3.8
    Pytorch>=1.8
    numpy
    ninja
    MinkowskiEngine
    tensorboard
    pyyaml
    configargparse
    scripy
    open3d
    h5py
    plyfile
    scikit-image



### Data preparation

Download the SemanticKITTI dataset from 
[HERE](http://semantic-kitti.org/assets/data_odometry_voxels.zip). Unzip it into the same directory as `SISC`.



## Training and inference
The configuration for training/inference is stored in `opt.yaml`, which can be modified as needed.

### Scene Completion

Run the following command for a certain `task` (train/valid/visualize):

    CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 main_sc.py --task=[task] --experiment_name=[experiment_name]


### Semantic Scene Completion
#### SSC option A
Run the following command for a certain `task` (ssc_pretrain/ssc_valid/train/valid/visualize):

    CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 main_ssc_a.py --task=[task] --experiment_name=[experiment_name]

Here, use ssc_pretrain/ssc_valid to train/validate the SSC part. Then the pre-trained model can be used to further train the whole model.

#### SSC option B
Run the following command for a certain `task` (train/valid/visualize):

    CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1 main_ssc_b.py --task=[task] --experiment_name=[experiment_name]

