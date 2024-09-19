# Multi-View Vertebra Localization and Identification from CT Images
 by [Han Wu](http://hanwu.website/), Jiadong Zhang, [Yu Fang](https://yuffish.github.io/), Zhentao Liu, Nizhuan Wang, [Zhiming Cui](https://shanghaitech-impact.github.io/) and [Dinggang Shen](http://idea.bme.shanghaitech.edu.cn/home/people/faculty).
 
arXiv link: [https://arxiv.org/abs/2307.12845](https://arxiv.org/abs/2307.12845)
paper link: [https://link.springer.com/chapter/10.1007/978-3-031-43904-9_14](https://link.springer.com/chapter/10.1007/978-3-031-43904-9_14)
## Introduction
This repository is the reference code for our paper 'Multi-View Vertebra Localization and Identification from CT Images' in MICCAI 2023.
 ![Overall Pipeline](./asset/pipeline.png)



## Get Started
This repository is based on PyTorch 1.12.1 + Plastimatch 1.9.4.

### Installation
For installation of the Plastimatch, you can refer to [Plastimatch](https://plastimatch.org/).

### Training WorkFlow
1. Preprocess the data
```c
preprocess/preprocess.py
```
2. DRR Generation
```c
preprocess/generate_drr.py
preprocess/generate_drr_heatmap.py
```
3. Multi-View Contrastive Learning
```c
contrastive_learning/train_multi_view.py
```
4. Localization/ Identification network training
```c
train/train_localization.py
train/train_id_as_seg.py
```

### Inference WorkFlow
1. Preprocess the data
```c
preprocess/preprocess.py
```
2. DRR Generation
```c
preprocess/generate_drr.py
```
3. Single-View Localization & Identification
4. Multi-View Fusion
5. Evaluation
```c
# 3-5 are all in eval_all.py
eavl/eval_all.py
```
## Data Link
Public dataset:
- (VerSe'19 train): https://s3.bonescreen.de/public/VerSe-complete/dataset-verse19training.zip
- (VerSe'19 validation): https://s3.bonescreen.de/public/VerSe-complete/dataset-verse19validation.zip
- (VerSe'19 test): https://s3.bonescreen.de/public/VerSe-complete/dataset-verse19test.zip


## Citation
```c
@inproceedings{wu2023multi,
  title={Multi-view vertebra localization and identification from ct images},
  author={Wu, Han and Zhang, Jiadong and Fang, Yu and Liu, Zhentao and Wang, Nizhuan and Cui, Zhiming and Shen, Dinggang},
  booktitle={International Conference on Medical Image Computing and Computer-Assisted Intervention},
  pages={136--145},
  year={2023},
  organization={Springer}
}
```
