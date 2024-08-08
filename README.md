# PRG4SSL-MNAR

This repo is the official Pytorch implementation of our paper:

> **Towards Semi-supervised Learning with Non-random Missing Labels**  
> **Authors**: ***[Yue Duan](https://njuyued.github.io/)**, Zhen Zhao, Lei Qi, Lei Wang, Luping Zhou and Yinghuan Shi* 
 
- 🔗 Quick links: [[PDF](https://arxiv.org/pdf/2308.08872)/[Abs](https://arxiv.org/abs/2308.08872)-arXiv | [PDF](https://openaccess.thecvf.com/content/ICCV2023/papers/Duan_Towards_Semi-supervised_Learning_with_Non-random_Missing_Labels_ICCV_2023_paper.pdf)/[Abs](https://openaccess.thecvf.com/content/ICCV2023/html/Duan_Towards_Semi-supervised_Learning_with_Non-random_Missing_Labels_ICCV_2023_paper.html)-Published | [Poster](figures/poster.jpg) | [Zhihu](https://zhuanlan.zhihu.com/p/653555164) | [Code Download](https://github.com/NJUyued/PRG4SSL-MNAR/archive/refs/heads/master.zip)]
 
 - 📰 Latest news:
     - **We write a detailed explanation (in chinese) to this work on [Zhihu](https://zhuanlan.zhihu.com/p/653555164).**
     - Our paper is accepted by **IEEE/CVF International Conference on Computer Vision (ICCV) 2023** 🎉🎉. Thanks to users.
 - 📑 Related works:
     - 📍 **[MOST RELEVANT]** Interested in **robust SSL in MNAR setting** with mismatched distributions? 👉 Check out our ECCV'22 paper **RDA** [[PDF-arXiv](https://arxiv.org/pdf/2208.04619) | [Code](https://github.com/NJUyued/RDA4RobustSSL)].
     - 🆕 **[LATEST]** Interested in the **SSL in fine-grained visual classification (SS-FGVC)**? 👉 Check out our AAAI'24 paper **SoC** [[PDF-arXiv](https://arxiv.org/pdf/2312.12237) | [Code](https://github.com/NJUyued/SoC4SS-FGVC/)].
     - Interested in the conventional SSL or more application of **complementary label in SSL**? 👉 Check out our TNNLS paper **MutexMatch** [[PDF-arXiv](https://arxiv.org/pdf/2203.14316) | [Code](https://github.com/NJUyued/MutexMatch4SSL/)].

## Introduction

Semi-supervised learning (SSL) tackles the label missing problem by enabling the effective usage of unlabeled data. While existing SSL methods focus on the traditional setting, a practical and challenging scenario called label *Missing Not At Random (MNAR)* is usually ignored. In MNAR, the labeled and unlabeled data fall into different class distributions resulting in biased label imputation, which deteriorates the performance of SSL models. In this work, class transition tracking based **Pseudo-Rectifying Guidance (PRG)** is devised for MNAR. We explore the class-level guidance information obtained by the Markov random walk, which is modeled on a dynamically created graph built over the class tracking matrix. PRG unifies the history information of each class transition caused by the pseudo-rectifying procedure to activate the model's enthusiasm for neglected classes, so as the quality of pseudo-labels on both popular classes and rare classes in MNAR could be improved. 

<div align=center>

<img width="750px" src="/figures/framework.Jpeg"> 
 
</div>

## Requirements
- numpy==1.21.6
- pandas==1.3.2
- Pillow==10.0.0
- scikit_learn==1.3.0
- torch==1.8.0
- torchvision==0.9.0
## How to Train
### Important Args
- `--last`: Set this flag to use the model of $\textrm{PRG}^{\textrm{Last}}$.
- `--alpha`: class invariance coefficient. By default, `--alpha 1` is set. When set `--last`, please set `--alpha 3`.
- `--nb`: Number of tracked bathches.
- `--mismatch [none/prg/cadr/darp/darp_reversed]` : Select the MNAR protocol. `none` means the conventional balanced setting. See Sec. 4 in our paper for the details of MNAR protocols.
- `--n0` : When `--mismatch prg`, this arg means the imbalanced ratio $N_0$ for labeled data; When `--mismatch [darp/darp_reversed]`, this arg means the imbalanced ratio $\gamma_l$ for labeled data.
- `--gamma` : When `--mismatch cadr`, this arg means the imbalanced ratio $\gamma$ for labeled data. When `--mismatch prg`, this arg means the imbalanced ratio $\gamma$ for unlabeled data; When `--mismatch DARP/DARP_reversed`, this arg means the imbalanced ratio $\gamma_u$ for unlabeled data. 
- `--num_labels` : Amount of labeled data used in conventional balanced setting. 
- `--net` : By default, Wide ResNet (WRN-28-2) are used for experiments. If you want to use other backbones for tarining, set `--net [resnet18/preresnet/cnn13]`. We provide alternatives as follows: ResNet-18, PreAct ResNet and CNN-13.
- `--dataset [cifar10/cifar100/miniimage]` and `--data_dir`  : Your dataset name and path.  
- `--num_eval_iter` : After how many iterations, we evaluate the model. Note that although we show the accuracy of pseudo-labels on unlabeled data in the evaluation, this is only to show the training process. We did not use any information about labels for unlabeled data in the training. 

### Training with Single GPU

We recommend using a single GPU for training to better reproduce our results. Multi-GPU training is feasible, but our results are all obtained from single GPU training.

```
python train_prg.py --world-size 1 --rank 0 --gpu [0/1/...] @@@other args@@@
```
### Training with Multi-GPUs

- Using DataParallel

```
python train_prg.py --world-size 1 --rank 0 @@@other args@@@
```

- Using DistributedDataParallel with single node


```
python train_prg.py --world-size 1 --rank 0 --multiprocessing-distributed @@@other args@@@
```

## Examples of Running
By default, the model and `dist&index.txt` will be saved in `\--save_dir\--save_name`. The file `dist&index.txt` will display detailed settings of MNAR. This code assumes 1 epoch of training, but the number of iterations is 2\*\*20. For CIFAR-100, you need set `--widen_factor 8` for WRN-28-8 whereas WRN-28-2 is used for CIFAR-10.  Note that you need set `--net resnet18` for mini-ImageNet. 

### MNAR Settings
#### CADR's protocol in Tab. 1
- CIFAR-10 with $\gamma=20$
```
python train_prg.py --world-size 1 --rank 0 --lr_decay cos --seed 1 --num_eval_iter 1000 --overwrite --save_name cifar10 --dataset cifar10 --num_classes 10 --num_labels 40 --mismatch cadr --gamma 20 --gpu 0
```

- CIFAR-100 with $\gamma=50$ 
```
python train_prg.py --world-size 1 --rank 0 --lr_decay cos --seed 1 --num_eval_iter 1000 --overwrite --save_name cifar100 --dataset cifar100 --num_classes 100 --num_labels 400 --mismatch cadr --gamma 50 --gpu 0 --widen_factor 8
```

- mini-ImageNet with $\gamma=50$ 
```
python train_prg.py --world-size 1 --rank 0 --lr_decay cos --seed 1 --num_eval_iter 1000 --overwrite --save_name miniimage --dataset miniimage --num_classes 100 --num_labels 1000 --mismatch cadr --gamma 50 --gpu 0 --net resnet18 
```

#### Our protocol in Tab. 2
- CIFAR-10 with 40 labels and $N_0=10$
```
python train_prg.py --world-size 1 --rank 0 --lr_decay cos --seed 1 --num_eval_iter 1000 --overwrite --save_name cifar10 --dataset cifar10 --num_classes 10 --num_labels 40 --mismatch prg --n0 10 --gpu 0
```

- CIFAR-100 with 400 labels and $N_0=40$ 
```
python train_prg.py --world-size 1 --rank 0 --lr_decay cos --seed 1 --num_eval_iter 1000 --overwrite --save_name cifar100 --dataset cifar100 --num_classes 100 --num_labels 400 --mismatch prg --n0 40 --gpu 0 --widen_factor 8
```

- mini-ImageNet with 1000 labels and $N_0=40$
```
python train_prg.py --world-size 1 --rank 0 --lr_decay cos --seed 1 --num_eval_iter 1000 --overwrite --save_name miniimage --dataset miniimage --num_classes 100 --num_labels 1000 --mismatch prg --n0 40 --gpu 0 --net resnet18 
```

#### Our protocol in Fig. 6(a)
- CIFAR-10 with 40 labels, $N_0=10$ and $\gamma=5$ 

```
python train_prg.py --world-size 1 --rank 0 --lr_decay cos --seed 1 --num_eval_iter 1000 --overwrite --save_name cifar10 --dataset cifar10 --num_classes 10 --num_labels 40 --mismatch prg --n0 10 --gamma 5 --gpu 0
```


#### Our protocol in Tab. 10
- CIFAR-10 with 40 labels and $\gamma=20$ 

```
python train_prg.py --world-size 1 --rank 0 --lr_decay cos --seed 1 --num_eval_iter 1000 --overwrite --save_name cifar10 --dataset cifar10 --num_classes 10 --num_labels 40 --mismatch prg --gamma 20 --gpu 0
```

#### DARP's protocol in Fig. 6(a)
- CIFAR-10 with $\gamma_l=100$ and $\gamma_u=1$ 

```
python train_prg.py --world-size 1 --rank 0 --lr_decay cos --seed 1 --num_eval_iter 1000 --overwrite --save_name cifar10 --dataset cifar10 --num_classes 10 --mismatch darp --n0 100 --gamma 1 --gpu 0
```


- CIFAR-10 with $\gamma_l=100$ and $\gamma_u=100$ (reversed) 
```
python train_prg.py --world-size 1 --rank 0 --lr_decay cos --seed 1 --num_eval_iter 1000 --overwrite --save_name cifar10 --dataset cifar10 --num_classes 10 --mismatch darp_reversed --n0 100 --gamma 100 --gpu 0
```


### Conventional Setting 
#### Matched and balanced distribution in Tab. 11
- CIFAR-10 with 40 labels

```
python train_prg.py --world-size 1 --rank 0 --lr_decay cos --seed 1 --num_eval_iter 1000 --overwrite --save_name cifar10 --dataset cifar10 --num_classes 10 --num_labels 40  --gpu 0
```

## Resume Training and Evaluation
If you restart the training, please use `--resume --load_path @your_weight_path`.

For evaluation, run

```
python eval_prg.py --load_path @your_weight_path --dataset [cifar10/cifar100/miniimage] --data_dir @your_dataset_path --num_classes @number_of_classes
```
    
By default, WideResNet-28-2 backbone is used for CIFAR-10. Use `--widen-factor 8` (i.e., WideResNet-28-8) for CIFAR-100 and `--net resnet18` for mini-ImageNet.



## Results (e.g., seed=1)

| Dateset | Labels | N0 |gamma|Acc|Setting|Method|Weight|
| :-----:| :----: | :----: |:----: |:----: |:----: |:----: |:----: |
|CIFAR-10 | 40 | - |-|94.05 |Conventional settings|PRG|[here][cf10-con-40-p]|
| | 250 | - |- |94.36 |||[here][cf10-c-250-p]|
| | 4000 | - |- |95.48 |||[here][cf10-con-4000-p]|
| | 40 | - |-|93.79 |Conventional settings|PRG^Last|[here][cf10-c-40-l]|
| | 250 | - |- |94.76 |||[here][cf10-c-250-l]|
| | 4000 | - |- |95.75 |||[here][cf10-c-4000-l]|
| | - | - |20 |94.04 |CADR's protocol|PRG|[here][cf10-c-20-p]|
| | - | - |50 |93.78 |||[here][cf10-c-50-p]|
| | - | - |100 |94.51 |||[here][cf10-c-100-p]|
| | - | - |20 |94.74 |CADR's protocol|PRG^Last|[here][cf10-c-20-l]|
| | - | - |50 |94.74 |||[here][cf10-c-50-l]|
| | - | - |100 |94.75 |||[here][cf10-c-100-l]|
| | 40 | 10 |- |93.81 |Ours protocol|PRG|[here][cf10-o-4010-p]|
| | 40 | 20 |- |93.39 |||[here][cf10-o-4020-p]|
| | 40 | 10 |2 |90.25 |||[here][cf10-o-40102-p]|
| | 40 | 10 |5 |82.84 |||[here][cf10-o-40105-p]|
| | 100 | 40 |5 |79.58 |||[here][cf10-o-100405-p]|
| | 100 | 40 |10 |78.61 |||[here][cf10-o-1004010-p]|
| | 250 | 100 |- |93.76 |||[here][cf10-o-250100-p]|
| | 250 | 200 |- |91.65 |||[here][cf10-o-250200-p]|
| | 40 | 10 |- |91.59 |Ours protocol|PRG^Last|[here][cf10-o-4010-l]|
| | 40 | 20 |- |80.31 |||[here][cf10-o-4020-l]|
| | 250 | 100 |- |91.36 |||[here][cf10-o-250100-l]|
| | 250 | 200 |- |62.16 |||[here][cf10-o-250200-l]|
|  | DARP | 100 |1 |94.41 |DARP's protocol|PRG|[here][cf10-d-1-p]|
|  | DARP | 100 |50 |78.28 |||[here][cf10-d-50-p]|
|  | DARP | 100 |150 |75.21 |||[here][cf10-d-150-p]|
|  | DARP (reversed) | 100 |100 |80.86 |||[here][cf10-d-100-p]|
|CIFAR-100  | 400 | - |- |48.70 |Conventional settings|PRG|[here][cf100-c-400-p]|
|  | 2500 | - |- |69.81|||[here][cf100-con-2500-p]|
|  | 10000 | - |- |76.91 |||[here][cf100-con-10000-p]|
|  | 400 | - |- |48.66 |Conventional settings|PRG^Last|[here][cf100-con-400-l]|
|  | 2500 | - |- |70.03|||[here][cf100-con-2500-l]|
|  | 10000 | - |- |76.93 |||[here][cf100-con-10000-l]|
|  | - | - |50 |58.57 |CADR's protocol|PRG|[here][cf100-c-50-p]|
|  | - | - |100 |62.28 |||[here][cf100-c-100-p]|
|  | - | - |200 |59.33 |||[here][cf100-c-200-p]|
|  | - | - |50 |60.32 |CADR's protocol|PRG^Last|[here][cf100-c-50-l]|
|  | - | - |100 |62.13 |||[here][cf100-c-100-l]|
|  | - | - |200 |58.70 |||[here][cf100-c-200-l]|
|  | 2500 | 100 |- |57.56 |Ours protocol|PRG|[here][cf100-o-2500100-p]|
|  | 2500 | 200 |- |51.21 |||[here][cf100-o-2500200-p]|
|  | 2500 | 100 |- |59.40 |Ours protocol|PRG^Last|[here][cf100-o-2500100-l]|
|  | 2500 | 200 |- |42.09 |||[here][cf100-o-2500200-l]|
|mini-ImageNet | 1000| -|- |45.74 |Conventional settings|PRG|[here][mini-con-1000-p]|
| | 1000| -|- |48.63 |Conventional settings|PRG^Last|[here][mini-con-1000-l]|
| | -| -|50 |43.74 |CADR's protocol|PRG|[here][mini-c-50-p]|
| | -| - |100 |43.74 |||[here][mini-c-100-p]|
| | -| -|50 |42.22 |CADR's protocol|PRG^Last|[here][mini-c-50-l]|
| | -| - |100 |43.74 |||[here][mini-c-100-l]|
| | 1000| 40 |- |40.75 |Ours protocol|PRG|[here][mini-o-100040-p]|
| | 1000| 80 |- |35.86|||[here][mini-o-100080-p]|
| | 1000| 40 |- |39.79|Ours protocol|PRG^Last|[here][mini-o-100040-l]|
| | 1000| 80 |- |32.64|||[here][mini-o-100080-l]|

[cf10-o-4010-p]: https://1drv.ms/u/s!Ao848hI985sshjh2bBYISxTEZ7XV?e=eRKPSo
[cf100-o-2500100-p]: https://1drv.ms/u/s!Ao848hI985sshjqrNL0LjoopBC1z?e=pXsaa3
[cf100-c-50-p]: https://1drv.ms/u/s!Ao848hI985sshjyMzIqjED8QnAFz?e=cW0Gue
[cf100-c-200-p]: https://1drv.ms/u/s!Ao848hI985sshj4rvVK_PKMggLgp?e=15n0i1
[cf100-c-100-l]: https://1drv.ms/u/s!Ao848hI985sshkDQSOXOORRu5r14?e=cWO7pp
[mini-con-1000-l]: https://1drv.ms/u/s!Ao848hI985sshkJcenD9uLqNw3h2?e=gF7gZa
[mini-c-50-p]: https://1drv.ms/u/s!Ao848hI985sshkRgrVU8raF2CHEe?e=3rnf47
[mini-c-100-l]: https://1drv.ms/u/s!Ao848hI985sshkYiM9ipxeFqowBC?e=ncyW11
[mini-c-50-l]: https://1drv.ms/u/s!Ao848hI985sshkiDKPmlKnPf1xum?e=1gH6Nv
[mini-c-100-p]: https://1drv.ms/u/s!Ao848hI985sshkpQ7KRHCpejAtoU?e=vI9gkR
[mini-o-100080-l]: https://1drv.ms/u/s!Ao848hI985sshkz3877a3TG6Zr7B?e=6DbBtE
[mini-o-100040-l]: https://1drv.ms/u/s!Ao848hI985sshk4eGbncS5GRtjKa?e=QviAlu
[mini-o-100080-p]: https://1drv.ms/u/s!Ao848hI985sshk9uECYaxBojOcry?e=9AqRUR
[mini-o-100040-p]: https://1drv.ms/u/s!Ao848hI985sshlLRLNZfDOUdow07?e=TjxLU6
[mini-con-1000-p]: https://1drv.ms/u/s!Ao848hI985sshlT-gXNvizGnDEGr?e=qjkoVV
[cf100-c-200-l]: https://1drv.ms/u/s!Ao848hI985sshlYOsLfi8ADvKbEo?e=4wfqwK
[cf100-con-10000-p]: https://1drv.ms/u/s!Ao848hI985sshljSzpPs2O_jja7d?e=XMPj6N
[cf100-con-400-l]: https://1drv.ms/u/s!Ao848hI985sshloXC_uwhNUfLwj_?e=l603Ef
[cf100-con-400-p]: https://1drv.ms/u/s!Ao848hI985sshlyc01tO-hDVcqiG?e=EJuAaZ
[cf10-c-100-p]: https://1drv.ms/u/s!Ao848hI985sshl5UOBw8uHz9YG3d?e=uzhEjw
[cf10-c-50-p]: https://1drv.ms/u/s!Ao848hI985sshmAT84qZz3dLC5Rc?e=28alwc
[cf10-d-150-p]: https://1drv.ms/u/s!AtUffnM8UitOa0SmqsnYFmBOMsA?e=xd3xAo
[cf10-d-1-p]: https://1drv.ms/u/s!AtUffnM8UitOcuOFyKDat9oMs1U?e=ZXD5iQ
[cf100-c-400-p]: https://1drv.ms/u/s!AtUffnM8UitOdMMwHjm9-56tDOI?e=yK3psR
[cf10-c-40-l]: https://1drv.ms/u/s!AtUffnM8UitOdoYLbsKHKOxSMAs?e=yE8H0D
[cf10-c-250-p]: https://1drv.ms/u/s!AtUffnM8UitOeH87BaEAjKT7vdA?e=B0oRFw
[cf10-c-250-l]: https://1drv.ms/u/s!AtUffnM8UitOev36fXBIGuYKzkk?e=OlkmEV
[cf10-c-4000-l]: https://1drv.ms/u/s!AtUffnM8UitOfFiNS1dZg5-0a6k?e=Z0vcNF
[cf10-o-40102-p]: https://1drv.ms/u/s!AtUffnM8UitOfvP-Mlkhd2a_gvg?e=xa4Rhg
[cf10-o-40105-p]: https://1drv.ms/u/s!AtUffnM8UitOgQD0at-nEeqXsN-q?e=fp7CHM
[cf10-o-4020-p]: https://1drv.ms/u/s!AtUffnM8UitOgQIA7co1RPfjh7_b?e=e9z5OZ
[cf10-o-100405-p]: https://1drv.ms/u/s!AtUffnM8UitOgQTP1cX8-xjIstzh?e=5Kqhey
[cf10-o-1004010-p]: https://1drv.ms/u/s!AtUffnM8UitOgQZwo98G1rKkIWX_?e=Fx1ASK
[cf10-o-4010-l]: https://1drv.ms/u/s!AtUffnM8UitOgQg9W8CSy5-4GgeI?e=qSejNA
[cf100-c-50-l]: https://1drv.ms/u/s!AtUffnM8UitOgQr1e1vNL1duogA-?e=Hp8Vey
[cf100-c-100-p]: https://1drv.ms/u/s!AtUffnM8UitOgQxopVbWxcEFJ7h7?e=MpeqUP
[cf100-o-2500200-p]: https://1drv.ms/u/s!AtUffnM8UitOgRCWxs8kj3WvHROz?e=dgfapC
[cf100-con-10000-l]: https://1drv.ms/u/s!AtUffnM8UitOgRKWHVzUIlPrkUg3?e=JJOfji
[cf100-con-2500-l]: https://1drv.ms/u/s!AtUffnM8UitOgRSKqP6t1Cjemklf?e=X89xzs
[cf100-con-2500-p]: https://1drv.ms/u/s!AtUffnM8UitOgRalQo-9MR-Ngd9_?e=FeNWzD
[cf10-c-100-l]: https://1drv.ms/u/s!AtUffnM8UitOgRhDb_-RPfwH5pNC?e=sSSI2u
[cf10-c-50-l]: https://1drv.ms/u/s!AtUffnM8UitOgRrR_a34rnSdG-kb?e=B7uT5D
[cf10-c-20-l]: https://1drv.ms/u/s!AtUffnM8UitOgRzekRXF7tRGeGYn?e=eSfOPn
[cf10-c-20-p]: https://1drv.ms/u/s!AtUffnM8UitOgR4cQBTLj4ZV2Ke2?e=8zpPfs
[cf10-d-100-p]: https://1drv.ms/u/s!AtUffnM8UitOgSBZwtGTsjCAc7cE?e=jowKh8
[cf10-d-50-p]: https://1drv.ms/u/s!AtUffnM8UitOgSKqZstAt4K3U6yC?e=Rob3OP
[cf10-o-250200-l]: https://1drv.ms/u/s!AtUffnM8UitOgSQSqu_80fiadsYa?e=GPZ67D
[cf10-o-250100-l]: https://1drv.ms/u/s!AtUffnM8UitOgSaSHRV_dVQeoKGn?e=Z0TObf
[cf10-o-4020-l]: https://1drv.ms/u/s!AtUffnM8UitOgShaMJaXT4HgbK91?e=ekc3vz
[cf10-o-250200-p]: https://1drv.ms/u/s!AtUffnM8UitOgSqo1f_byw3wp2SA?e=p9j0cs
[cf10-o-250100-p]: https://1drv.ms/u/s!AtUffnM8UitOgSwa_m44cTqqGJJx?e=jXApqL
[cf10-con-4000-p]: https://1drv.ms/u/s!AtUffnM8UitOgS8yw07lZjsBHcvA?e=7uec44
[cf10-con-40-p]: https://1drv.ms/u/s!AtUffnM8UitOgTGbNsSRWO6Y9uiI?e=ghL1pc
[cf100-o-2500200-l]: https://1drv.ms/u/s!AtUffnM8UitOgQ4-szmII-WWtJFp?e=jjGJ9c
[cf100-o-2500100-l]: https://1drv.ms/u/s!AtUffnM8UitOgTNLFkXMaDEHA7SH?e=LHDVNP

## Citation
Please cite our paper if you find PRG useful:

```
@inproceedings{duan2023towards,
  title={Towards Semi-supervised Learning with Non-random Missing Labels},
  author={Duan, Yue and Zhao, Zhen and Qi, Lei and Zhou, Luping and Wang, Lei and Shi, Yinghuan},
  booktitle={IEEE/CVF International Conference on Computer Vision},
  year={2023}
}
```

or

```
@article{duan2023towards,
  title={Towards Semi-supervised Learning with Non-random Missing Labels},
  author={Duan, Yue and Zhao, Zhen and Qi, Lei and Zhou, Luping and Wang, Lei and Shi, Yinghuan},
  journal={arXiv preprint arXiv:2308.08872},
  year={2023}
}
```

