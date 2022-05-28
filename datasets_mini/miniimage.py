from PIL import Image, ImageFilter

import os, sys
import random
import numpy as np
import pandas as pd
import os.path as osp

import torch
import torchvision
from torch.utils.data import Dataset
from torchvision import transforms
# from torchvision.transforms import InterpolationMode

from datasets_mini.randaugment import RandAugment
from torch.utils.data import BatchSampler, RandomSampler

from datasets_mini.mismatch_imbalance_sampler import get_mismatched_imbalance_samples
from datasets_mini.match_sampler import get_matched_lt_samples


def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)


class TwoCropsTransform:
    """Take 2 random augmentations of one image."""

    def __init__(self, trans_weak, trans_strong):
        self.trans_weak = trans_weak
        self.trans_strong = trans_strong

    def __call__(self, x):
        x1 = self.trans_weak(x)
        x2 = self.trans_strong(x)
        return [x1, x2]


class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


class MiniImage(Dataset):
    def __init__(self, root, indexs, train=True, mode=None, isize=84):
        super(MiniImage, self).__init__()
        self.mode = mode
        # 1. data
        self.root = root
        if train:
            df_data = pd.read_csv(os.path.join(self.root, 'train.csv'))
        else:
            df_data = pd.read_csv(os.path.join(self.root, 'test.csv'))

        self.image_names = df_data['filename'].values  # ndarray
        self.labels = df_data['label'].values

        # refer to featmatch
        self.isize = isize
        self.Tpre = transforms.Compose([transforms.Resize(isize, Image.LANCZOS), transforms.CenterCrop(isize)])
        # self.Tpre = transforms.Compose([transforms.Resize(isize, transforms.InterpolationMode.LANCZOS), transforms.CenterCrop(isize)])

        # 2.indexes
        self.indexs = indexs
        if indexs is not None:
            self.image_names = self.image_names[indexs]
            self.labels = np.array(self.labels)[indexs]
            # print("="*50, self.labels.dtype)

        # 3. trans
        # mean = [0.4776, 0.4491, 0.4020] # test  --- # calculated by myself
        # std = [0.2227, 0.2183, 0.2174]
        self.mean = [0.4779, 0.4497, 0.4018] # train
        self.std = [0.2229, 0.2181, 0.2175]

        # transforms
        if self.mode == 'unlabeled_mt':
            self.trans = self.get_mt_transforms()
        elif self.mode == 'unlabeled_fixmatch':
            self.trans = self.get_fixmatch_transforms()
        elif self.mode == 'labeled':
            self.trans = self.get_labeled_transforms()
        else:
            self.trans = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std),
            ])

    def get_labeled_transforms(self):
        trans_labeled = transforms.Compose([
            transforms.RandomCrop(size=self.isize,
                                  padding=int(self.isize*0.125),
                                  padding_mode='reflect'),  # T.PadandRandomCrop(border=4, cropsize=(32, 32)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std),
        ])
        return trans_labeled

    def get_mt_transforms(self, use_blur=False):
        if use_blur:
            trans_weak = transforms.Compose([
                transforms.RandomCrop(size=self.isize,
                                    padding=int(self.isize*0.125),
                                    padding_mode='reflect'),
                transforms.RandomHorizontalFlip(),
                transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.mean, std=self.std),
            ])
        else:
            trans_weak = transforms.Compose([
                transforms.RandomCrop(size=self.isize,
                                    padding=int(self.isize*0.125),
                                    padding_mode='reflect'),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.mean, std=self.std),
            ])

        return TwoCropsTransform(trans_weak, trans_weak)

    def get_fixmatch_transforms(self):
        trans_weak = transforms.Compose([
            transforms.RandomCrop(size=self.isize,
                                  padding=int(self.isize*0.125),
                                  padding_mode='reflect'),  # T.PadandRandomCrop(border=4, cropsize=(32, 32)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std),
        ])
        trans_strong = transforms.Compose([
            transforms.RandomCrop(size=self.isize,
                                  padding=int(self.isize*0.125),
                                  padding_mode='reflect'),
            transforms.RandomHorizontalFlip(p=0.5),
            RandAugment(3, 5),
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std),
        ])
        return TwoCropsTransform(trans_weak, trans_strong)
    
    def __getitem__(self, index):
        img_name = self.image_names[index]
        img_path = os.path.join(self.root, "images", img_name)
        image = Image.open(img_path).convert('RGB')
        # preprocessing follow featmatch
        image = self.Tpre(image)
        target = self.labels[index]
        if self.mode == 'labeled' or self.mode == None:
            return self.trans(image), target
        else:
            return self.trans(image), target, index


    def get_index(self):
        return self.indexs

    def __len__(self):
        return len(self.labels)


def x_u_split(save_path, labels, L, n_class=100, n_each_class=500, 
            flag_mismatch=False, n_labeled_class_max=None, 
            long_tail_gamma=1, long_tail_label_ratio=None, include_lb_to_ulb=True):
    samply_dist = None
    samply_dist_u = None

    if L is not None:
        assert L in [400, 1000, 2000, 2500, 4000, 10000]
    if flag_mismatch:
        samply_dist = get_mismatched_imbalance_samples(n_labeled_class_max, n_class, L)
    else:
        if long_tail_gamma == 1 or long_tail_gamma == 0:  
            labeled_frac = L / (n_each_class * n_class)
            print(labeled_frac)
            samply_dist = get_matched_lt_samples(1, n_each_class, n_class, labeled_frac)
            print(samply_dist)
            # return standard dataset
        else:
            # samply_dist_u = get_matched_lt_samples(long_tail_gamma, n_each_class, n_class, None)
            # samply_dist = get_mismatched_imbalance_samples(n_labeled_class_max, n_class, L)
            samply_dist = get_matched_lt_samples(long_tail_gamma, n_each_class, n_class, None)
    
    # distribution
    label_dist = np.array(samply_dist) / sum(samply_dist)
    label_dist = label_dist.astype(np.float32)
    assert len(label_dist) == n_class
    real_L = sum(samply_dist)

    # get index
    labeled_idx, unlabeled_idx = [], []
    for i in range(n_class):
        indices = np.where(labels == i)[0]
        np.random.shuffle(indices)
        n_labels = samply_dist[i]
        if samply_dist_u is not None:
            n_unlabels = samply_dist_u[i]
        else:
            n_unlabels = len(indices)
        
        if include_lb_to_ulb:
            inds_x, inds_u = indices[:n_labels], indices[:n_unlabels]
        else:
            inds_x, inds_u = indices[:n_labels], indices[n_labels:n_unlabels]

        labeled_idx.extend(inds_x.tolist())
        unlabeled_idx.extend(inds_u.tolist())
    print("====== data size:", len(labeled_idx), len(unlabeled_idx))
    
    # save dists
    file_save_dist = 'dist&index.txt'
    file_path_dist = osp.join(save_path, file_save_dist) if osp.exists(save_path) else file_save_dist
    with open(file_path_dist, mode='a') as fff:
        fff.write("="*100)
        fff.write('\nLB: ')
        fff.write(",".join([str(ss) for ss in samply_dist]))
        if samply_dist_u is not None:
            fff.write('\nULB:')
            fff.write(",".join([str(ss) for ss in samply_dist_u]))
        fff.write('\n')

    # save labels
    file_save_dist = 'dist&index.txt'
    file_path_dist = osp.join(save_path, file_save_dist) if osp.exists(save_path) else file_save_dist
    info = '\nlabel: {}, mismatch: {}, N0: {}, gamma: {}'.format(str(real_L),'bem' if flag_mismatch else 'none','none' if n_labeled_class_max==0 else str(n_labeled_class_max),str(1) if long_tail_gamma==0 else str(long_tail_gamma))
    with open(file_path_dist, mode='a') as filename:     
        filename.write('\n')
        filename.write(",".join([str(ss) for ss in labeled_idx]))
        filename.write('\n')

    return labeled_idx, unlabeled_idx, label_dist


def get_train_loader(save_path, dataset, batch_size, mu, n_iters_per_epoch, 
        L=100, num_workers=1, root='data', augmethod='fixmatch', 
        n_each_class=500, flag_mismatch=False, n_labeled_class_max=None, 
        long_tail_gamma=1, long_tail_label_ratio=None):

    assert dataset.lower().startswith("miniimage")
    df_tmp = pd.read_csv(os.path.join(root, "train.csv"))
    tmp_labels = df_tmp['label'].values
    num_classes = 100
    
    labeled_idx, unlabeled_idx, dist_label = x_u_split(save_path, tmp_labels, L, num_classes, 
        n_each_class=n_each_class, flag_mismatch=flag_mismatch, n_labeled_class_max=n_labeled_class_max, 
        long_tail_gamma=long_tail_gamma, long_tail_label_ratio=long_tail_label_ratio)

    ds_x = MiniImage(
        root=root, 
        indexs=labeled_idx, 
        train=True,
        mode="labeled",
    )
    sampler_x = RandomSampler(ds_x, replacement=True, num_samples=n_iters_per_epoch * batch_size)
    batch_sampler_x = BatchSampler(sampler_x, batch_size, drop_last=True)  # yield a batch of samples one time
    dl_x = torch.utils.data.DataLoader(
        ds_x,
        batch_sampler=batch_sampler_x,
        num_workers=num_workers,
        pin_memory=True,
        worker_init_fn=worker_init_fn
    )

    ds_u = MiniImage(
        root=root, 
        indexs=unlabeled_idx, 
        train=True, 
        mode='unlabeled_{}'.format(augmethod)
    )
    sampler_u = RandomSampler(ds_u, replacement=True, num_samples=mu * n_iters_per_epoch * batch_size)
    batch_sampler_u = BatchSampler(sampler_u, batch_size * mu, drop_last=True)
    sampler_u_eval = RandomSampler(ds_u, replacement=True, num_samples=len(ds_u))
    batch_sampler_u_eval = BatchSampler(sampler_u_eval, 1024, drop_last=True)
    dl_u = torch.utils.data.DataLoader(
        ds_u,
        batch_sampler=batch_sampler_u,
        num_workers=num_workers,
        pin_memory=True,
        worker_init_fn=worker_init_fn
    )
    dl_u_eval = torch.utils.data.DataLoader(
        ds_u,
        batch_sampler=batch_sampler_u_eval,
        num_workers=num_workers,
        pin_memory=True,
        worker_init_fn=worker_init_fn
    )
    return dl_x, dl_u, dl_u_eval, dist_label


def get_val_loader(dataset, batch_size, num_workers=1, pin_memory=True, root='data'):
    assert dataset.lower().startswith("miniimage")
    ds = MiniImage(root, indexs=None, train=False, mode=None)

    dl = torch.utils.data.DataLoader(
        ds,
        shuffle=False,
        batch_size=batch_size,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=True,
        worker_init_fn=worker_init_fn
    )
    
    return dl, ds
