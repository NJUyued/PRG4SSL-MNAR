from PIL import Image

import os, sys
import numpy as np
import pandas as pd

import torch
import torchvision
from torch.utils.data import Dataset
from torchvision import transforms


# from datasets import transform as T
# from datasets.randaugment import RandomAugment
from datasets.randaugment import RandAugment
from torch.utils.data import BatchSampler, RandomSampler


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

    
class STL10SSL(torchvision.datasets.STL10):
    def __init__(self, root, train=True, unlabel=False, twocrop=False, 
        download=False, isize=96, numfolds=5, unlabel_aug="fixmatch", include_lb_to_ulb=True):
        
        # transformations
        mean = [0.44087965, 0.42790789, 0.38678672] # train
        std = [0.23089217, 0.22623343, 0.22368798]
        # mean = [0.44723064, 0.43964267, 0.40495682] # test
        # std = [0.22490758, 0.22174002, 0.22371016]

        trans_weak = transforms.Compose([
            transforms.RandomCrop(size=isize,
                                  padding=int(isize*0.125),
                                  padding_mode='reflect'),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        
        trans_strong = transforms.Compose([
            transforms.RandomCrop(size=isize,
                                  padding=int(isize*0.125),
                                  padding_mode='reflect'),
            transforms.RandomHorizontalFlip(p=0.5),
            RandAugment(3, 5),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        
        trans_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        
        # set configs of datasets
        data_split="test"
        data_trans = trans_test
        if train and not unlabel: # 1) labeled data for training
            data_split = "train"
            data_trans = trans_weak

        elif train and unlabel:  # 2) unlabeled data 
            if include_lb_to_ulb:
                data_split = "train+unlabeled"
            else:
                data_split = "unlabeled"
            numfolds = None
            if twocrop:
                if unlabel_aug == "fixmatch":
                    data_trans = TwoCropsTransform(trans_weak, trans_strong)
                else:
                    data_trans = TwoCropsTransform(trans_weak, trans_weak)
            else:
                data_trans = trans_weak
        elif not train:
            data_split = "test" # 3) testing data
            data_trans = trans_test
        
        # invoke the stl function
        if train:
            super().__init__(root, split=data_split, transform=data_trans, folds=numfolds, target_transform=None, download=download)
        else:
            super().__init__(root, split=data_split, transform=data_trans, target_transform=None, download=download)


def get_val_loader(dataset, batch_size, num_workers=1, pin_memory=True, root='data'):
    assert dataset.lower() == "stl10"

    ds = STL10SSL(root=root, train=False, unlabel=False, twocrop=False, download=False, isize=96)
    dl = torch.utils.data.DataLoader(
        ds,
        shuffle=False,
        batch_size=batch_size,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        worker_init_fn = worker_init_fn,
    )
    return dl


# def get_train_loader(dataset, batch_size, mu, n_iters_per_epoch, 
#         L=100, num_workers=1, root='data', augmethod='fixmatch', 
#         n_each_class=500, flag_mismatch=False, n_labeled_class_max=None, 
#         long_tail_gamma=1, long_tail_label_ratio=None):

def get_train_loader(dataset, batch_size, mu, n_iters_per_epoch, 
        L=1, num_workers=1, root='./data', augmethod='fixmatch', 
        n_each_class=None, flag_mismatch=None, n_labeled_class_max=None, 
        long_tail_gamma=None, long_tail_label_ratio=None):
    assert dataset.lower() == "stl10"
    folds = L
    assert folds in [0,1,2,3,4,5,6,7,8,9]

    ds_x = STL10SSL(root=root, 
        train=True, unlabel=False, 
        twocrop=False, download=False, 
        isize=96, numfolds=folds)
    sampler_x = RandomSampler(ds_x, replacement=True, num_samples=n_iters_per_epoch * batch_size)
    batch_sampler_x = BatchSampler(sampler_x, batch_size, drop_last=True)
    dl_x = torch.utils.data.DataLoader(
        ds_x,
        batch_sampler=batch_sampler_x,
        num_workers=num_workers,
        pin_memory=True,
        worker_init_fn=worker_init_fn,
    )
    
    ds_u = STL10SSL(root=root, 
        train=True, unlabel=True, 
        twocrop=True, download=False, 
        isize=96, numfolds=folds, 
        unlabel_aug=augmethod, 
        include_lb_to_ulb=True)
    sampler_u = RandomSampler(ds_u, replacement=True, num_samples=mu * n_iters_per_epoch * batch_size)
    batch_sampler_u = BatchSampler(sampler_u, batch_size * mu, drop_last=True)
    dl_u = torch.utils.data.DataLoader(
        ds_u,
        batch_sampler=batch_sampler_u,
        num_workers=num_workers,
        pin_memory=True,
        worker_init_fn = worker_init_fn,
    )
    print("="*20, "data size:", len(ds_x), len(ds_u))

    return dl_x, dl_u, None
