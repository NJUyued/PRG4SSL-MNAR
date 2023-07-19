from math import trunc
import numpy as np
import math 
import os.path as osp

import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import BatchSampler

from .augmentation.randaugment import RandAugment
from .data_utils import get_sampler_by_name, get_data_loader, get_onehot, split_ssl_data, write_dis_idx
from .dataset import BasicDataset

import torchvision
from torchvision import datasets, transforms

mean, std = {}, {}
mean['cifar10'] = [x / 255 for x in [125.3, 123.0, 113.9]]
mean['cifar100'] = [x / 255 for x in [129.3, 124.1, 112.4]]
mean['stl10'] = [x / 255 for x in [0.44087965, 0.42790789, 0.38678672]]
mean['svhn'] = [x / 255 for x in [0.4380, 0.4440, 0.4730]]

std['cifar10'] = [x / 255 for x in [63.0, 62.1, 66.7]]
std['cifar100'] = [x / 255 for x in [68.2,  65.4,  70.4]]
std['stl10'] = [x / 255 for x in [0.23089217, 0.22623343, 0.22368798]]
std['svhn'] = [x / 255 for x in [0.1751, 0.1771, 0.1744]]



def get_transform(mean, std, name, train=True):
    if name=='stl10':
        if train:
            return transforms.Compose([transforms.RandomHorizontalFlip(),
                                        transforms.RandomCrop(96, padding=4),
                                        transforms.ToTensor(), 
                                        transforms.Normalize(mean, std)])
        else:
            return transforms.Compose([transforms.ToTensor(), 
                                        transforms.Normalize(mean, std)])    
    elif name=='svhn':
        if train:
            return transforms.Compose([transforms.RandomCrop(32, padding=4),
                                        transforms.ToTensor(), 
                                        transforms.Normalize(mean, std)])
        else:
            return transforms.Compose([transforms.ToTensor(), 
                                        transforms.Normalize(mean, std)])
    else:
        if train:
            return transforms.Compose([transforms.RandomHorizontalFlip(),
                                        transforms.RandomCrop(32, padding=4),
                                        transforms.ToTensor(), 
                                        transforms.Normalize(mean, std)])
        else:
            return transforms.Compose([transforms.ToTensor(), 
                                        transforms.Normalize(mean, std)])

    
class SSL_Dataset:
    """
    SSL_Dataset class gets dataset (cifar10, cifar100) from torchvision.datasets,
    separates labeled and unlabeled data,
    and return BasicDataset: torch.utils.data.Dataset (see datasets.dataset.py)
    """
    def __init__(self,
                 name='cifar10',
                 train=True,
                 num_classes=10,
                 data_dir='./data',
                 mismatch='none',
                 n0=0,
                 gamma=0):
        """
        Args
            name: name of dataset in torchvision.datasets (cifar10, cifar100)
            train: True means the dataset is training dataset (default=True)
            num_classes: number of label classes
            data_dir: path of directory, where data is downloaed or stored.
        """
        
        self.name = name
        self.train = train
        self.data_dir = data_dir
        self.num_classes = num_classes
        self.transform = get_transform(mean[name], std[name], name, train)
        self.mismatch = mismatch
        self.n0 = n0
        self.gamma = gamma
        
    def get_data(self):
        """
        get_data returns data (images) and targets (labels)
        """
        if self.name=='stl10':
            if self.train:
                dset = getattr(torchvision.datasets, self.name.upper())
                dset_lb = dset(self.data_dir, split='train', download=True)
                dset_lb_ulb = dset(self.data_dir, split='unlabeled', download=True)
                    
                data_lb, targets_lb = dset_lb.data, dset_lb.labels
                data_lb_ulb, targets_lb_ulb = dset_lb_ulb.data, dset_lb_ulb.labels
                return data_lb, targets_lb, data_lb_ulb, targets_lb_ulb
            else:
                dset = getattr(torchvision.datasets, self.name.upper())
                dset = dset(self.data_dir, split='test', download=True)
                data, targets = dset.data, dset.labels
                return data, targets
        elif self.name=='svhn':
            if self.train:
                dset = getattr(torchvision.datasets, self.name.upper())
                dset = dset(self.data_dir, split='train', download=True)      
                data, targets = dset.data, dset.labels
                return data, targets
            else:
                dset = getattr(torchvision.datasets, self.name.upper())
                dset = dset(self.data_dir, split='test', download=True)
                data, targets = dset.data, dset.labels
                return data, targets
        else:
            dset = getattr(torchvision.datasets, self.name.upper())
            dset = dset(self.data_dir, train=self.train, download=True)      
            data, targets = dset.data, dset.targets
            return data, targets
        
        
    
    
    def get_dset(self, use_strong_transform=False, 
                 strong_transform=None, onehot=False):
        """
        get_dset returns class BasicDataset, containing the returns of get_data.
        
        Args
            use_strong_tranform: If True, returned dataset generates a pair of weak and strong augmented images.
            strong_transform: list of strong_transform (augmentation) if use_strong_transform is True
            onehot: If True, the label is not integer, but one-hot vector.
        """
        
        data, targets = self.get_data()
        num_classes = self.num_classes
        transform = self.transform
        data_dir = self.data_dir
        
        return BasicDataset(data, targets, num_classes, transform, 
                            use_strong_transform, strong_transform, onehot)
    
    
    def get_ssl_dset(self, num_labels, save_path, index=None, include_lb_to_ulb=True,
                            use_strong_transform=True, strong_transform=None, 
                            onehot=False,
                            ):
        """
        get_ssl_dset split training samples into labeled and unlabeled samples.
        The labeled data is balanced samples over classes.
        
        Args:
            num_labels: number of labeled data.
            index: If index of np.array is given, labeled data is not randomly sampled, but use index for sampling.
            include_lb_to_ulb: If True, consistency regularization is also computed for the labeled data.
            use_strong_transform: If True, unlabeld dataset returns weak & strong augmented image pair. 
                                  If False, unlabeled datasets returns only weak augmented image.
            strong_transform: list of strong transform (RandAugment in FixMatch)
            oenhot: If True, the target is converted into onehot vector.
            
        Returns:
            BasicDataset (for labeled data), BasicDataset (for unlabeld data)
        """
        
        if self.name=='stl10':
            data_lb, targets_lb, data_lb_ulb, targets_lb_ulb = self.get_data()
            num_classes = self.num_classes
            transform = self.transform
            data_dir = self.data_dir
            distri_lb = []
            if self.mismatch=='darp':
                lb_idx = []
                lb_data = []
                lbs = []
                for c in range(num_classes):
                    idx = np.where(targets_lb == c)[0]
                    idx = np.random.choice(idx, int(num_labels/num_classes), False)             
                    lt_len = round(450 * math.pow(self.gamma,-(c/9)))
                    distri_lb.append(lt_len)
                    temp_data = data_lb[idx]
                    temp_lb = targets_lb[idx]
                    lb_idx.extend(idx[0:lt_len])
                    lb_data.extend(temp_data[0:lt_len])
                    lbs.extend(temp_lb[0:lt_len])
                data_lb = np.array(lb_data)
                targets_lb = np.array(lbs)
                # ulb_idx = np.array(sorted(list(set(range(len(data_lb_ulb))) - set(np.array(lb_idx))))) #unlabeled_data index of data
                # data_lb_ulb =  data_lb_ulb[ulb_idx]
                # targets_lb_ulb =  targets_lb_ulb[ulb_idx]
                
                # save dist&index
                file_save_dist = 'dist&index.txt'
                file_path_dist = osp.join(save_path, file_save_dist) if osp.exists(save_path) else file_save_dist
                info = '\nlabels: {}, mismatch: {}, N0: {}, gamma: {}'.format(num_labels,self.mismatch,'none' if self.n0==0 else str(self.n0),str(1) if self.gamma==0 else str(self.gamma))
                write_dis_idx(file_path_dist=file_path_dist,mode='lb',dis=distri_lb,idx=lb_idx,info=info)

                if include_lb_to_ulb:
                    ulb_dset = BasicDataset(np.concatenate((data_lb_ulb, data_lb), axis=0), 
                                np.concatenate((targets_lb_ulb, targets_lb), axis=0), num_classes, 
                                transform, use_strong_transform, strong_transform, onehot)
                else:
                    ulb_dset = BasicDataset(data_lb_ulb, targets_lb_ulb, num_classes, 
                                transform, use_strong_transform, strong_transform, onehot)
            else:  
                ulb_dset = BasicDataset(data_lb_ulb, targets_lb_ulb, num_classes, 
                                transform, use_strong_transform, strong_transform, onehot)
            lb_dset = BasicDataset(data_lb, targets_lb, num_classes, 
                                transform, False, None, onehot)
            eval_ulb_dset = BasicDataset(data_lb_ulb, targets_lb_ulb, num_classes, 
                                transform, use_strong_transform, strong_transform, onehot)
            return lb_dset, ulb_dset, eval_ulb_dset
        else:
            name = self.name
            data, targets = self.get_data()
            num_classes = self.num_classes
            transform = self.transform
            data_dir = self.data_dir

            lb_data, lb_targets, ulb_data, ulb_targets, ulb_idx = split_ssl_data(name, data, targets, 
                                                                        num_labels, num_classes, save_path,
                                                                        index, include_lb_to_ulb,
                                                                        self.mismatch,
                                                                        self.n0,
                                                                        self.gamma)
            
            lb_dset = BasicDataset(lb_data, lb_targets, num_classes, 
                                transform, False, None, onehot)
            
            ulb_dset = BasicDataset(ulb_data, ulb_targets, num_classes, 
                                transform, use_strong_transform, strong_transform, onehot)

            
            return lb_dset, ulb_dset
