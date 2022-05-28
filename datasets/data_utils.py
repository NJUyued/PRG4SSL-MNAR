import torch
import torchvision
from torchvision import datasets
from torch.utils.data import sampler, DataLoader
from torch.utils.data.sampler import BatchSampler
import torch.distributed as dist
import numpy as np
import math
import os.path as osp

from datasets.DistributedProxySampler import DistributedProxySampler
from datasets.mismatch_imbalance_sampler import get_mismatched_imbalance_samples
    
def write_dis_idx(file_path_dist, mode, dis, idx, info=None):
    with open(file_path_dist, mode='a') as fff:
        if mode=='ulb':
            fff.write('\nULB_num:')
            fff.write(str(len(idx)))
            fff.write('\nULB:')
            fff.write(",".join([str(ss) for ss in dis]))
            fff.write('\nULB_index:') 
            fff.write(",".join([str(ss) for ss in idx]))
            fff.write('\n')
        elif mode=='lb':
            fff.write("="*100)
            fff.write(info)
            fff.write('\nLB_num:')
            fff.write(str(len(idx)))
            fff.write('\nLB: ')
            fff.write(",".join([str(ss) for ss in dis]))  
            fff.write('\nLB_index:')       
            fff.write(",".join([str(ss) for ss in idx]))

def split_ssl_data(dataset, data, target, num_labels, num_classes, save_path, index=None, include_lb_to_ulb=True, noisy='none', noisy_radio=0, 
                         mismatch='none',
                         n0=0,
                         gamma=0):
    """
    data & target is splitted into labeled and unlabeld data.
    
    Args
        index: If np.array of index is given, select the data[index], target[index] as labeled samples.
        include_lb_to_ulb: If True, labeled data is also included in unlabeld data
    """
    data, target = np.array(data), np.array(target)
    lb_data, lbs, lb_idx, distri_lb = sample_labeled_data(data, target, num_labels, num_classes, save_path, index, noisy, noisy_radio, mismatch, n0, gamma, dataset)
    ulb_idx = np.array(sorted(list(set(range(len(data))) - set(lb_idx)))) #unlabeled_data index of data
    ulb_data = []
    ulb_lbs = []
    dic = {'cifar10':5000,'cifar100':500}
    dic_darp = {'cifar10':3000,'cifar100':300}
    samply_dist_u = []
    unlabeled_idx = []
    nc = num_classes - 1
    file_save_dist = 'dist&index.txt'
    file_path_dist = osp.join(save_path, file_save_dist) if osp.exists(save_path) else file_save_dist    
    if mismatch=='bda' and gamma!=0:   
        assert gamma<=200 and gamma>0  
        data =  data[ulb_idx]
        target =  target[ulb_idx]
        all = 0      
        for c in range(num_classes):
            idx = np.where(target == c)[0]
            lt_len = math.ceil(dic[dataset] * math.pow(gamma,-((nc-c)/nc)))
            if lt_len<len(idx):
                idx = np.random.choice(idx, lt_len, False)    
                samply_dist_u.append(lt_len)
            else:
                idx = np.random.choice(idx, len(idx), False) 
                samply_dist_u.append(len(idx))
            # temp_data = data[idx]
            # temp_lb = target[idx]
            # ulb_data.extend(temp_data[0:lt_len])
            # ulb_lbs.extend(temp_lb[0:lt_len])
            ulb_data.extend(data[idx])
            ulb_lbs.extend(target[idx])
            unlabeled_idx.extend(idx)
            all += lt_len     
        write_dis_idx(file_path_dist=file_path_dist,mode='ulb',dis=samply_dist_u,idx=unlabeled_idx)
        if include_lb_to_ulb:
            print(np.array(samply_dist_u)/all)
            print(samply_dist_u)
            return lb_data, lbs, np.concatenate((ulb_data, lb_data), axis=0), np.concatenate((ulb_lbs, lbs), axis=0), None
        else:
            return lb_data, lbs, data[ulb_idx], target[ulb_idx], ulb_idx
    if mismatch=='lt' and gamma!=0:   
        assert gamma<=100 and gamma>0  
        all = 0         
        for c in range(num_classes):
            idx = np.where(target == c)[0]
            lt_len = math.ceil(dic[dataset] * math.pow(gamma,-((nc-c)/nc)))
            if lt_len<len(idx):
                idx = np.random.choice(idx, lt_len, False)    
                samply_dist_u.append(lt_len)
            else:
                idx = np.random.choice(idx, len(idx), False) 
                samply_dist_u.append(len(idx))
            ulb_data.extend(data[idx])
            ulb_lbs.extend(target[idx])
            unlabeled_idx.extend(idx)
            all += lt_len     
        write_dis_idx(file_path_dist=file_path_dist,mode='ulb',dis=samply_dist_u,idx=unlabeled_idx)
        if include_lb_to_ulb:
            print(np.array(samply_dist_u)/all)
            print(samply_dist_u)
            return lb_data, lbs, np.concatenate((ulb_data, lb_data), axis=0), np.concatenate((ulb_lbs, lbs), axis=0), None
        else:
            return lb_data, lbs, data[ulb_idx], target[ulb_idx], ulb_idx
    elif mismatch=='DARP_reversed' or mismatch=='DARP' and gamma!=0:   
        assert gamma<=200 and gamma>0  
        data =  data[ulb_idx]
        target =  target[ulb_idx]
        for c in range(num_classes):
            idx = np.where(target == c)[0]
            lt_len = math.ceil(dic_darp[dataset] * math.pow(gamma,-((nc-c)/nc))) if mismatch=='DARP_reversed' else math.ceil(dic_darp[dataset] * math.pow(gamma,-(c/nc)))
            idx = np.random.choice(idx, lt_len, False) 
            ulb_data.extend(data[idx])
            ulb_lbs.extend(target[idx]) 
            samply_dist_u.append(len(idx)) 
            unlabeled_idx.extend(idx) 
        write_dis_idx(file_path_dist=file_path_dist,mode='ulb',dis=samply_dist_u,idx=unlabeled_idx)    
        if include_lb_to_ulb:
            return lb_data, lbs, np.concatenate((ulb_data, lb_data), axis=0), np.concatenate((ulb_lbs, lbs), axis=0), None
        else:
            return lb_data, lbs, data[ulb_idx], target[ulb_idx], ulb_idx
    elif mismatch=='ood':
        assert n0>0
        data =  data[ulb_idx]
        target =  target[ulb_idx]
        # c = np.random.choice([i for i in range(num_classes)], n0, False) 
        c = [1,2,3,4,5,6,7,8]
        for cc in c:
            idx = np.where(target == cc)[0]
            ulb_data.extend(data[idx])
            ulb_lbs.extend(target[idx]) 
            samply_dist_u.append(len(idx)) 
            unlabeled_idx.extend(idx) 
        write_dis_idx(file_path_dist=file_path_dist,mode='ulb',dis=samply_dist_u,idx=unlabeled_idx) 
        return lb_data, lbs, ulb_data, ulb_lbs, unlabeled_idx
    else:
        for c in range(num_classes):
            idx = np.where(target == c)[0]
            samply_dist_u.append(len(idx)-distri_lb[c])
        # save dist&index
        write_dis_idx(file_path_dist=file_path_dist,mode='ulb',dis=samply_dist_u,idx=ulb_idx)

        if include_lb_to_ulb:
            return lb_data, lbs, data, target, None
        else:
            return lb_data, lbs, data[ulb_idx], target[ulb_idx], ulb_idx
    
    
def sample_labeled_data(data, target, 
                         num_labels,
                         num_classes,
                         save_path,
                         index=None,
                         noisy='none', 
                         noisy_radio=0,
                         mismatch='none',
                         n0=0,
                         gamma=0,
                         dataset='cifar10'):
    '''
    samples for labeled data
    (sampling with balanced ratio over classes)
    '''
    assert num_labels % num_classes == 0
    
    if not index is None:
        index = np.array(index, dtype=np.int32)
        return data[index], target[index], index
    
    dic_lt = {'cifar10':5000,'cifar100':500}
    samples_per_class = int(num_labels / num_classes)
    lb_data = []
    lbs = []
    lb_idx = []
    distri_lb = []
    nc = num_classes-1
    if noisy_radio==0:
        if mismatch=='MNAR':
            for c in range(num_classes):
                idx = np.where(target == c)[0]            
                lt_len = math.ceil(gamma * math.pow(gamma,-(c/nc)))
                distri_lb.append(lt_len)
                idx = np.random.choice(idx, lt_len, False)
                lb_idx.extend(idx)
                lb_data.extend(data[idx])
                lbs.extend(target[idx])
        elif mismatch=='DARP' or mismatch=='DARP_reversed':
            dic = {'cifar10':1500,'cifar100':150,'stl10':450}
            for c in range(num_classes):
                idx = np.where(target == c)[0]            
                lt_len = math.ceil(dic[dataset] * math.pow(n0,-(c/nc)))
                distri_lb.append(lt_len)
                idx = np.random.choice(idx, lt_len, False) 
                lb_idx.extend(idx)
                lb_data.extend(data[idx])
                lbs.extend(target[idx])
        elif mismatch=='bda' and n0!=0:         
            distri_lb = get_mismatched_imbalance_samples(n0, num_classes, num_labels)
            for c in range(num_classes):
                idx = np.where(target == c)[0]
                idx = np.random.choice(idx, distri_lb[c], False)
                lb_idx.extend(idx)
                temp_data = data[idx]
                temp_lb = target[idx]
                lb_data.extend(temp_data)
                lbs.extend(temp_lb)
        elif mismatch=='lt' and n0!=0 and gamma!=0: 
            for c in range(num_classes):
                idx = np.where(target == c)[0]
                lt_len = math.ceil(dic_lt[dataset] * math.pow(gamma,-((nc-c)/nc)))

                idx = np.random.choice(idx, math.ceil(lt_len/n0), False) 
                distri_lb.append(len(idx))
                lb_idx.extend(idx)
                temp_data = data[idx]
                temp_lb = target[idx]
                lb_data.extend(temp_data)
                lbs.extend(temp_lb)
        else:
            for c in range(num_classes):
                distri_lb.append(samples_per_class)
                idx = np.where(target == c)[0]
                idx = np.random.choice(idx, samples_per_class, False)
                lb_idx.extend(idx)
                temp_data = data[idx]
                temp_lb = target[idx]
                lb_data.extend(temp_data)
                lbs.extend(temp_lb)
        # save dist&index
        file_save_dist = 'dist&index.txt'
        file_path_dist = osp.join(save_path, file_save_dist) if osp.exists(save_path) else file_save_dist
        info = '\nmismatch: {}, N0: {}, gamma: {}'.format(mismatch,'none' if n0==0 else str(n0),str(1) if gamma==0 else str(gamma))
        write_dis_idx(file_path_dist=file_path_dist,mode='lb',dis=distri_lb,idx=lb_idx,info=info)

    else:
        assert noisy_radio >= 0 and noisy_radio <= 100
        assert noisy != 'none'
        num_noisy = int((noisy_radio/100.0) * samples_per_class)
        print('---')
        print(int(num_noisy))
        print('---')
        ays_map = [0,1,0,5,7,3,6,7,8,1]
        for c in range(num_classes):
            idx = np.where(target == c)[0]
            idx = np.random.choice(idx, samples_per_class, False)
            lb_idx.extend(idx)
            
            temp_lb = target[idx].copy()
            idx_noisy = np.random.choice([i for i in range(len(temp_lb))], num_noisy, False)
            for i in range(num_noisy):
                if noisy=='exc':
                    print(str(idx[i])+'->',end='')  
                    print(str(temp_lb[idx_noisy[i]])+'-->',end='')  
                    temp_lb[idx_noisy[i]] = (temp_lb[idx_noisy[i]]+np.random.randint(1,num_classes,size=1)[0])%num_classes
                    print(str(temp_lb[idx_noisy[i]])) 
                if noisy=='inc':
                    print(str(idx[i])+'->',end='')  
                    print(str(temp_lb[idx_noisy[i]])+'-->',end='')                    
                    temp_lb[idx_noisy[i]] = (temp_lb[idx_noisy[i]]+np.random.randint(0,num_classes,size=1)[0])%num_classes
                    print(str(temp_lb[idx_noisy[i]])) 
                if noisy=='asy':   
                    print(str(idx[i])+'->',end='') 
                    print(str(temp_lb[idx_noisy[i]])+'-->'+str(ays_map[c]))      
                    temp_lb[idx_noisy[i]] = ays_map[c]

            lb_data.extend(data[idx])
            lbs.extend(temp_lb)
    return np.array(lb_data), np.array(lbs), np.array(lb_idx), np.array(distri_lb)


def get_sampler_by_name(name):
    '''
    get sampler in torch.utils.data.sampler by name
    '''
    sampler_name_list = sorted(name for name in torch.utils.data.sampler.__dict__ 
                      if not name.startswith('_') and callable(sampler.__dict__[name]))
    try:
        if name == 'DistributedSampler':
            return torch.utils.data.distributed.DistributedSampler
        else:
            return getattr(torch.utils.data.sampler, name)
    except Exception as e:
        print(repr(e))
        print('[!] select sampler in:\t', sampler_name_list)

        
def get_data_loader(dset,
                    batch_size = None,
                    shuffle = False,
                    num_workers = 4,
                    pin_memory = True,
                    data_sampler = None,
                    replacement = True,
                    num_epochs = None,
                    num_iters = None,
                    generator = None,
                    drop_last=True,
                    distributed=False):
    """
    get_data_loader returns torch.utils.data.DataLoader for a Dataset.
    All arguments are comparable with those of pytorch DataLoader.
    However, if distributed, DistributedProxySampler, which is a wrapper of data_sampler, is used.
    
    Args
        num_epochs: total batch -> (# of batches in dset) * num_epochs 
        num_iters: total batch -> num_iters
    """
    
    assert batch_size is not None
        
    if data_sampler is None:
        return DataLoader(dset, batch_size=batch_size, shuffle=shuffle, 
                          num_workers=num_workers, pin_memory=pin_memory)
    
    else:
        if isinstance(data_sampler, str):
            data_sampler = get_sampler_by_name(data_sampler)
        
        if distributed:
            assert dist.is_available()
            num_replicas = dist.get_world_size()
        else:
            num_replicas = 1
        
        if (num_epochs is not None) and (num_iters is None):
            num_samples = len(dset)*num_epochs
        elif (num_epochs is None) and (num_iters is not None):
            num_samples = batch_size * num_iters * num_replicas
        else:
            num_samples = len(dset)

        if data_sampler.__name__ == 'RandomSampler':    
            data_sampler = data_sampler(dset, replacement, num_samples, generator)
        else:
            raise RuntimeError(f"{data_sampler.__name__} is not implemented.")
        
        if distributed:
            '''
            Different with DistributedSampler, 
            the DistribuedProxySampler does not shuffle the data (just wrapper for dist).
            '''
            data_sampler = DistributedProxySampler(data_sampler)

        batch_sampler = BatchSampler(data_sampler, batch_size, drop_last)
        return DataLoader(dset, batch_sampler=batch_sampler, 
                          num_workers=num_workers, pin_memory=False)

    
def get_onehot(num_classes, idx):
    onehot = np.zeros([num_classes], dtype=np.float32)
    onehot[idx] += 1.0
    return onehot
