from __future__ import print_function, division
import os
from models.nets.net import *
import torch
import torch.nn as nn
import numpy as np
from utils import net_builder
from datasets.ssl_dataset import SSL_Dataset
from datasets.data_utils import get_data_loader


class TotalNet(nn.Module):
    def __init__(self, net_builder, num_classes, net_name):
        super(TotalNet, self).__init__()
        if net_name=='resnet18':
            base_net = net_builder(num_classes=num_classes)    
            self.feature_extractor = ResNet18(num_classes, base_net)  
        else:
            self.feature_extractor = net_builder(num_classes=num_classes)                  
        
    def forward(self, x):
        f = self.feature_extractor(x)
        return f

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--load_path', type=str, default='./saved_models/prg/model_best.pth')
    parser.add_argument('--use_train_model', action='store_true')

    '''
    Backbone Net Configurations
    '''
    parser.add_argument('--net', type=str, default='WideResNet')
    parser.add_argument('--net_from_name', type=bool, default=False)
    parser.add_argument('--depth', type=int, default=28)
    parser.add_argument('--widen_factor', type=int, default=2)
    parser.add_argument('--leaky_slope', type=float, default=0.1)
    parser.add_argument('--dropout', type=float, default=0.0)

    '''
    Data Configurations
    '''
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--dataset', type=str, default='cifar10')
    parser.add_argument('--num_classes', type=int, default=10)
    args = parser.parse_args()
    
    checkpoint_path = os.path.join(args.load_path)
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    load_model = checkpoint['train_model'] if args.use_train_model else checkpoint['eval_model']
    
    _net_builder = net_builder(args.net, 
                               args.net_from_name,
                               {'depth': args.depth, 
                                'widen_factor': args.widen_factor,
                                'leaky_slope': args.leaky_slope,
                                'dropRate': args.dropout})
    
    eval_model = TotalNet(_net_builder, args.num_classes, args.net) 
    eval_model.load_state_dict(load_model,strict=True)
    if torch.cuda.is_available():
        eval_model.cuda()
    eval_model.eval()
    feature_extractor = eval_model.feature_extractor
    
    
    if args.dataset=='miniimage':
        from datasets_mini.miniimage import  get_val_loader
        eval_loader, eval_dset= get_val_loader(dataset=args.dataset, batch_size=args.batch_size, num_workers=1, root=args.data_dir)
    else:
        _eval_dset = SSL_Dataset(name=args.dataset, train=False, data_dir=args.data_dir)
        eval_dset = _eval_dset.get_dset()
        
        eval_loader = get_data_loader(eval_dset,
                                    args.batch_size, 
                                    num_workers=1)
    
    num_classes = args.num_classes
    total_loss = 0.0
    total_acc = 0.0
    total_num = 0.0
    with torch.no_grad():
        for x, y in eval_loader:
            y = y.long()
            x, y = x.cuda(), y.cuda()
            num_batch = x.shape[0]
            total_num += num_batch

            logits, feature = feature_extractor.forward(x, ood_test=True) 
                
            max_probs, max_idx = torch.max(torch.softmax(logits, dim=-1), dim=-1)
            max_probs_sort, idx_sort = torch.sort(logits, descending=True)

            acc = torch.sum(max_idx == y)           
            total_acc += acc.detach()

    print(f"Test Accuracy: {total_acc/len(eval_dset)}")
