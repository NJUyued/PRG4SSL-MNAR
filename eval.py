import os
import torch
import torch.nn as nn

from models.nets.net import *
from utils import net_builder
from datasets.ssl_dataset import SSL_Dataset
from datasets.data_utils import get_data_loader
from train_utils import GM
from sklearn.metrics import *

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--load_path', type=str, default='./saved_models/prg/model_best.pth')
    parser.add_argument('--use_train_model', action='store_true')

    '''
    Backbone Net Configurations
    '''
    parser.add_argument('--net', type=str, default='wrn', 
            help='use {wrn,resnet18,preresnet,cnn13} for {Wide ResNet,ResNet-18,PreAct ResNet,CNN-13}')
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
                               {'depth': args.depth, 
                                'widen_factor': args.widen_factor,
                                'leaky_slope': args.leaky_slope,
                                'dropRate': args.dropout})
    
    eval_model = _net_builder(args.num_classes) 
    eval_model.load_state_dict(load_model,strict=True)
    if torch.cuda.is_available():
        eval_model.cuda()
    eval_model.eval()
    
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
    y_true = []
    y_pred = []
    with torch.no_grad():
        for x, y in eval_loader:
            y = y.long()
            x, y = x.cuda(), y.cuda()
            num_batch = x.shape[0]
            total_num += num_batch

            logits = eval_model(x) 
                
            max_probs, max_idx = torch.max(torch.softmax(logits, dim=-1), dim=-1)
            max_probs_sort, idx_sort = torch.sort(logits, descending=True)

            acc = torch.sum(max_idx == y)           
            total_acc += acc.detach()

            y_true.extend(y.cpu().tolist())
            y_pred.extend(max_idx.cpu().tolist())

    
    report = classification_report(y_true, y_pred, zero_division=1)
    precision = precision_score(y_true, y_pred, average='macro', zero_division=1)
    recall = recall_score(y_true, y_pred, average='macro')  
    gm = GM(y_pred, y_true)
    print(f"Test Accuracy: {total_acc/len(eval_dset)}, Precision: {precision}, Recall: {recall}, GM: {gm}\nReport: {report}")
