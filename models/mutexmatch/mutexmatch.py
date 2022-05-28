from models.nets.net import *
from torch import optim
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.cuda.amp import autocast, GradScaler
from random import randint
from types import MethodType,FunctionType
import numpy as np
import os
import contextlib
from train_utils import AverageMeter

from .fixmatch_utils import consistency_loss_dis_3, Get_Scalar, normalize_d, normalize_class
from train_utils import ce_loss

class TotalNet(nn.Module):
    def __init__(self, net_builder, num_classes, widen_factor, net_name):
        super(TotalNet, self).__init__()
        if net_name=='resnet18':
            base_net = net_builder(num_classes=num_classes)    
            self.feature_extractor = ResNet18(num_classes, base_net)  
        elif net_name=='cnn13':
            self.feature_extractor = cnn13(num_classes=num_classes)  
        else:
            self.feature_extractor = net_builder(num_classes=num_classes)                  
        classifier_output_dim = num_classes
        self.classifier_reverse = ReverseCLS(self.feature_extractor.output_num(), classifier_output_dim)
        # self.discriminator_separate = AdversarialNetwork(64 * widen_factor)
        
    def forward(self, x):
        f = self.feature_extractor(x)
        f, _, __, y = self.classifier_reverse(f)
        return y

class FixMatch:
    def __init__(self, net_builder, num_classes, ema_m, T, p_cutoff, lambda_u,\
                 hard_label=True, t_fn=None, p_fn=None, it=0, num_eval_iter=1000, tb_log=None, logger=None, widen_factor=2, net_name=''):
        """
        class Fixmatch contains setter of data_loader, optimizer, and model update methods.
        Args:
            net_builder: backbone network class (see net_builder in utils.py)
            num_classes: # of label classes 
            ema_m: momentum of exponential moving average for eval_model
            T: Temperature scaling parameter for output sharpening (only when hard_label = False)
            p_cutoff: confidence cutoff parameters for loss masking
            lambda_u: ratio of unsupervised loss to supervised loss
            hard_label: If True, consistency regularization use a hard pseudo label.
            it: initial iteration count
            num_eval_iter: freqeuncy of iteration (after 500,000 iters)
            tb_log: tensorboard writer (see train_utils.py)
            logger: logger (see utils.py)
        """
        
        super(FixMatch, self).__init__()

        # momentum update param
        self.loader = {}
        self.num_classes = num_classes
        self.ema_m = ema_m

        # create the encoders
        # network is builded only by num_classes,
        # other configs are covered in main.py
        
        self.train_model = TotalNet(net_builder, num_classes, widen_factor, net_name) 
        self.eval_model = TotalNet(net_builder, num_classes, widen_factor, net_name) 
        self.num_eval_iter = num_eval_iter
        self.t_fn = Get_Scalar(T) #temperature params function
        self.p_fn = Get_Scalar(p_cutoff) #confidence cutoff function
        self.lambda_u = lambda_u
        self.tb_log = tb_log
        self.use_hard_label = hard_label
        
        self.optimizer = None
        self.scheduler = None
        
        self.it = 0
        
        self.logger = logger
        self.print_fn = print if logger is None else logger.info
        
        for param_q, param_k in zip(self.train_model.parameters(), self.eval_model.parameters()):
            param_k.data.copy_(param_q.detach().data)  # initialize
            param_k.requires_grad = False  # not update by gradient for eval_net
            
        self.eval_model.eval()
            
            
    @torch.no_grad()
    def _eval_model_update(self):
        """
        Momentum update of evaluation model (exponential moving average)
        """
        for param_train, param_eval in zip(self.train_model.parameters(), self.eval_model.parameters()):
            param_eval.copy_(param_eval * self.ema_m + param_train.detach() * (1-self.ema_m))
        
        for buffer_train, buffer_eval in zip(self.train_model.buffers(), self.eval_model.buffers()):
            buffer_eval.copy_(buffer_train)            
    
     
    def set_data_loader(self, loader_dict):
        self.loader_dict = loader_dict
        self.print_fn(f'[!] data loader keys: {self.loader_dict.keys()}')    
            
    
    def set_optimizer(self, optimizer, scheduler=None):
        self.optimizer = optimizer
        self.scheduler = scheduler
    
    
    def train(self, args, logger=None):
        """
        Train function of FixMatch.
        From data_loader, it inference training data, computes losses, and update the networks.
        """
        
        device = torch.device('cuda')
        # self.train_model.to(device)
        # feature_extractor = nn.DataParallel(self.train_model.feature_extractor, device_ids=[0], output_device=torch.device('cuda')).train(True)
        feature_extractor = self.train_model.module.feature_extractor.train(True) if hasattr(self.train_model, 'module') else self.train_model.feature_extractor.train(True)
        cls_reverse = self.train_model.module.classifier_reverse.train(True) if hasattr(self.train_model, 'module') else self.train_model.classifier_reverse.train(True)
        feature_extractor.cuda(args.gpu)
        cls_reverse.cuda(args.gpu)
        ngpus_per_node = torch.cuda.device_count()

        #lb: labeled, ulb: unlabeled
        self.train_model.train()
        
        # for gpu profiling
        start_batch = torch.cuda.Event(enable_timing=True)
        end_batch = torch.cuda.Event(enable_timing=True)
        start_run = torch.cuda.Event(enable_timing=True)
        end_run = torch.cuda.Event(enable_timing=True)
        
        start_batch.record()
        best_eval_acc, best_it = 0.0, 0
        
        scaler = GradScaler()
        amp_cm = autocast if args.amp else contextlib.nullcontext
        x_bank = None
        y_bank = None
        sup_flag = True
        label_bank = {}
        distri = torch.ones((128,args.num_classes),dtype=torch.float32).cuda(args.gpu)
        distri_high = torch.ones((128,args.num_classes),dtype=torch.float32).cuda(args.gpu)
        distri_low = torch.ones((128,args.num_classes),dtype=torch.float32).cuda(args.gpu)
        label_count = 0
        label_matrix = torch.ones(args.num_classes,args.num_classes,128)
        # diag = torch.diag(label_matrix)
        # a_diag = torch.diag_embed(diag)
        # label_matrix = label_matrix - a_diag
        label_matrix_epoch = torch.ones(args.num_classes,args.num_classes)
        diag = torch.diag(label_matrix_epoch)
        a_diag = torch.diag_embed(diag)
        label_matrix_epoch = label_matrix_epoch - a_diag
        for (x_lb, y_lb), data in zip(self.loader_dict['train_lb'], self.loader_dict['train_ulb']):
            
            y_lb = y_lb.long()
            if args.dataset == 'miniimage':
                x_ulb_w = data[0][0]
                x_ulb_s = data[0][1]
            else:
                x_ulb_w = data[0]
                x_ulb_s = data[1]
                idx = data[3]

            # prevent the training iterations exceed args.num_train_iter
            if self.it > args.num_train_iter:
                break
            
            end_batch.record()
            torch.cuda.synchronize()
            start_run.record()
            
            num_lb = x_lb.shape[0]
            num_ulb = x_ulb_w.shape[0]
            assert num_ulb == x_ulb_s.shape[0]
            
            x_lb, x_ulb_w, x_ulb_s = x_lb.cuda(args.gpu), x_ulb_w.cuda(args.gpu), x_ulb_s.cuda(args.gpu)
            y_lb = y_lb.cuda(args.gpu)
            
            inputs = torch.cat((x_lb, x_ulb_w, x_ulb_s))
            
            # inference and calculate sup/unsup losses
            with amp_cm():
                logits, feature = feature_extractor.forward(inputs, ood_test=True, sel=1) 
                feature_lb = feature[:num_lb]
                feature_ulb_w, feature_ulb_s = feature[num_lb:].chunk(2)
                fc2_lb=logits[:num_lb]
                fc2_ulb_w, fc2_ulb_s = logits[num_lb:].chunk(2)
                
                logits_x_lb = logits[:num_lb]
                logits_x_ulb_w, logits_x_ulb_s = logits[num_lb:].chunk(2)

                pseudo_label = torch.softmax(logits_x_ulb_w, dim=-1)
                max_probs, max_idx = torch.max(pseudo_label, dim=-1)
                mask = max_probs.ge(0.95).float()
                mask_idx = np.where(mask.cpu()==1)[0]
                mask_low = max_probs.lt(0.95).float()
                mask_idx_low = np.where(mask_low.cpu()==1)[0]

                ##
                feature, logits_reverse, predict_prob = cls_reverse.forward(feature)
                logits_x_lb_reverse = logits_reverse[:num_lb]
                logits_x_ulb_w_reverse, logits_x_ulb_s_reverse = logits_reverse[num_lb:].chunk(2)

                feature_separate, logits_reverse_separate, predict_prob_separate = cls_reverse.forward(feature.detach())

                pseudo_label_reverse = torch.softmax(logits.detach(), dim=-1)
                # max_probs, max_idx= torch.max(pseudo_label_reverse, dim=-1)          
                min_probs_reverse, min_idx_reverse = torch.min(pseudo_label_reverse, dim=-1)

                ## nda
                # if self.it > 1000:
                #     matrix_sum = torch.sum(label_matrix)
                #     matrix_prob = []
                #     for i in range(10):
                #         temp = torch.sum(label_matrix, dim=1)[i]/matrix_sum
                #         matrix_prob += [i for j in range(round(temp.item()*10))]

                #     index = np.random.choice(matrix_prob, 1)
                #     select_idx = np.where(max_idx[mask_idx].cpu().numpy() == index)[0]
                #     if len(select_idx) != 0:
                #         logits_x_lb[label_count] = logits[np.random.choice(select_idx, 1)]
                #         y_lb[label_count] = torch.from_numpy(index)
                #     label_count += 1
                #     label_count = label_count % num_lb
                ##
                # logits_x_lb_c = logits_x_lb.clone()
                # y_lb_c = y_lb.clone()
                # if self.it >= 1:              
                #     matrix_sum = torch.sum(label_matrix)
                #     matrix_prob = []
                #     for i in range(10):
                #         temp = (torch.sum(label_matrix, dim=1)[i]+torch.sum(label_matrix, dim=0)[i])/matrix_sum
                #         matrix_prob += [i for j in range(round(temp.item()*100))]
                #     for i in range(num_lb):
                #         index = np.random.choice(matrix_prob, 1)
                #         # select_idx = np.where(max_idx[mask_idx].cpu().numpy() == index)[0] # from unlabeled data
                #         select_idx = np.where(y_lb.cpu().numpy() == index)[0]
                #         if len(select_idx) != 0:
                #             logits_x_lb_c[i] = logits_x_lb[np.random.choice(select_idx, 1).tolist()[0]]
                #             y_lb_c[i] = torch.from_numpy(index)

                for i in range(len(idx)):
                    if not idx[i].cpu().item() in label_bank.keys():
                        label_bank[idx[i].cpu().item()] = max_idx[i].cpu().item()
                    else:
                        if label_bank[idx[i].cpu().item()] != max_idx[i].cpu().item():
                            label_matrix[label_bank[idx[i].cpu().item()],max_idx[i].cpu().item(),label_count] = label_matrix[label_bank[idx[i].cpu().item()],max_idx[i].cpu().item(),label_count]+1
                            label_matrix_epoch[label_bank[idx[i].cpu().item()],max_idx[i].cpu().item()] = label_matrix_epoch[label_bank[idx[i].cpu().item()],max_idx[i].cpu().item()]+1
                            
                            label_bank[idx[i].cpu().item()] = max_idx[i].cpu().item()    
                        # else:
                        #     label_matrix[label_bank[idx[i].cpu().item()],max_idx[i].cpu().item(),label_count] = label_matrix[label_bank[idx[i].cpu().item()],max_idx[i].cpu().item(),label_count]+1
                        #     label_matrix_epoch[label_bank[idx[i].cpu().item()],max_idx[i].cpu().item()] = label_matrix_epoch[label_bank[idx[i].cpu().item()],max_idx[i].cpu().item()]+1           
                distri[label_count] = pseudo_label.detach().mean(0)
                distri_high[label_count] = pseudo_label[mask_idx].detach().mean(0)
                distri_low[label_count] = pseudo_label[mask_idx_low].detach().mean(0)
                label_count = (label_count + 1) % 128
                ##

                # res = [0] * len(max_idx_reverse)
                # all_possible_labels = set(range(args.num_classes))
                # for each in range(len(res)):
                #     potential_anws = all_possible_labels - set([max_idx_reverse[each].item()])
                #     res[each] = np.random.choice(list(potential_anws))
                # res_torch = torch.tensor(res, dtype=max_idx_reverse.dtype)
                # res_torch = res_torch.cuda()
                
                # hyper-params for update
                T = self.t_fn(self.it)
                # p_cutoff = self.p_fn(self.it)
                p_cutoff = 0.95
                # lambda_d = self.p_fn(self.it)
               
                del logits
              
                # sup_loss = ce_loss(logits_x_lb, y_lb, reduction='mean') if self.it <= 1000 else 0.75 * ce_loss(logits_x_lb, y_lb, reduction='mean') + 0.25 * ce_loss(logits_x_ulb_w[np.array(x_bank)], torch.from_numpy(np.array(y_bank)).cuda(args.gpu).long(), reduction='mean')
                # sup_loss = ce_loss(logits_x_lb, y_lb, reduction='mean') if self.it <= 20000 else ce_loss(logits_x_lb, y_lb, reduction='mean') + ce_loss(logits_x_lb_c, y_lb_c, reduction='mean')
                sup_loss = ce_loss(logits_x_lb, y_lb, reduction='mean')
                reverse_loss = ce_loss(logits_reverse_separate, min_idx_reverse, reduction='mean')
                # reverse_loss = ce_loss(logits_reverse_separate, res_torch, reduction='mean')
                # reverse_loss = ce_loss(logits_reverse_separate, pseudo_label_reverse, use_hard_labels=False, reduction='none').mean() 
                
                # new work     
                label_matrix_tmp = torch.mean(label_matrix, dim=2)  
                diag = torch.diag(torch.ones(10,10)*0.0111)
                a_diag = torch.diag_embed(diag)
                x_sum = torch.sum(label_matrix_tmp)
                x = label_matrix_tmp / x_sum  
                label_matrix_tmp = x + a_diag
                x_sum = torch.sum(label_matrix_epoch)
                x = label_matrix_epoch / x_sum  
                label_matrix_tmp_ecpch = x + a_diag
                # label_matrix_tmp = torch.ones(10,10)*0.0111 / label_matrix_tmp           
                # label_matrix_tmp = label_matrix_tmp.mean(1).repeat(10,1) / label_matrix_tmp_ecpch 
                # label_matrix_tmp = torch.flip(label_matrix_tmp.cuda(args.gpu),dims=[1]) / torch.mean(distri,dim=0)
                # label_matrix_high = normalize_class(torch.ones_like(label_matrix_tmp) - label_matrix_tmp).cuda(args.gpu) / torch.mean(distri,dim=0)
                label_matrix_high = label_matrix_tmp.cuda(args.gpu) / torch.mean(distri,dim=0)
                label_matrix_low = torch.mean(distri_high,dim=0) / torch.mean(distri_low,dim=0)
                # label_matrix_tmp = label_matrix_tmp.cuda(args.gpu) / torch.mean(distri,dim=0)
                unsup_loss, masked_reverse_loss, mask = consistency_loss_dis_3(
                # unsup_loss, mask = consistency_loss_dis_3(
                                              logits_x_ulb_w_reverse,
                                              logits_x_ulb_s_reverse,
                                              logits_x_ulb_w, 
                                              logits_x_ulb_s,  
                                              label_matrix_high,  
                                              label_matrix_low,                                                                     
                                              'ce', T, p_cutoff,
                                               use_hard_labels=args.hard_label)
                
                # total_loss = sup_loss + self.lambda_u * unsup_loss + reverse_loss +  masked_reverse_loss      
                total_loss = sup_loss + self.lambda_u * unsup_loss
                    
            # parameter updates
            if args.amp:
                scaler.scale(total_loss).backward()
                scaler.step(self.optimizer)
                scaler.update()
            else:
                total_loss.backward() 
                self.optimizer.step()
                
            self.scheduler.step()
            self.train_model.zero_grad()
            
            
            with torch.no_grad():
                self._eval_model_update()
            
            end_run.record()
            torch.cuda.synchronize()
            
            #tensorboard_dict update
            tb_dict = {}
            tb_dict['train/sup_loss'] = sup_loss.detach() 
            tb_dict['train/unsup_loss'] = unsup_loss.detach() 
            tb_dict['train/reverse_loss'] = reverse_loss.detach()
            tb_dict['train/masked_reverse_loss'] = masked_reverse_loss.detach()
            tb_dict['train/total_loss'] = total_loss.detach() 
            tb_dict['train/mask_ratio'] = 1.0 - mask.detach() 
            tb_dict['lr'] = self.optimizer.param_groups[0]['lr']
            tb_dict['train/prefecth_time'] = start_batch.elapsed_time(end_batch)/1000.
            tb_dict['train/run_time'] = start_run.elapsed_time(end_run)/1000.
            
            
            if self.it % self.num_eval_iter == 0:
                # nda
                # matrix_sum = torch.sum(label_matrix)
                # matrix_prob = []
                # for i in range(10):
                #     temp = (torch.sum(label_matrix, dim=1)[i]+torch.sum(label_matrix, dim=0)[i])/matrix_sum
                #     matrix_prob += [i for j in range(round(temp.item()*10))]
                # x_bank = torch.ones_like(y_lb)
                # y_bank = torch.ones_like(y_lb)
                # x_bank = []
                # y_bank = []
                # for i in range(num_lb):
                #     index = np.random.choice(matrix_prob, 1)
                #     # select_idx = np.where(max_idx[mask_idx].cpu().numpy() == index)[0] # from unlabeled data
                #     select_idx = np.where(y_lb.cpu().numpy() == index)[0]
                #     if len(select_idx) != 0:
                #         # x_bank[i] = torch.from_numpy(np.random.choice(select_idx, 1))
                #         # y_bank[i] = torch.from_numpy(index)
                #         x_bank.append(np.random.choice(select_idx, 1).tolist()[0])
                #         y_bank.append(index.tolist()[0])

                # nda
                x_sum = torch.sum(torch.mean(label_matrix,dim=2))
                x = torch.mean(label_matrix,dim=2) / x_sum   
                self.print_fn(x)
                self.print_fn('='*100)
                x_sum = torch.sum(label_matrix_epoch)
                x = label_matrix_epoch / x_sum   
                self.print_fn(x)
                self.print_fn('='*100)
                self.print_fn(label_matrix_epoch)
                label_matrix_epoch = torch.ones(args.num_classes,args.num_classes)
                diag = torch.diag(label_matrix_epoch)
                a_diag = torch.diag_embed(diag)
                label_matrix_epoch = label_matrix_epoch - a_diag

                eval_dict = self.evaluate(args=args,lb_loader=self.loader_dict['train_lb'],ulb_loader=self.loader_dict['eval_ulb'],p_cutoff=p_cutoff)
                tb_dict.update(eval_dict)
                
                save_path = os.path.join(args.save_dir, args.save_name)
                
                if tb_dict['eval/top-1-acc'] > best_eval_acc:
                    best_eval_acc = tb_dict['eval/top-1-acc']
                    best_it = self.it
                
                self.print_fn(f"{self.it} iteration, USE_EMA: {hasattr(self, 'eval_model')}, {tb_dict}, BEST_EVAL_ACC: {best_eval_acc}, at {best_it} iters")
            
            if not args.multiprocessing_distributed or \
                    (args.multiprocessing_distributed and args.rank % ngpus_per_node == 0):
                
                if self.it == best_it:
                    self.save_model('model_best.pth', save_path)
                
                if not self.tb_log is None:
                    self.tb_log.update(tb_dict, self.it)

            self.it +=1
            del tb_dict
            start_batch.record()
            if self.it > 2**19:
                self.num_eval_iter = 1000
        
        eval_dict = self.evaluate(args=args,lb_loader=self.loader_dict['train_lb'],ulb_loader=self.loader_dict['eval_ulb'],p_cutoff=p_cutoff)
        eval_dict.update({'eval/best_acc': best_eval_acc, 'eval/best_it': best_it})
        return eval_dict
            
            
    @torch.no_grad()
    def evaluate(self, eval_loader=None, args=None , lb_loader=None ,ulb_loader=None,p_cutoff=None):
        use_ema = hasattr(self, 'eval_model')
        
        eval_model = self.eval_model if use_ema else self.train_model
        eval_model.eval()
        feature_extractor = self.eval_model.module.feature_extractor if hasattr(self.eval_model, 'module') else self.eval_model.feature_extractor
        cls_reverse = self.eval_model.module.classifier_reverse if hasattr(self.eval_model, 'module') else self.eval_model.classifier_reverse
        if eval_loader is None:
            eval_loader = self.loader_dict['eval']
        
        total_loss = 0.0
        total_acc = 0.0
        total_acc_fix = 0.0
        total_acc_fix2 = torch.zeros((1,9))
        total_num = 0.0
        
        for x, y in eval_loader:
            y = y.long()
            x, y = x.cuda(args.gpu), y.cuda(args.gpu)
            # x, y = x.cuda(), y.cuda()
            num_batch = x.shape[0]
            total_num += num_batch

            logits, feature = feature_extractor.forward(x, ood_test=True, sel=1) 
            
            max_probs, max_idx = torch.max(torch.softmax(logits, dim=-1), dim=-1)
            max_probs_sort, idx_sort = torch.sort(logits, descending=True)
           
            feature, logits_reverse, predict_prob = cls_reverse.forward(feature)
            pseudo_label_reverse = torch.softmax(logits_reverse, dim=-1)
            max_probs_reverse, max_idx_reverse = torch.max(pseudo_label_reverse, dim=-1)
            min_probs_reverse, min_idx_reverse = torch.min(pseudo_label_reverse, dim=-1)
            max_probs_sort_reverse, idx_sort_reverse = torch.sort(pseudo_label_reverse, descending=True)

            max_idx_fix = max_idx.clone()
            for i in range(max_idx.size(0)):
                if max_idx[i] in idx_sort_reverse[i,0:8]:
                    max_idx_fix[i] = idx_sort[i,1]

            for i in range(1,10):
                conf = (0.95 - i * 0.1) * torch.ones_like(max_probs)
                max_idx_fix2 = torch.where(max_probs>conf, max_idx, idx_sort_reverse[:,9])
                acc_fix2 = torch.sum(max_idx_fix2 == y)
                total_acc_fix2[0,i-1] += acc_fix2.detach().cpu()
            loss = F.cross_entropy(logits, y, reduction='mean')
            acc = torch.sum(max_idx == y)
            acc_fix = torch.sum(min_idx_reverse == y)         
            total_loss += loss.detach()*num_batch
            total_acc += acc.detach()
            total_acc_fix += acc_fix.detach()         
        
        if not use_ema:
            eval_model.train()       

        distri = torch.ones((10,args.num_classes),dtype=torch.float32).cuda(args.gpu)
        c = 0
        acc_sep = torch.ones((1,args.num_classes),dtype=torch.float32)
        acc_p=0.0
        acc_p_r=0.0
        totalnum=0.0
        totalnum_p=0.0
        tpc = np.zeros((10,10),dtype=int)
        tnc = np.zeros((10,10),dtype=int)
        distur = 0
        for  data in zip(ulb_loader):
            
            if args.dataset == 'miniimage':
                image, target = data[0][0], data[0][1]
                image = image[0]
            else:
                image, image_s, target = data[0][0], data[0][1], data[0][2]
            # image , image_s , target = data[0][0] , data[0][1]  ,data[0][2]
            image = image.type(torch.FloatTensor).cuda()
            num_batch = image.shape[0]
 
            logits, feature = feature_extractor.forward(image, ood_test=True, sel=1)    
            pseudo_label = torch.softmax(logits, dim=-1)
            max_probs, max_idx = torch.max(pseudo_label, dim=-1)
            mask = max_probs.ge(p_cutoff)    
            maskindex = np.where(mask.cpu()==1)[0]

            # 统计单独训练准确率
            # max_probs_reverse_sort, idx_sort = torch.sort(pseudo_label,descending=True)
            # for i in range(100):
            #     acc_sep[0,i] +=  torch.sum(idx_sort[:,i].cpu() == target).detach() 

            # for i in range(10):
            #     # disidx = torch.where((target.cuda(args.gpu)==i) & (max_idx!=i))   
            #     disidx = torch.where((target.cuda(args.gpu)==i) )       
            #     if len(pseudo_label[disidx].data)!=0:
            #         distri[i,:] += pseudo_label[disidx].detach().mean(0)    
            # c += 1    

            mask = mask.float()            
            maskindex_total = np.where(mask.cpu()==1)[0]

            feature, logits_reverse, predict_prob = cls_reverse.forward(feature)
            pseudo_label_reverse = torch.softmax(logits_reverse, dim=-1)
            max_probs_reverse, max_idx_reverse = torch.max(pseudo_label_reverse, dim=-1)
            max_probs_reverse_sort, idx_sort = torch.sort(max_probs_reverse,descending=True)
            
            acc_p_r += pseudo_label_reverse.cpu().max(1)[1].eq(target).sum().cpu().numpy()

            if type(distur)==int:
                distur = torch.sum(pseudo_label_reverse, dim=0)
            else:
                distur = distur + torch.sum(pseudo_label_reverse, dim=0)

            if not len(maskindex_total)==0:
                acc_p += pseudo_label[maskindex_total].cpu().max(1)[1].eq(target[maskindex_total]).sum().cpu().numpy()              
            totalnum += mask.numel()
            totalnum_p += len(maskindex_total)
            # for i in range(len(pseudo_label)):              
            #     tpc[target[i],max_idx[i]] = tpc[target[i],max_idx[i]] + 1
            #     tnc[target[i],max_idx_reverse[i]] = tnc[target[i],max_idx_reverse[i]] + 1
        # self.print_fn(acc_sep/totalnum)
        # self.print_fn(distri/c)
        # np.set_printoptions(suppress=True)

        # for i in range(10):
        #     print('[',end='')
        #     for j in range(10):
        #         print(float(tpc[i,j])/50,end='')
        #         if j!= 9:
        #             print(',',end='') 
        #     print('],')
        # print('---------------')   

        # for i in range(10):
        #     print('[',end='')
        #     for j in range(10):
        #         print(float(tnc[i,j])/50,end='')
        #         if j!= 9:
        #             print(',',end='') 
        #     print('],')
        if totalnum_p==0:
            pseudo_label_acc=0
        else:
            pseudo_label_acc=acc_p/totalnum_p

        return {'eval/loss': total_loss/total_num, 'eval/top-1-acc': total_acc/total_num , 
                'eval/top-1-acc(TNC)': total_acc_fix/total_num ,
                'ulb/total_num':len(lb_loader.dataset),
                'ulb/pseudo_label_acc':pseudo_label_acc,'ulb/pseudo_label_radio':totalnum_p/totalnum,
                'ulb/pseudo_label_reverse_acc':1-(acc_p_r/totalnum),
                'p_cutoff':p_cutoff}
    
    
    def save_model(self, save_name, save_path):
        save_filename = os.path.join(save_path, save_name)
        train_model = self.train_model.module if hasattr(self.train_model, 'module') else self.train_model
        eval_model = self.eval_model.module if hasattr(self.eval_model, 'module') else self.eval_model
        torch.save({'train_model': train_model.state_dict(),
                    'eval_model': eval_model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'scheduler': self.scheduler.state_dict(),
                    'it': self.it}, save_filename)
        
        self.print_fn(f"model saved: {save_filename}")
    
    
    def load_model(self, load_path):
        checkpoint = torch.load(load_path,map_location=torch.device('cpu'))
        
        train_model = self.train_model.module if hasattr(self.train_model, 'module') else self.train_model
        eval_model = self.eval_model.module if hasattr(self.eval_model, 'module') else self.eval_model
        
        for key in checkpoint.keys():
            if hasattr(self, key) and getattr(self, key) is not None:
                if 'train_model' in key:
                    train_model.load_state_dict(checkpoint[key], strict=False)
                elif 'eval_model' in key:
                    eval_model.load_state_dict(checkpoint[key], strict=False)
                elif key == 'it':
                    self.it = checkpoint[key]
                elif key == 'scheduler':
                    self.scheduler.load_state_dict(checkpoint[key])
                elif key == 'optimizer':
                    self.optimizer.load_state_dict(checkpoint[key]) 
                else:
                    getattr(self, key).load_state_dict(checkpoint[key])
                self.print_fn(f"Check Point Loading: {key} is LOADED")
            else:
                self.print_fn(f"Check Point Loading: {key} is **NOT** LOADED")

if __name__ == "__main__":
    pass
