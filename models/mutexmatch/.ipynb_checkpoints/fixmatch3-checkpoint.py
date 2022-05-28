from net import *
from lib import *
from torch import optim
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.cuda.amp import autocast, GradScaler

import os
import contextlib
from train_utils import AverageMeter

from .fixmatch_utils import consistency_loss, consistency_loss3, Get_Scalar
from train_utils import ce_loss

class TotalNet(nn.Module):
    def __init__(self,net_builder, num_classes, widen_factor):
        super(TotalNet, self).__init__()
        self.feature_extractor = net_builder(num_classes=num_classes) 
        classifier_output_dim = num_classes
        # self.classifier = CLS(self.feature_extractor.output_num(), classifier_output_dim, bottle_neck_dim=32)
        # self.discriminator = AdversarialNetwork(128)
        self.discriminator_separate = AdversarialNetwork(64 * widen_factor)

    def forward(self, x):
        f = self.feature_extractor(x)
        f, _, __, y = self.classifier(f)
        d = self.discriminator(_)
        d_0 = self.discriminator_separate(_)
        return y, d, d_0

class FixMatch:
    def __init__(self, net_builder, num_classes, ema_m, T, p_cutoff, lambda_u,\
                 hard_label=True, t_fn=None, p_fn=None, it=0, num_eval_iter=1000, tb_log=None, logger=None, widen_factor=2):
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
        
        # self.train_model = net_builder(num_classes=num_classes) 
        self.train_model = TotalNet(net_builder, num_classes, widen_factor) 
        # self.eval_model = net_builder(num_classes=num_classes)
        self.eval_model = TotalNet(net_builder, num_classes, widen_factor) 
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
        
        # device = torch.device('cuda')
        # self.train_model.to(device)
        # feature_extractor = nn.DataParallel(self.train_model.feature_extractor, device_ids=[0], output_device=torch.device('cuda')).train(True)
        # discriminator_separate = nn.DataParallel(self.train_model.discriminator_separate, device_ids=[0], output_device=torch.device('cuda')).train(True)
        feature_extractor = self.train_model.feature_extractor.train(True)
        discriminator_separate = self.train_model.discriminator_separate.train(True)
        
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

        p_cutoff = 0.95
        for (x_lb, y_lb), (x_ulb_w, x_ulb_s, target , idx) in zip(self.loader_dict['train_lb'], self.loader_dict['train_ulb']):
            
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
                # _,fc1 = feature_extractor.forward(inputs, ood_test=True, sel=1)             
                # fc1, feature, logits, predict_prob  = classifier.forward(fc1)
                logits, feature = feature_extractor.forward(inputs, ood_test=True, sel=1) 
                # logits = self.train_model(inputs)
                

                feature_lb=feature[:num_lb]
                feature_ulb_w, feature_ulb_s = feature[num_lb:].chunk(2)
                fc2_lb=logits[:num_lb]
                fc2_ulb_w, fc2_ulb_s = logits[num_lb:].chunk(2)
                
                logits_x_lb = logits[:num_lb]
                logits_x_ulb_w, logits_x_ulb_s = logits[num_lb:].chunk(2)

                # _,fc1 = feature_extractor.forward(inputs, ood_test=True, sel=2)
                
                # fc1, feature, logits, predict_prob  = classifier.forward(fc1)
                # # logits = self.train_model(inputs)
                
                # feature_lb2=feature[:num_lb]
                # feature_ulb_w2, feature_ulb_s2 = feature[num_lb:].chunk(2)
                # fc2_lb2=logits[:num_lb]
                # fc2_ulb_w2, fc2_ulb_s2 = logits[num_lb:].chunk(2)
                
                # logits_x_lb2 = logits[:num_lb]
                # logits_x_ulb_w2, logits_x_ulb_s2 = logits[num_lb:].chunk(2)

                # _,fc1 = feature_extractor.forward(inputs, ood_test=True, sel=3)  
                # fc1, feature, logits, predict_prob  = classifier.forward(fc1)
                logits, feature = feature_extractor.forward(inputs, ood_test=True, sel=3) 
                feature_lb3=feature[:num_lb]
                feature_ulb_w3, feature_ulb_s3 = feature[num_lb:].chunk(2)
                fc2_lb3=logits[:num_lb]
                fc2_ulb_w3, fc2_ulb_s3 = logits[num_lb:].chunk(2)
                
                logits_x_lb3 = logits[:num_lb]
                logits_x_ulb_w3, logits_x_ulb_s3 = logits[num_lb:].chunk(2)

                # hyper-params for update
                T = self.t_fn(self.it)
                # p_cutoff = self.p_fn(self.it)

                prob_discriminator_lb_separate = discriminator_separate.forward(feature_lb3.detach())
                prob_discriminator_w_separate = discriminator_separate.forward(feature_ulb_w)
                # prob_discriminator_w_separate2 = discriminator_separate.forward(feature_ulb_w2.detach())
                prob_discriminator_w_separate3 = discriminator_separate.forward(feature_ulb_w3.detach())
                prob_discriminator_s_separate = discriminator_separate.forward(feature_ulb_s)

                adv_loss_separate = torch.zeros(1, 1).to(torch.device('cuda'))
                adv_loss_separate += nn.BCELoss()(prob_discriminator_lb_separate, torch.ones_like(prob_discriminator_lb_separate))
                adv_loss_separate += nn.BCELoss()(prob_discriminator_w_separate3, torch.zeros_like(prob_discriminator_w_separate3))
               
                del logits
                
                sup_loss = ce_loss(logits_x_lb, y_lb, reduction='mean')

                unsup_loss, dis_loss, mask = consistency_loss3(logits_x_ulb_w, 
                                              prob_discriminator_w_separate,  
                                              prob_discriminator_s_separate,
                                              logits_x_ulb_s,                                                                           
                                              'ce', T, p_cutoff,
                                               use_hard_labels=args.hard_label)

                total_loss = sup_loss + self.lambda_u * unsup_loss + adv_loss_separate + dis_loss
                
            
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
            tb_dict['train/dis_loss'] = dis_loss.detach()
            tb_dict['train/adv_loss_separate'] = adv_loss_separate.detach()
            tb_dict['train/total_loss'] = total_loss.detach() 
            tb_dict['train/mask_ratio'] = 1.0 - mask.detach() 
            tb_dict['lr'] = self.optimizer.param_groups[0]['lr']
            tb_dict['train/prefecth_time'] = start_batch.elapsed_time(end_batch)/1000.
            tb_dict['train/run_time'] = start_run.elapsed_time(end_run)/1000.
            
            
            if self.it % self.num_eval_iter == 0:
                
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
        feature_extractor = self.eval_model.feature_extractor
        # classifier=self.eval_model.classifier
        if eval_loader is None:
            eval_loader = self.loader_dict['eval']
        
        total_loss = 0.0
        total_acc = 0.0
        total_num = 0.0
        for x, y in eval_loader:
            
            x, y = x.cuda(args.gpu), y.cuda(args.gpu)
            num_batch = x.shape[0]
            total_num += num_batch
            
            # _,fc1 = feature_extractor.forward(x,ood_test=True)               
            # fc1, feature, logits, predict_prob  = classifier.forward(fc1)
            logits, feature = feature_extractor.forward(x, ood_test=True, sel=1) 
            # logits = eval_model(x)
            loss = F.cross_entropy(logits, y, reduction='mean')
            acc = torch.sum(torch.max(logits, dim=-1)[1] == y)
            
            total_loss += loss.detach()*num_batch
            total_acc += acc.detach()
        
        if not use_ema:
            eval_model.train()

        acc_p=0.0
        totalnum=0.0
        totalnum_p=0.0
        for  data in zip(ulb_loader):
            
            image , image_s , target = data[0][0] , data[0][1]  ,data[0][2]
            image = image.type(torch.FloatTensor).cuda()
            image_s = image_s.type(torch.FloatTensor).cuda()
            num_batch = image.shape[0]

            # discriminator = eval_model.discriminator
            discriminator_separate = eval_model.discriminator_separate
            # _, fc1 = feature_extractor.forward(image, ood_test=True)
            # fc1, feature, logits, predict_prob = classifier.forward(fc1)  
            logits, feature = feature_extractor.forward(image, ood_test=True, sel=1) 
            logit_dis_separate = discriminator_separate.forward(feature)    
            # logit_dis = discriminator.forward(feature)
            pseudo_label = torch.softmax(logits, dim=-1)
            max_probs, max_idx = torch.max(pseudo_label, dim=-1)
            mask = max_probs.ge(p_cutoff)    
            maskindex = np.where(mask.cpu()==1)[0]
            # print('w---------')
            # print(logit_dis_separate.mean())

            # _, fc1_s = feature_extractor.forward(image_s, ood_test=True)
            # fc1_s, feature_s, logit_s, predict_prob_s = classifier.forward(fc1_s)
            # logits_s, feature_s = feature_extractor.forward(image_s, ood_test=True, sel=1) 
            # pseudo_label_s = torch.softmax(logits_s, dim=-1)
            # max_probs_s, max_idx_s = torch.max(pseudo_label_s, dim=-1)
            # logit_dis_separate_s = discriminator_separate.forward(feature_s)
            # print('s---------')
            # print(logit_dis_separate_s.mean())
            # print('cha---------')
            # print((logit_dis_separate - logit_dis_separate_s).mean())

            mask = mask.float()
            
            maskindex_total = np.where(mask.cpu()==1)[0]
            if not len(maskindex_total)==0:
                acc_p += pseudo_label[maskindex_total].cpu().max(1)[1].eq(target[maskindex_total]).sum().cpu().numpy()
            totalnum += mask.numel()
            totalnum_p += len(maskindex_total)

            # n_dis = np.zeros((1,10),dtype=int)
            n_dis_separate = np.zeros((1,10),dtype=int)
            for i in range(10):
                # n_dis[0,i]+=torch.where((logit_dis[:,0]>(i/10)) & (logit_dis[:,0]<((i+1)/10)))[0].numel()
                n_dis_separate[0,i]+=torch.where((logit_dis_separate[:,0]>(i/10)) & (logit_dis_separate[:,0]<=((i+1)/10)))[0].numel()
            tmp_dis=0
            tmp_dis_sep=0
            for i in range(5):
                # tmp_dis +=  n_dis[0,i]
                tmp_dis_sep +=  n_dis_separate[0,i]
        if totalnum_p==0:
            pseudo_label_acc=0
        else:
            pseudo_label_acc=acc_p/totalnum_p
        
        return {'eval/loss': total_loss/total_num, 'eval/top-1-acc': total_acc/total_num , 
                # 'ulb/dis01':(n_dis/n_dis.sum())[0,0] ,
                # 'ulb/dis12':(n_dis/n_dis.sum())[0,1],
                # 'ulb/dis23':(n_dis/n_dis.sum())[0,2],
                # 'ulb/dis34':(n_dis/n_dis.sum())[0,3],
                # 'ulb/dis45':(n_dis/n_dis.sum())[0,4],
                # 'ulb/dis56':(n_dis/n_dis.sum())[0,5],
                # 'ulb/dis67':(n_dis/n_dis.sum())[0,6],
                # 'ulb/dis78':(n_dis/n_dis.sum())[0,7],
                # 'ulb/dis89':(n_dis/n_dis.sum())[0,8] , 
                # 'ulb/dis90':(n_dis/n_dis.sum())[0,9],
                'ulb/dis_sep01':(n_dis_separate/n_dis_separate.sum())[0,0],
                'ulb/dis_sep12':(n_dis_separate/n_dis_separate.sum())[0,1],
                'ulb/dis_sep23':(n_dis_separate/n_dis_separate.sum())[0,2],
                'ulb/dis_sep34':(n_dis_separate/n_dis_separate.sum())[0,3],
                'ulb/dis_sep45':(n_dis_separate/n_dis_separate.sum())[0,4],
                'ulb/dis_sep56':(n_dis_separate/n_dis_separate.sum())[0,5],
                'ulb/dis_sep67':(n_dis_separate/n_dis_separate.sum())[0,6],
                'ulb/dis_sep78':(n_dis_separate/n_dis_separate.sum())[0,7],
                'ulb/dis_sep89':(n_dis_separate/n_dis_separate.sum())[0,8],
                'ulb/dis_sep90':(n_dis_separate/n_dis_separate.sum())[0,9],
                # 'ulb/dis_acc':tmp_dis/n_dis.sum(),'ulb/dis_sep_acc':tmp_dis_sep/n_dis_separate.sum(),
                'ulb/dis_sep_acc':tmp_dis_sep/n_dis_separate.sum(),
                'ulb/total_num':len(lb_loader.dataset),
                'ulb/pseudo_label_acc':pseudo_label_acc,'ulb/pseudo_label_radio':totalnum_p/totalnum,
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
                    train_model.load_state_dict(checkpoint[key])
                elif 'eval_model' in key:
                    eval_model.load_state_dict(checkpoint[key])
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
