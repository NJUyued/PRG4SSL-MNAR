import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import contextlib
from sklearn.metrics import *
from torch.cuda.amp import autocast, GradScaler
from models.nets.resnet18 import *
from train_utils import AverageMeter, GM
from .prg_utils import consistency_loss_prg, Get_Scalar
from train_utils import ce_loss


class PRG:
    def __init__(self, net_builder, num_classes, ema_m, T, p_cutoff, lambda_u,\
                 hard_label=True, t_fn=None, p_fn=None, it=0, num_eval_iter=1000, tb_log=None, logger=None):
        
        super(PRG, self).__init__()

        # momentum update param
        self.loader = {}
        self.num_classes = num_classes
        self.ema_m = ema_m

        # create the encoders
        self.train_model = net_builder(num_classes=num_classes) 
        self.eval_model = net_builder(num_classes=num_classes)
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
        Train function of PRG.
        From data_loader, it inference training data, computes losses, and update the networks.
        """
        
        ngpus_per_node = torch.cuda.device_count()

        self.train_model.train()
        
        # for gpu profiling
        start_batch = torch.cuda.Event(enable_timing=True)
        end_batch = torch.cuda.Event(enable_timing=True)
        start_run = torch.cuda.Event(enable_timing=True)
        end_run = torch.cuda.Event(enable_timing=True)
        
        start_batch.record()
        best_eval_acc, best_it = 0.0, 0
        
        N = args.Nb
        alpha = args.alpha
        scaler = GradScaler()
        amp_cm = autocast if args.amp else contextlib.nullcontext
        label_bank = {}
        distri = torch.zeros((N,args.num_classes),dtype=torch.float32).cuda(args.gpu)
        label_count = 0
        class_transition_matrix = torch.zeros(args.num_classes,args.num_classes,N)
        for (x_lb, y_lb), data in zip(self.loader_dict['train_lb'], self.loader_dict['train_ulb']):
            
            y_lb = y_lb.long()
            if args.dataset == 'miniimage':
                x_ulb_w = data[0][0]
                x_ulb_s = data[0][1]
                idx = data[2]
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
            
            # inference and calculate losses
            with amp_cm():
                logits = self.train_model(inputs)
                
                logits_x_lb = logits[:num_lb]
                logits_x_ulb_w, logits_x_ulb_s = logits[num_lb:].chunk(2)

                pseudo_label = torch.softmax(logits_x_ulb_w, dim=-1)
                max_probs, max_idx = torch.max(pseudo_label, dim=-1)
                
                # class transition tracking (PRG)
                if not args.last:
                    for i in range(len(idx)):
                        if not idx[i].cpu().item() in label_bank.keys():
                            label_bank[idx[i].cpu().item()] = max_idx[i].cpu().item()
                        else:
                            if label_bank[idx[i].cpu().item()] != max_idx[i].cpu().item():
                                class_transition_matrix[label_bank[idx[i].cpu().item()],max_idx[i].cpu().item(),label_count] = class_transition_matrix[label_bank[idx[i].cpu().item()],max_idx[i].cpu().item(),label_count]+1
                                label_bank[idx[i].cpu().item()] = max_idx[i].cpu().item()               
                    distri[label_count] = pseudo_label.detach().mean(0)
                    label_count = (label_count + 1) % N
                    class_transition_matrix[:,:,label_count] = torch.zeros(args.num_classes,args.num_classes)
                
                # hyper-params for update
                T = self.t_fn(self.it)
                p_cutoff = self.p_fn(self.it)
               
                del logits
              
                sup_loss = ce_loss(logits_x_lb, y_lb, reduction='mean')
                
                # new work     
                CTT_tmp = torch.mean(class_transition_matrix, dim=2)  
                # diag = torch.diag(torch.ones(args.num_classes,args.num_classes)*(alpha/(args.num_classes**2-args.num_classes)))
                diag = torch.diag(torch.ones(args.num_classes,args.num_classes)*(alpha/(args.num_classes-1)))
                a_diag = torch.diag_embed(diag)
                ## matrix-wise norm
                # x_sum = torch.sum(CTT_tmp)
                # H = CTT_tmp / x_sum  
                # H = H + a_diag
                ## row-wise norm
                H = CTT_tmp / (CTT_tmp.abs().sum(1, keepdim=True) + 1e-16)
                H = H + a_diag 

                H_prime = H.cuda(args.gpu) / torch.mean(distri,dim=0)
            
                unsup_loss, mask = consistency_loss_prg(
                                              logits_x_ulb_w, 
                                              logits_x_ulb_s,  
                                              H_prime,  
                                              label_bank,
                                              idx, 
                                              args.last,                                                                  
                                              'ce', T, p_cutoff,
                                               use_hard_labels=args.hard_label)
    
                total_loss = sup_loss + self.lambda_u * unsup_loss
                
                # class transition tracking (PRG^last)
                if args.last:
                    pseudo_label = torch.softmax(logits_x_ulb_w, dim=-1)
                    max_probs, max_idx = torch.max(pseudo_label, dim=-1)

                    for i in range(len(idx)):
                        if not idx[i].cpu().item() in label_bank.keys():
                            label_bank[idx[i].cpu().item()] = max_idx[i].cpu().item()
                        else:
                            if label_bank[idx[i].cpu().item()] != max_idx[i].cpu().item():
                                class_transition_matrix[label_bank[idx[i].cpu().item()],max_idx[i].cpu().item(),label_count] = class_transition_matrix[label_bank[idx[i].cpu().item()],max_idx[i].cpu().item(),label_count]+1
                                label_bank[idx[i].cpu().item()] = max_idx[i].cpu().item()     
                    distri[label_count] = pseudo_label.detach().mean(0)        
                    label_count = (label_count + 1) % N
                    class_transition_matrix[:,:,label_count] = torch.zeros(args.num_classes,args.num_classes)
                    
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
    def evaluate(self, eval_loader=None, args=None , lb_loader=None, ulb_loader=None, p_cutoff=None):
        use_ema = hasattr(self, 'eval_model')
        
        eval_model = self.eval_model if use_ema else self.train_model
        eval_model.eval()
      
        if eval_loader is None:
            eval_loader = self.loader_dict['eval']
        
        total_loss = 0.0
        total_acc = 0.0
        total_num = 0.0
        y_true = []
        y_pred = []
        y_logits = []

        for x, y in eval_loader:
            y = y.long()
            x, y = x.cuda(args.gpu), y.cuda(args.gpu)
            num_batch = x.shape[0]
            total_num += num_batch
            logits = self.eval_model(x)            
            max_probs, max_idx = torch.max(torch.softmax(logits, dim=-1), dim=-1)
            y_true.extend(y.cpu().tolist())
            y_pred.extend(torch.max(logits, dim=-1)[1].cpu().tolist())
            y_logits.extend(torch.softmax(logits, dim=-1).cpu().tolist())
            loss = F.cross_entropy(logits, y, reduction='mean')
            acc = torch.sum(max_idx == y)  
            total_loss += loss.detach()*num_batch
            total_acc += acc.detach()      
        # report = classification_report(y_true, y_pred, zero_division=1)
        precision = precision_score(y_true, y_pred, average='macro', zero_division=1)
        recall = recall_score(y_true, y_pred, average='macro')  

        gm = GM(y_pred, y_true)
        # self.print_fn(report)
        
        if not use_ema:
            eval_model.train()       

        acc_class = torch.ones((1,args.num_classes),dtype=torch.float32)
        acc_p=0.0

        totalnum=0.0
        totalnum_p=0.0
        for data in zip(ulb_loader):
            
            if args.dataset == 'miniimage':
                image, target = data[0][0], data[0][1]
                image = image[0]
            else:
                image, image_s, target = data[0][0], data[0][1], data[0][2]

            image = image.type(torch.FloatTensor).cuda()
            num_batch = image.shape[0]
 
            logits = self.eval_model(image)      
            pseudo_label = torch.softmax(logits, dim=-1)
            max_probs, max_idx = torch.max(pseudo_label, dim=-1)
            mask = max_probs.ge(p_cutoff)    
            maskindex = np.where(mask.cpu()==1)[0]

            mask = mask.float()            
            maskindex_total = np.where(mask.cpu()==1)[0]
            
            if not len(maskindex_total)==0:
                acc_p += pseudo_label[maskindex_total].cpu().max(1)[1].eq(target[maskindex_total]).sum().cpu().numpy()              
            totalnum += mask.numel()
            totalnum_p += len(maskindex_total)

            # class-wise acc
            # for i in range(len(pseudo_label)):  
            #     if pseudo_label[i].cpu().max(0)[1]==target[i]:
            #         acc_class[0,target[i]] += 1
       
        # self.print_fn('class-wise acc:')
        # self.print_fn(acc_class/(totalnum/args.num_classes))
        
        if totalnum_p==0:
            pseudo_label_acc = 0
        else:
            pseudo_label_acc = acc_p/totalnum_p

        return {'eval/loss': total_loss/total_num, 'eval/top-1-acc': total_acc/total_num, 
                'eval/precision': precision, 'eval/recall': recall, 'eval/GM': gm,     
                'ulb/pseudo_label_acc':pseudo_label_acc,'ulb/pseudo_label_ratio':totalnum_p/totalnum,
                'p_cutoff':p_cutoff,'ulb/total_num':len(lb_loader.dataset),}
    
    
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
