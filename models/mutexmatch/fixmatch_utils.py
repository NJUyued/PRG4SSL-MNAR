# # from lib import *
# import math
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import numpy as np
# from train_utils import ce_loss

# def normalize_d(x):
#     x_sum = torch.sum(x)
#     x = x / x_sum
#     return x.detach()

# def normalize_class(x):
#     x_sum = torch.sum(x,dim=1)
#     x = x / x_sum
#     return x.detach()


# class Get_Scalar:
#     def __init__(self, value):
#         self.value = value
        
#     def get_value(self, iter):
#         return self.value
    
#     def __call__(self, iter):
#         return self.value


# def consistency_loss_dis_3(logits_x_ulb_w_reverse, logits_x_ulb_s_reverse, logits_w, logits_s, label_matrix_high, label_matrix_low, name='ce', T=1.0, p_cutoff=0.0, use_hard_labels=True):
#     # for fixmatch4
#     assert name in ['ce', 'L2']
#     logits_w = logits_w.detach()
#     logits_x_ulb_w_reverse = logits_x_ulb_w_reverse.detach()

#     if name == 'L2':
#         assert logits_w.size() == logits_s.size()
#         return F.mse_loss(logits_s, logits_w, reduction='mean')
    
#     elif name == 'L2_mask':
#         pass

#     elif name == 'ce':
#         pseudo_label = torch.softmax(logits_w, dim=-1)
#         max_probs, max_idx = torch.max(pseudo_label, dim=-1)
#         # mask = max_probs.ge(p_cutoff).float()

#         ## 
#         # mask = max_probs.ge(p_cutoff).float()
#         # mask_low = max_probs.lt(p_cutoff).float()
#         # maskindex_high = np.where(mask.cpu()==1)[0]
#         # maskindex_low = np.where(mask_low.cpu()==1)[0]

#         # pseudo_label_high = pseudo_label[maskindex_high]
#         # max_idx_high = max_idx[maskindex_high]
        
#         # pseudo_label_low = pseudo_label[maskindex_low]
#         # max_idx_low = max_idx[maskindex_low]
#         # for i in range(pseudo_label_high.size(0)):
#         #     pseudo_label_high[i,:] = normalize_d(pseudo_label_high[i,:] * label_matrix_high[max_idx_high[i],:])
#         # pseudo_label_low = normalize_d(pseudo_label_low * label_matrix_low)
#         # # for i in range(pseudo_label_low.size(0)):
#         # #     pseudo_label_low[i,:] = normalize_d(pseudo_label_low[i,:] * label_matrix_low[max_idx_low[i],:])
#         # pseudo_label[maskindex_high] = pseudo_label_high
#         # pseudo_label[maskindex_low] = pseudo_label_low
#         # max_probs, max_idx = torch.max(pseudo_label, dim=-1)
#         # mask = max_probs.ge(p_cutoff).float()

#         ## MNAR
#         for i in range(pseudo_label.size(0)):
#             pseudo_label[i,:] = normalize_d(pseudo_label[i,:] * label_matrix_high[max_idx[i],:])

#         max_probs, max_idx = torch.max(pseudo_label, dim=-1)
#         mask = max_probs.ge(p_cutoff).float()
#         mask_dis = max_probs.lt(p_cutoff).float()

#         pseudo_label_reverse = torch.softmax(logits_x_ulb_w_reverse, dim=-1)
#         max_probs_reverse, max_idx_reverse = torch.max(pseudo_label_reverse, dim=-1)

#         # label_matrix_tmp = (label_matrix/label_matrix.t()).detach()
#         # pseudo_label = pseudo_label * label_matrix_tmp[max_idx]
        
#         if use_hard_labels:
#             masked_loss = ce_loss(logits_s, max_idx, use_hard_labels, reduction='none') * mask
#             masked_reverse_loss = ce_loss(logits_x_ulb_s_reverse, pseudo_label_reverse, use_hard_labels = False, reduction='none')  
#             # masked_reverse_loss = ce_loss(logits_x_ulb_s_reverse, max_idx_reverse, reduction='none') * mask_dis   
#         else:
#             pseudo_label = torch.softmax(logits_w/T, dim=-1)
#             masked_loss = ce_loss(logits_s, pseudo_label, use_hard_labels) * mask
#         return masked_loss.mean(), masked_reverse_loss.mean(), mask.mean()

#     else:
#         assert Exception('Not Implemented consistency_loss')

# def consistency_loss_tric(logits_w, logits_s, label_matrix_high, label_matrix_low, label_bank, idx, name='ce', T=1.0, p_cutoff=0.0, use_hard_labels=True):
#     assert name in ['ce', 'L2']
#     logits_w = logits_w.detach()

#     if name == 'L2':
#         assert logits_w.size() == logits_s.size()
#         return F.mse_loss(logits_s, logits_w, reduction='mean')
    
#     elif name == 'L2_mask':
#         pass

#     elif name == 'ce':
#         pseudo_label = torch.softmax(logits_w, dim=-1)
#         # max_probs, max_idx = torch.max(pseudo_label, dim=-1)
#         # mask = max_probs.ge(p_cutoff).float()
#         # for i in range(pseudo_label.size(0)):
#         #     pseudo_label[i,:] = normalize_d(pseudo_label[i,:] * label_matrix_high[max_idx[i],:])
#         for i in range(pseudo_label.size(0)):
#             if idx[i].cpu().item() in label_bank.keys():
#                 pseudo_label[i,:] = normalize_d(pseudo_label[i,:] * label_matrix_high[label_bank[idx[i].cpu().item()],:])

#         max_probs, max_idx = torch.max(pseudo_label, dim=-1)
#         mask = max_probs.ge(p_cutoff).float()
#         mask_dis = max_probs.lt(p_cutoff).float()
     
#         if use_hard_labels:
#             masked_loss = ce_loss(logits_s, max_idx, use_hard_labels, reduction='none') * mask
#         else:
#             pseudo_label = torch.softmax(logits_w/T, dim=-1)
#             masked_loss = ce_loss(logits_s, pseudo_label, use_hard_labels) * mask
#         return masked_loss.mean(), mask.mean()

#     else:
#         assert Exception('Not Implemented consistency_loss')

# def consistency_loss_fixmatch(logits_w, logits_s, name='ce', T=1.0, p_cutoff=0.0, use_hard_labels=True):
#     assert name in ['ce', 'L2']
#     logits_w = logits_w.detach()

#     if name == 'L2':
#         assert logits_w.size() == logits_s.size()
#         return F.mse_loss(logits_s, logits_w, reduction='mean')
    
#     elif name == 'L2_mask':
#         pass

#     elif name == 'ce':
#         pseudo_label = torch.softmax(logits_w, dim=-1)
#         max_probs, max_idx = torch.max(pseudo_label, dim=-1)
#         mask = max_probs.ge(p_cutoff).float()
     
#         if use_hard_labels:
#             masked_loss = ce_loss(logits_s, max_idx, use_hard_labels, reduction='none') * mask
#         else:
#             pseudo_label = torch.softmax(logits_w/T, dim=-1)
#             masked_loss = ce_loss(logits_s, pseudo_label, use_hard_labels) * mask
#         return masked_loss.mean(), mask.mean()

#     else:
#         assert Exception('Not Implemented consistency_loss')

# from lib import *
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from train_utils import ce_loss

def normalize_d(x):
    x_sum = torch.sum(x)
    x = x / x_sum
    return x.detach()

def normalize_class(x):
    x_sum = torch.sum(x,dim=1)
    x = x / x_sum
    return x.detach()


class Get_Scalar:
    def __init__(self, value):
        self.value = value
        
    def get_value(self, iter):
        return self.value
    
    def __call__(self, iter):
        return self.value


def consistency_loss_dis_3(logits_x_ulb_w_reverse, logits_x_ulb_s_reverse, logits_w, logits_s, label_matrix_high, label_matrix_low, name='ce', T=1.0, p_cutoff=0.0, use_hard_labels=True):
    # for fixmatch4
    assert name in ['ce', 'L2']
    logits_w = logits_w.detach()
    logits_x_ulb_w_reverse = logits_x_ulb_w_reverse.detach()

    if name == 'L2':
        assert logits_w.size() == logits_s.size()
        return F.mse_loss(logits_s, logits_w, reduction='mean')
    
    elif name == 'L2_mask':
        pass

    elif name == 'ce':
        pseudo_label = torch.softmax(logits_w, dim=-1)
        max_probs, max_idx = torch.max(pseudo_label, dim=-1)
        # mask = max_probs.ge(p_cutoff).float()

        ## 
        # mask = max_probs.ge(p_cutoff).float()
        # mask_low = max_probs.lt(p_cutoff).float()
        # maskindex_high = np.where(mask.cpu()==1)[0]
        # maskindex_low = np.where(mask_low.cpu()==1)[0]

        # pseudo_label_high = pseudo_label[maskindex_high]
        # max_idx_high = max_idx[maskindex_high]
        
        # pseudo_label_low = pseudo_label[maskindex_low]
        # max_idx_low = max_idx[maskindex_low]
        # for i in range(pseudo_label_high.size(0)):
        #     pseudo_label_high[i,:] = normalize_d(pseudo_label_high[i,:] * label_matrix_high[max_idx_high[i],:])
        # pseudo_label_low = normalize_d(pseudo_label_low * label_matrix_low)
        # # for i in range(pseudo_label_low.size(0)):
        # #     pseudo_label_low[i,:] = normalize_d(pseudo_label_low[i,:] * label_matrix_low[max_idx_low[i],:])
        # pseudo_label[maskindex_high] = pseudo_label_high
        # pseudo_label[maskindex_low] = pseudo_label_low
        # max_probs, max_idx = torch.max(pseudo_label, dim=-1)
        # mask = max_probs.ge(p_cutoff).float()

        ## MNAR
        for i in range(pseudo_label.size(0)):
            pseudo_label[i,:] = normalize_d(pseudo_label[i,:] * label_matrix_high[max_idx[i],:])

        max_probs, max_idx = torch.max(pseudo_label, dim=-1)
        mask = max_probs.ge(p_cutoff).float()
        mask_dis = max_probs.lt(p_cutoff).float()

        pseudo_label_reverse = torch.softmax(logits_x_ulb_w_reverse, dim=-1)
        max_probs_reverse, max_idx_reverse = torch.max(pseudo_label_reverse, dim=-1)

        # label_matrix_tmp = (label_matrix/label_matrix.t()).detach()
        # pseudo_label = pseudo_label * label_matrix_tmp[max_idx]
        
        if use_hard_labels:
            masked_loss = ce_loss(logits_s, max_idx, use_hard_labels, reduction='none') * mask
            masked_reverse_loss = ce_loss(logits_x_ulb_s_reverse, pseudo_label_reverse, use_hard_labels = False, reduction='none')  
            # masked_reverse_loss = ce_loss(logits_x_ulb_s_reverse, max_idx_reverse, reduction='none') * mask_dis   
        else:
            pseudo_label = torch.softmax(logits_w/T, dim=-1)
            masked_loss = ce_loss(logits_s, pseudo_label, use_hard_labels) * mask
        return masked_loss.mean(), masked_reverse_loss.mean(), mask.mean()

    else:
        assert Exception('Not Implemented consistency_loss')

def consistency_loss_tric(logits_w, logits_s, label_matrix_high, label_matrix_low, label_bank, idx, name='ce', T=1.0, p_cutoff=0.0, use_hard_labels=True):
    assert name in ['ce', 'L2']
    logits_w = logits_w.detach()

    if name == 'L2':
        assert logits_w.size() == logits_s.size()
        return F.mse_loss(logits_s, logits_w, reduction='mean')
    
    elif name == 'L2_mask':
        pass

    elif name == 'ce':
        pseudo_label = torch.softmax(logits_w, dim=-1)
        max_probs, max_idx = torch.max(pseudo_label, dim=-1)
        mask = max_probs.ge(p_cutoff).float()
        for i in range(pseudo_label.size(0)):
            pseudo_label[i,:] = normalize_d(pseudo_label[i,:] * label_matrix_high[max_idx[i],:])
        # for i in range(pseudo_label.size(0)):
        #     if idx[i].cpu().item() in label_bank.keys():
        #         pseudo_label[i,:] = normalize_d(pseudo_label[i,:] * label_matrix_high[label_bank[idx[i].cpu().item()],:])

        max_probs, max_idx = torch.max(pseudo_label, dim=-1)
        mask = max_probs.ge(p_cutoff).float()
        mask_dis = max_probs.lt(p_cutoff).float()
     
        if use_hard_labels:
            masked_loss = ce_loss(logits_s, max_idx, use_hard_labels, reduction='none') * mask
        else:
            pseudo_label = torch.softmax(logits_w/T, dim=-1)
            masked_loss = ce_loss(logits_s, pseudo_label, use_hard_labels) * mask
        return masked_loss.mean(), mask.mean()

    else:
        assert Exception('Not Implemented consistency_loss')