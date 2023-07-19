import torch
import torch.nn.functional as F
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

def consistency_loss_prg(logits_w, logits_s, H_prime, label_bank, idx, last=False, name='ce', T=1.0, p_cutoff=0.0, use_hard_labels=True):
    assert name in ['ce', 'L2']
    logits_w = logits_w.detach()

    if name == 'L2':
        assert logits_w.size() == logits_s.size()
        return F.mse_loss(logits_s, logits_w, reduction='mean')
    
    elif name == 'L2_mask':
        pass

    elif name == 'ce':
        pseudo_label = torch.softmax(logits_w/T, dim=-1)
        max_probs, max_idx = torch.max(pseudo_label, dim=-1)
        mask = max_probs.ge(p_cutoff).float()
        if last:
            for i in range(pseudo_label.size(0)):
                if idx[i].cpu().item() in label_bank.keys():
                    pseudo_label[i,:] = normalize_d(pseudo_label[i,:] * H_prime[label_bank[idx[i].cpu().item()],:])
        else:
            for i in range(pseudo_label.size(0)):
                pseudo_label[i,:] = normalize_d(pseudo_label[i,:] * H_prime[max_idx[i],:])

        max_probs, max_idx = torch.max(pseudo_label, dim=-1)
        mask = max_probs.ge(p_cutoff).float()
     
        if use_hard_labels:
            masked_loss = ce_loss(logits_s, max_idx, use_hard_labels, reduction='none') * mask
        else:
            masked_loss = ce_loss(logits_s, pseudo_label, use_hard_labels) * mask
        return masked_loss.mean(), mask.mean()

    else:
        assert Exception('Not Implemented consistency_loss')
