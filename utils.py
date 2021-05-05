import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import os

def clip_gradient(optimizer, grad_clip):
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)



def adjust_lr(optimizer, epoch, decay_rate, decay_epoch):
    if (epoch+1) in decay_epoch:
        # decay = decay_rate ** ((epoch) // decay_epoch) 
        for param_group in optimizer.param_groups:
            param_group['lr'] *= decay_rate
    for param_group in optimizer.param_groups:
        print('Learning Rate: {}'.format(param_group['lr']))


def bce2d_new(input, target, reduction='mean'):
    assert(input.size() == target.size())
    pos = torch.eq(target, 1).float()
    neg = torch.eq(target, 0).float()
    # ing = ((torch.gt(target, 0) & torch.lt(target, 1))).float()

    num_pos = torch.sum(pos)
    num_neg = torch.sum(neg)
    num_total = num_pos + num_neg

    alpha = num_neg  / num_total
    beta = 1.1 * num_pos  / num_total
    # target pixel = 1 -> weight beta
    # target pixel = 0 -> weight 1-beta
    weights = alpha * pos + beta * neg

    return F.binary_cross_entropy_with_logits(input, target, weights, reduction=reduction)


def ACTLoss(p_sal, p_edge, g_sal, g_edge, m=4):
    bce = nn.BCEWithLogitsLoss(reduction='none')
    w = torch.where(p_edge>g_edge, p_edge, g_edge)
    act = (bce(p_sal, g_sal) * (w * m + 1)).mean()
    return act



class CEL(nn.Module):
    def __init__(self):
        super(CEL, self).__init__()
        self.eps = 1e-6

    def forward(self, pred, target):
        pred = pred.sigmoid()
        intersection = pred * target
        numerator = (pred - intersection).sum() + (target - intersection).sum()
        denominator = pred.sum() + target.sum()
        return numerator / (denominator + self.eps)


def print_network(model):
    num_params = 0
    for p in model.parameters():
        num_params += p.numel()
    print("The number of parameters: {}".format(num_params))
