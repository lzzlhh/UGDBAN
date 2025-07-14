

import torch
import numpy as np
def calc_coeff(iter_num,max_iter):
    alpha = 10
    low = 0.0
    high = 1.0
    return np.float(2.0 * (high - low) / (1.0 + np.exp(-alpha * iter_num / max_iter)) - (high - low) + low)

def grl_hook(coeff):
    def fun1(grad):
        return -coeff * grad.clone()
    return fun1
def Entropy(input_):
    bs = input_.size(0)
    epsilon = 1e-5
    entropy = -input_ * torch.log(input_ + epsilon)
    entropy = torch.sum(entropy, dim=1)
    return entropy
def EntropyLoss(input_):
    mask = input_.ge(0.000001)###与0.000001对比，大于则取1，反之取0
    mask_out = torch.masked_select(input_, mask)##平铺成为一维向量
    entropy = -(torch.sum(mask_out * torch.log(mask_out)))##计算熵
    return entropy / float(input_.size(0))
