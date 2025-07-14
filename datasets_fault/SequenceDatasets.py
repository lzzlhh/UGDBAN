#!/usr/bin/python
# -*- coding:utf-8 -*-

import torch
from torch.utils.data import Dataset
import os
from PIL import Image
from torchvision import transforms
import numpy as np
from datasets_fault.sequence_aug import *
class dataset(Dataset):
    def __init__(self, list_data, test=False, transform=None):
        # 初始化代码
        self.test = test
        if self.test:
            self.seq_data = list_data['data'].tolist()
        else:
            self.seq_data = list_data['data'].tolist()
            self.labels = list_data['label'].tolist()
        if transform is None:
            self.transforms = Compose([
                Reshape()
            ])
        else:
            self.transforms = transform
    def __len__(self):
        return len(self.seq_data)

    def __getitem__(self, item):
        if self.test:
            seq = self.seq_data[item]
            seq = self.transforms(seq)
            return seq, item  # 返回seq和item
        else:
            seq = self.seq_data[item]
            label = self.labels[item]
            seq = self.transforms(seq)
            return seq, label, item  # 返回seq、label和item

class newdataset(Dataset):
    def __init__(self, data,label, test=False, transform='mean-std'):
        # 初始化代码
        self.test = test
        if self.test:
            self.seq_data = data
        else:
            self.seq_data = data
            self.labels = label
        if transform is None:
            self.transforms = Compose([
                Reshape()
            ])
        else:
            self.transforms = Compose([
                Reshape(),
                Retype(),
                Normalize(transform),
            ])
    def __len__(self):
        return len(self.seq_data)

    def __getitem__(self, item):
        if self.test:
            seq = self.seq_data[item]
            seq = self.transforms(seq)
            return seq, item  # 返回seq和item
        else:
            seq = self.seq_data[item]
            label = self.labels[item]
            seq = self.transforms(seq)
            return seq, label, item  # 返回seq、label和item
class VibrationDataset(Dataset):
    def __init__(self, data, targets, indexs, pseudo_idx=None, pseudo_target=None,
                 nl_idx=None, nl_mask=None):

        self.targets = np.array(targets)
        self.nl_mask = np.ones((len(self.targets), len(np.unique(self.targets))))

        if nl_mask is not None:
            self.nl_mask[nl_idx] = nl_mask

        if pseudo_target is not None:
            self.targets[pseudo_idx] = pseudo_target

        if indexs is not None:
            # data = data.transpose(0, 2, 1)
            indexs = np.array(indexs)
            self.data = np.array(data)  # 将数据转换为数组
            self.data = self.data.transpose(0, 2, 1)
            self.data = self.data[indexs]  # 使用整数标量数组作为索引
            self.targets = np.array(self.targets)[indexs]
            self.nl_mask = np.array(self.nl_mask)[indexs]
            self.indexs = indexs
        else:
            self.indexs = np.arange(len(self.targets))
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        seq, target = self.data[index], self.targets[index]
        # seq = Image.fromarray(img)
        return seq, target, self.indexs[index], self.nl_mask[index]