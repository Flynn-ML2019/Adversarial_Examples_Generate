#!/usr/bin/env python3
# -*- coding: utf-8 -*-    
from torch.utils.data import DataLoader, Dataset
import numpy as np
import torch
from sklearn.model_selection import train_test_split
class TensorDataset(Dataset): #包装数据用于迭代
    def __init__(self, data_tensor, target_tensor):
        self.data_tensor = torch.LongTensor(data_tensor)
        self.target_tensor = torch.LongTensor(target_tensor)
    def __getitem__(self, index):
        return self.data_tensor[index], self.target_tensor[index]
    def __len__(self):
        return self.data_tensor.shape[0]             

class Tools(object):
    def get_data(self,opt): #获取句子 标签 句子有效长度
        labels = []
        arr = np.zeros([opt.allSequence, opt.maxSelength])
        good=opt.ids[:opt.allSequence//2]
        bad=opt.ids[opt.allSequence//2:]
        num=0   
        for i in range(opt.allSequence//2):        
            line=good[i]
            if len(line) > opt.maxSelength:
                    arr[num] = line[0:opt.maxSelength]
            else:
                    arr[num] = line  
            labels.append([1, 0])
            num=num+1
            line=bad[i]
            if len(line) > opt.maxSelength:
                    arr[num] = line[0:opt.maxSelength]
            else:
                    arr[num] = line
            labels.append([0, 1])
            num=num+1
        return arr,labels
    def split_data(self,opt,inputs_,lables_):
        X_train, X_valid, y_train, y_valid = train_test_split(inputs_, lables_, test_size=0.5, random_state=42,stratify=lables_)
        trainset = TensorDataset(X_train, y_train)#训练集
        testset = TensorDataset(X_valid, y_valid)#测试集
        trainloader = torch.utils.data.DataLoader(trainset, batch_size=opt.batch_size_train)
        testloader = torch.utils.data.DataLoader(testset, batch_size=opt.batch_size_train)
        return trainloader,testloader