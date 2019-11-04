# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import sys 
sys.path.append('d:\\大数据相关')

import torch
from torch.autograd import Variable
from torch import nn
from torch.utils.data import DataLoader

from torchvision import transforms as tfs
from torchvision.datasets import MNIST


data_tf = tfs.Compose([
        tfs.ToTensor(),
        tfs.Normalize([0.5],[0.5])
        ])

train_set = MNIST('./data',train = True, transform = data_tf,download = True)
test_set = MNIST('./data',train = False, transform = data_tf,download = True)

train_data = DataLoader(train_set,64,True,num_workers = 4)
test_data = DataLoader(test_set,128,False,num_workers = 4)

class rnn_classify(nn.Module):
    def __init__(self,in_feature=28,hidden_feature=100,num_class=10,num_layers=2):
        super(rnn_classify,self).__init__()
        self.rnn = nn.LSTM(in_feature,hidden_feature,num_layers) #使用两层lstm
        self.classifier = nn.Linear(hidden_feature,num_class) #将最后一个rnn的输出使用全连接得到最后的结果
        
    def forward(self,x):
        '''
        x大小为（batch,1,28,28)所以我们需要转换成（28,batch,28)
        '''
        x = x.squeeze() #去掉(batch,1,28,28)中的1,变为（batch,28,28）
        x = x.permute(2,0,1) #将最后一维放到第一维
        out,_ = self.rnn(x) #使用默认的隐藏状态，得到的out是(28,batch,hidden_feature)
        out = out[1,:,:] #取序列中最后一个，大小是(batch,hidden_feature)
        out = self.classifier(out)
        return out
        
net = rnn_classify()
criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adadelta(net.parameters(),1e-1)

from utils import train
train(net,train_data,test_data,10,optimizer,criterion)