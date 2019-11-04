# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 23:46:26 2019

@author: Paul
"""
import numpy as np
import torch
from torchvision.datasets import mnist # 导入 pytorch 内置的 mnist 数据

from torch import nn
from torch.autograd import Variable
def train(net,train_data,test_data,i,optimizer,criterion):
    losses = []
    acces = []
    eval_losses = []
    eval_acces = []
    
    for e in range(i):
        train_loss = 0
        train_acc = 0
        net.train()
        for im, label in train_data:
            im = Variable(im)
            label = Variable(label)
            # 前向传播
            out = net(im)
            loss = criterion(out, label)
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # 记录误差
            train_loss += loss.item()
            # 计算分类的准确率
            _, pred = out.max(1)
            num_correct = (pred == label).sum().item()
            acc = num_correct / im.shape[0]
            train_acc += acc
    
        losses.append(train_loss / len(train_data))
        acces.append(train_acc / len(train_data))
        # 在测试集上检验效果
        eval_loss = 0
        eval_acc = 0
        net.eval() # 将模型改为预测模式
        for im, label in test_data:
            im = Variable(im)
            label = Variable(label)
            out = net(im)
            loss = criterion(out, label)
            # 记录误差
            eval_loss += loss.item()
            # 记录准确率
            _, pred = out.max(1)
            num_correct = (pred == label).sum().item()
            acc = num_correct / im.shape[0]
            eval_acc += acc
    
        eval_losses.append(eval_loss / len(test_data))
        eval_acces.append(eval_acc / len(test_data))
        print('epoch: {}, Train Loss: {:.6f}, Train Acc: {:.6f}, Eval Loss: {:.6f}, Eval Acc: {:.6f}'
              .format(e, train_loss / len(train_data), train_acc / len(train_data), 
                         eval_loss / len(test_data), eval_acc / len(test_data)))