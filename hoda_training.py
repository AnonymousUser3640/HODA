from hoda_utils import get_model,SoftLabelLoss,print_statistics,learning_information, draw_plot, imshow

import os
import sys
import time
import math
import zipfile
import tarfile
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.nn.init as init
import torch.nn.modules.flatten as flatten

import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import numpy as np

import os
import argparse
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
import pandas as pd

class model_class:
  def __init__(self, net, num_classes = 10, classes = None, device = 'cuda'):
    self.net = net
    self.num_classes = num_classes
    self.classes = classes
    self.device = device
    self.criterion = nn.CrossEntropyLoss()
    self.soft_criterion = SoftLabelLoss()
  
  def train_one_epoch(self, loader, soft_label = False):
    self.net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(loader):
      inputs, targets = inputs.to(self.device), targets.to(self.device)
      self.optimizer.zero_grad()
      outputs = self.net(inputs)
      if soft_label:
        loss = self.soft_criterion(outputs, targets)
      else:
        loss = self.criterion(outputs, targets)
      loss.backward()
      self.optimizer.step()
      train_loss += loss.item()
      _, predicted = outputs.max(1)
      total += targets.size(0)
      if soft_label:
        _,targets = targets.max(1)
      correct += predicted.eq(targets).sum().item()
    return (train_loss/(batch_idx+1)), 100.*correct/total, correct, total

  def test(self, loader, calc_loss=True, soft_label = False):
    label = []
    pred = []
    self.net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
      for batch_idx, (inputs, targets) in enumerate(loader):
        inputs, targets = inputs.to(self.device), targets.to(self.device)
        label.extend(targets.cpu().numpy())
        outputs = self.net(inputs)
        pred.extend(outputs.cpu().numpy())
        if len(list(targets.shape)) == 1:	
          soft_label = False
        if calc_loss:
          if soft_label:
            loss = self.soft_criterion(outputs, targets)
          else:
            loss = self.criterion(outputs, targets)
          test_loss += loss.item()
        else:
          test_loss = 0
        _, predicted = outputs.max(1)
        total += targets.size(0)
        if soft_label:
          _,targets = targets.max(1)
        correct += predicted.eq(targets).sum().item()
    label = np.array(label)
    pred = np.array(pred)
    return (test_loss/(batch_idx+1)), 100.*correct/total, correct, total, label, pred
            
  def save_model(self, file_name, acc, epoch_num):
    existence = self.load_model(file_name)
    if existence is not None:
      assert False, "Critical Error: You are writing on a saved model " + file_name
    state = {
            'net': self.net.state_dict(),
            'acc': acc,
            'epoch': epoch_num,
    }
    if not os.path.isdir('checkpoint'):
      os.mkdir('checkpoint')
    path = './checkpoint/' + file_name
    print('Saving', path)
    torch.save(state, path)


  def load_model(self, file_name):
    path = './checkpoint/' + file_name
    print('==> check for Loading from checkpoint', path)
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    if os.path.isfile(path):
      checkpoint = torch.load(path)
      self.net.load_state_dict(checkpoint['net'])
      acc = checkpoint['acc']
      num_epoch = checkpoint['epoch']
      return acc, num_epoch
    else:
      print(path,"does not exist")
      return None

  def save_learning_inf(self,learn_inf):
    if not os.path.isdir('learning_information'):
      os.mkdir('learning_information')
    path = './learning_information/' + learn_inf.name
    np.savez(path + '.npz',learning_index=learn_inf.learning_index,last_pred=learn_inf.pred_hist,label=learn_inf.label,learning_agree_index=learn_inf.learning_agree_index,learning_last_agree_index=learn_inf.learning_last_agree_index)
    print(path + '.npz','is saved')

  def load_learning_inf(self,loader,name):
    learn_inf = learning_information(name,loader,self.num_classes)
    path = './learning_information/' + name
    print('==> check for Loading from checkpoint', path)
    assert os.path.isdir('learning_information'), 'Error: no learning_information directory found!'
    if os.path.isfile(path + '.npz'):
      print(path + '.npz is loading!')
      index_file = np.load(path + '.npz')
      learn_inf.learning_index = index_file['learning_index']
      learn_inf.pred_hist = index_file['last_pred']
      learn_inf.label = index_file['label']
      learn_inf.learning_agree_index = index_file['learning_agree_index']
      learn_inf.learning_last_agree_index = index_file['learning_last_agree_index']
      return learn_inf
    else:
      print(path+'.npz',"does not exist")
      return None

  def update_learning_index(self,pred, prediction_hist, learning_index, verbose = 1):
    if verbose == 1:
      print("prediction_hist",np.argmax(prediction_hist,axis=1)[:10])
      print("pred",np.argmax(pred,axis=1)[:10])
    for i in range(len(pred)):
      if np.argmax(pred[i]) == np.argmax(prediction_hist[i]):
        learning_index[i] +=1
      else:
        learning_index[i] = 0
    preciction_hist = pred
    if verbose == 1:
      print("learning_index",learning_index[:10])
    return learning_index, preciction_hist
  
  def evaluate(self, loader, file_name = None, soft_label = False, verbose = 1):
    
    if file_name is not None:
      self.load_model(file_name)
    loss, acc, correct, total, label, pred = self.test(loader, soft_label=soft_label)
    if verbose > 0:
      print('Loss: %.3f | Acc: %.3f (%d/%d)' % (loss, acc, correct, total))
    if verbose == 2:
      print_statistics(y_true=label, y_pred=np.argmax(pred,axis=1),num_classes = self.num_classes ,classes_name=self.classes)
    return self.net, label, pred

  # Loader name for each loader in loader_list is mandatory.
  # First loader in loader_list must be validation data with the same number of classes with trainloader
  def train(self, trainloader,loader_list = None, loader_name = None, epoch = 3,lr=0.1,scheduler=None,optimizer=None, soft_label= False, save_model = True, save_model_each_epoch = False, file_name=None, return_learning_index = False,resume=None):
    assert len(loader_list) == len(loader_name), 'Error: sizes of loader_list and loader_name are not the same!'
    if return_learning_index:
      learning_inf_list = []
      if loader_list is not None: 
        for loader,name in zip(loader_list,loader_name):
          lear_inf = learning_information(name,loader,self.num_classes)
          learning_inf_list.append(lear_inf)
    assert len(loader_list) > 0, 'Error: no validation loader!'
    if optimizer is None:
      print(lr)
      self.optimizer = optim.SGD(self.net.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    else:
      self.optimizer = optimizer
    if scheduler is None:
      #self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=epoch,verbose=True)
      #self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.1, threshold=0.01, patience=5,verbose=True)
      self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=1, gamma=0.955,verbose=True)
    else:
      self.scheduler = scheduler
    train_loss = []
    val_loss = []
    train_acc = []
    val_acc = []
    if resume is not None:
      print("resume:",resume)	
      self.load_model(resume)
    for i in range(epoch):
      print("******************** Epoch:", i,'/',epoch,"********************" )
      start = time.time()
      tloss,tacc,correct,total = self.train_one_epoch(trainloader,soft_label)
      print('Training - Loss: %.3f | Acc: %.3f (%d/%d)' % (tloss, tacc, correct, total))
      train_loss.append(tloss)
      train_acc.append(tacc)
      
      for ind, loader in enumerate(loader_list):
        calc_loss = False
        if hasattr(loader, 'dataset'):	
          if hasattr(loader.dataset,'classes'):	
            if self.num_classes == len(loader.dataset.classes):	
              calc_loss = True	
            else:
              calc_loss = False
          elif hasattr(loader.dataset,'dataset'):	
            if hasattr(loader.dataset.dataset, 'classes'):	
              if self.num_classes == len(loader.dataset.dataset.classes):	
                calc_loss = True
        loss, acc, correct, total, label, pred = self.test(loader,calc_loss=calc_loss, soft_label=soft_label)
        if ind == 0:
          vloss = loss
          vacc = acc
          valid_label = label
          valid_pred = pred
        print('%s - Loss: %.3f | Acc: %.3f (%d/%d)' % (loader_name[ind],loss, acc, correct, total))
        if return_learning_index:
          print('\n')
          print("Start calculating learning index for",loader_name[ind])
          learning_inf_list[ind].learning_index, learning_inf_list[ind].pred_hist = self.update_learning_index(pred, learning_inf_list[ind].pred_hist, learning_inf_list[ind].learning_index)
          learning_inf_list[ind].label = label
          print("End calculating learning index for",loader_name[ind],"\n")
      val_loss.append(vloss)
      val_acc.append(vacc)
      if save_model_each_epoch:
        self.save_model(str(i)+'_'+file_name,vacc,i)
      self.scheduler.step()
      end = time.time()
      print("Elapsed time:",end - start)
    if save_model:
      self.save_model(file_name,vacc,epoch)
    draw_plot(train_loss,val_loss,'train','val','loss','loss','epoch')
    draw_plot(train_acc,val_acc,'train','val','Acc','acc','epoch')
    print_statistics(y_true=valid_label, y_pred=np.argmax(valid_pred,axis=1),num_classes = self.num_classes ,classes_name=self.classes)  

    if return_learning_index:
      for learning_inf in learning_inf_list:
        learning_inf.learning_index[learning_inf.learning_index == epoch] = epoch - 1
        self.save_learning_inf(learning_inf)
      return self.net, learning_inf_list
    else:
      return self.net, None

  def epoch_range_get_learning_index(self,loader = None, loader_name = None, file_name=None, epoch_range = None, calc_loss=False,soft_label= False):
    learn_inf = self.load_learning_inf(loader,loader_name)
    if learn_inf is not None:
      return self.net, learn_inf
    learn_inf = learning_information(loader_name,loader,self.num_classes)
    start_epoch = 0
    for i in epoch_range:
      print(i,'/',epoch_range)
      start = time.time()
      acc,num_epoch = self.load_model(str(i)+'_'+file_name)
      print("Model in epoch",num_epoch,"has been loaded with acc", acc)
      vloss, vacc, correct, total, label, pred = self.test(loader,calc_loss=calc_loss, soft_label=soft_label)
      print('Loss: %.3f | Acc: %.3f (%d/%d)' % (vloss, vacc, correct, total))
      learn_inf.learning_index, learn_inf.pred_hist = self.update_learning_index(pred, learn_inf.pred_hist, learn_inf.learning_index)
      learn_inf.label = label
      end = time.time()
      print("Elapsed time:",end - start)
    learn_inf.learning_index[learn_inf.learning_index == len(epoch_range)] = len(epoch_range) - 1
    self.save_learning_inf(learn_inf)
    return self.net, learn_inf
