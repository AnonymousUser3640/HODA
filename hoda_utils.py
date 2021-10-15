import torchvision.transforms.functional as functional
from hoda_models import  DenseNet121, VGG, densenet_cifar, LeNet, ResNet18, ResNet50, MobileNetV2

import os
import sys
import time
import math
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


# Turn off requires_grad of feature extractor layers for transfer learning
def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

# A new layer for normalize data that can be considered as the first layer of model
class Normalize(nn.Module):
    def __init__(self, mean, std):
        super(Normalize, self).__init__()
        self.mean = torch.tensor(mean, device=device)
        self.std = torch.tensor(std, device=device)

    def forward(self, input):
        input = functional.normalize(input, self.mean, self.std)
        return input

#Supported target dataset: cifar10, cifar100, mnist, svhn, stl10, caltech256, cub
#Supported model: resnet18, resnet50, densenet121, vgg19
def get_model(target_dataset,model_name,pretrained = False,feature_extracting=False,device='cuda'):
  if target_dataset in ['cifar10','cifar100','tinyimagenet','stl10']:
    input_dim = 32
  elif target_dataset in ['mnist','svhn']:
    input_dim = 28
  elif target_dataset in ['caltech256','cub','imagenet']:
    input_dim = 224
  else:
    assert False, 'Error, Unknown dataset'

  if target_dataset in ['cifar10','mnist','svhn','stl10']:
    out_dim = 10
  elif target_dataset in ['cifar100']:
    out_dim = 100
  elif target_dataset in ['cub']:
    out_dim = 200
  elif target_dataset in ['caltech256']:
    out_dim = 256
  else:
    assert False, 'Error, Unknown target dataset'

  if input_dim == 32:
    if model_name == 'resnet18' or model_name == 'resnet18_2' or model_name == 'resnet18_3':
      net = ResNet18(out_dim)
      print("resnet18!!!")
    elif model_name == 'resnet50':
      print("resnet50!!!")
      net = ResNet50(out_dim)
    elif model_name == 'resnet34':
      print("resnet34!!!")
      net = ResNet34(out_dim)
    elif model_name == 'vgg19':
      net = VGG('VGG19',out_dim)
    elif model_name == 'vgg16':
      print("VGG16")
      net = VGG('VGG16',out_dim)
    elif model_name == 'vgg16_2':
      print("VGG16")
      net = VGG('VGG16',out_dim)
    elif model_name == 'mobilenet' or model_name == 'mobilenet_2' or model_name == 'mobilenet_3':
      print("mobilenet")
      net = MobileNetV2(out_dim)
    elif model_name == 'densenet121' or model_name == 'densenet121_2' or model_name == 'densenet121_3':
      net = DenseNet121(out_dim)
    else:
      assert False, 'Error, Unknown model'
  elif input_dim == 28:
    if model_name == 'lenet':
      net = LeNet()
    else:
      assert False, 'Error, Unknown model'
  elif input_dim == 224:
    if model_name == 'resnet18':
      net = models.resnet18(pretrained=pretrained, progress=True)
      if feature_extracting == True:
        set_parameter_requires_grad(net, feature_extracting = True)
      num_ftrs = net.fc.in_features
      net.fc = nn.Linear(num_ftrs, out_dim)
    elif model_name == 'resnet50':
      net = models.resnet50(pretrained=pretrained, progress=True)
      if feature_extracting == True:
        set_parameter_requires_grad(net, feature_extracting = True)
      num_ftrs = net.fc.in_features
      net.fc = nn.Linear(num_ftrs, out_dim)
    elif model_name == 'vgg19':
      net = models.vgg19(pretrained=pretrained, progress=True)
      if feature_extracting == True:
        set_parameter_requires_grad(net, feature_extracting = True)
      net.classifier[6] = nn.Linear(4096,out_dim)
    elif model_name == 'densenet121':
      net = models.densenet121(pretrained=pretrained, progress=True)
      if feature_extracting == True:
        set_parameter_requires_grad(net, feature_extracting = True)
      num_ftrs = net.classifier.in_features
      net.classifier = nn.Linear(num_ftrs, out_dim)
    else:
      assert False, 'Error, Unknown model'
  else:
    assert False, 'Error, Unknown input_dim'
  '''
  if target_dataset in ['cifar10', 'cifar100']:
    net = nn.Sequential(Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010]),net)
  if target_dataset in ['caltech256', 'cub']:
    net = nn.Sequential(Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),net)
  '''
  net = net.to(device)
  return net


# Draw loss and acc plot
def draw_plot(first_serie,second_serie,name1,name2,title,ylabel,xlabel):
  plt.plot(first_serie)
  plt.plot(second_serie)
  plt.title(title)
  plt.ylabel(ylabel)
  plt.xlabel(xlabel)
  plt.legend([name1, name2])
  plt.show()

#print an image
def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    inp = np.clip(inp, 0, 1)
    if(len(inp.shape) == 3):
      plt.imshow(np.squeeze(inp))
    elif (len(inp.shape) == 2):
      plt.imshow(inp)
    else:
      print("Higher dimensional data")
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated

def print_num_of_samples_in_each_classes(dataset,num_classes=10):
    num_samples = np.zeros(num_classes)
    for batch_idx, (inputs, targets) in enumerate(dataset):
      for i in targets:
        num_samples[i] += 1
    for i in range(num_classes):
      print(classes[i],'\t',num_samples[i])

# print confusion matrix
def print_statistics(y_true=None,y_pred=None,num_classes=10,classes_name=None):
  cm = confusion_matrix(y_true, y_pred)
  recall = np.zeros(num_classes)
  precision = np.zeros(num_classes)
  pdframe = np.zeros((num_classes+1,num_classes+1))
  correct = 0
  for i in range(num_classes):
    recall[i] = np.round(cm[i,i]/np.sum(cm[i,:]),4)
    precision[i] = np.round(cm[i,i]/np.sum(cm[:,i]),4)
    correct += cm[i,i]
  acc = np.round(correct / np.sum(cm),4)
  pdframe[:num_classes,:num_classes] = cm
  pdframe[num_classes,:num_classes] = precision * 100
  pdframe[:num_classes,num_classes] = recall * 100
  pdframe[num_classes,num_classes] = acc * 100
  if classes_name is not None:
    r = list(classes_name)
    c = list(classes_name)
  else: 
    name_list = []
    for i in range(num_classes):
      name_list.append('class '+str(i))
    r = name_list
    c = name_list
  r.append('Precision')
  c.append('Recall')
  df = pd.DataFrame(pdframe,index=r,columns=c)
  print(df.to_string())
 
# Loss function for data with vetor label
class SoftLabelLoss(nn.Module):
    def __init__(self, dim=-1):
        super(SoftLabelLoss, self).__init__()
        self.dim = dim

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        loss = torch.sum(-target * pred, dim=self.dim)
        return torch.mean(loss)

# integrated information about learning process for one loader
class learning_information:
  def __init__(self, name, loader,num_classes=10):
    self.name = name
    self.learning_index = np.zeros(len(loader.dataset))
    self.learning_agree_index = np.zeros(len(loader.dataset))
    self.learning_last_agree_index = np.zeros(len(loader.dataset))
    self.pred_hist = np.zeros((len(loader.dataset),num_classes))
    self.label = np.zeros(len(loader.dataset))

def get_histogram(seq=None,max_degree=99,verbose=0):
  his = np.zeros(max_degree+1)
  for i in range(max_degree+1):
    his[i] = np.sum(seq == i)
  if verbose == 1:
    fig = plt.figure()
    ax = fig.add_axes([0,0,1,1])
    ax.bar(np.arange(max_degree+1),his)
  return his


def draw_plot_of_trust(learn_inf):
  correct_classified_samples = (np.argmax(learn_inf.pred_hist,axis=1) == learn_inf.label)
  percentage_of_correct_classified_samples = np.zeros(10)
  precentage_of_samples_in_hardness_group = np.zeros(10)
  for i in range(10):
    learn_ind = 99 - learn_inf.learning_index
    hardness_group_index = ( learn_ind >= (i * 10) ) * ( learn_ind < ( (i+1) * 10 ) )
    percentage_of_correct_classified_samples[i] = np.sum(correct_classified_samples[hardness_group_index]) / np.sum(hardness_group_index) * 100
    precentage_of_samples_in_hardness_group[i] = np.sum(hardness_group_index) / len(correct_classified_samples) * 100

  labels = ['0-9', '10-19', '20-29', '30-39', '40-49', '50-59', '60-69', '70-79', '80-89', '90-99']
  width = 0.45       # the width of the bars: can also be len(x) sequence

  fig, ax = plt.subplots()

  ax.bar(labels, percentage_of_correct_classified_samples, width, label='Correct Classified')
  ax.bar(labels, 100 - percentage_of_correct_classified_samples, width, bottom= percentage_of_correct_classified_samples, label='Wrong Classified')
  ax.plot(precentage_of_samples_in_hardness_group,label='precentage of samples',color='green')
  ax.set_ylabel('%')
  ax.set_xlabel('hardness degree range')
  ax.set_title('Precentage of correct classified samples in various hardness degree ranges')
  ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

  plt.show()
  return percentage_of_correct_classified_samples, precentage_of_samples_in_hardness_group