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

def get_dataset(name, train, shuffle, aug, batch_size,local_directory = None, dataset_drive_directory=None):
  if name == 'cifar10':
    if aug:
      transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
      ])
    else:
      transform = transforms.Compose([
        transforms.ToTensor(),
        #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
      ])
    dataset = torchvision.datasets.CIFAR10(root=local_directory, train=train, download=True, transform=transform)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
  
  elif name == 'cifar100':
    if aug:
      transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
      ])
    else:
      transform = transforms.Compose([
        transforms.ToTensor(),
      ])
    dataset = torchvision.datasets.CIFAR100(root=local_directory, train=train, download=True, transform=transform)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

  elif name == 'stl10':
    path = local_directory + 'stl10_binary'
    if os.path.isdir(path) == False:
      with tarfile.open(dataset_drive_directory + 'stl10_binary.tar.gz') as tar_ref:
        tar_ref.extractall(local_directory)
    else:
      print("STL10 has been loaded!!!")
    if aug:
      transform = transforms.Compose([
        transforms.Resize(32),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
      ])
    else:
      transform = transforms.Compose([
        transforms.Resize(32), 
        transforms.ToTensor(),
        #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
      ])
    if train == True:
      split = 'train'
    else:
      split = 'test'
    dataset = torchvision.datasets.STL10(root=local_directory, split=split, download=False, transform=transform)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

  elif name == 'mnist':
    from six.moves import urllib
    opener = urllib.request.build_opener()
    opener.addheaders = [('User-agent', 'Mozilla/5.0')]
    urllib.request.install_opener(opener)
    if aug:
      transform = transforms.Compose([
        transforms.RandomCrop(28, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
      ])
    else:
      transform = transforms.Compose([
        transforms.ToTensor(),
        #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
      ])
    dataset = torchvision.datasets.MNIST(root=local_directory, train=train, download=True, transform=transform)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
  elif name == 'emnist':
    print("EMNIST is not supported!!!")
    return None, None
    if aug:
      transform = transforms.Compose([
        transforms.RandomCrop(28, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
      ])
    else:
      transform = transforms.Compose([
        transforms.ToTensor(),
        #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
      ])
    dataset = torchvision.datasets.EMNIST(root=local_directory, split='balanced', download=True, transform=transform)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
  
  elif name == 'tinyimagenet':
    path = local_directory + 'tiny-imagenet-200'
    if os.path.isdir(path) == False:
      with zipfile.ZipFile(dataset_drive_directory + 'tiny-imagenet-200.zip', 'r') as zip_ref:
        zip_ref.extractall(local_directory)
    else:
      print("TinyImagenet has been loaded!!!")
    transform = {
    'train': transforms.Compose([
        transforms.Resize(32),
        transforms.ToTensor(),
    ]),
    'val': transforms.Compose([
             
            transforms.ToTensor(),
    ]),
    }
    data_dir = local_directory + 'tiny-imagenet-200'
    tin_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x), transform[x]) for x in ['train', 'val']}
    tin_loaders = {x: torch.utils.data.DataLoader(tin_datasets[x], batch_size=batch_size, shuffle=False) for x in ['train', 'val']}
    dataset = tin_datasets['train']
    loader = tin_loaders['train']

  elif name == 'svhn':
    if aug:
      transform = transforms.Compose([
        transforms.Resize(28),
        transforms.RandomCrop(28, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.Grayscale(),
        transforms.ToTensor(),
        #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
      ])
    else:
      transform = transforms.Compose([
        transforms.Resize(28),
        transforms.Grayscale(),
        transforms.ToTensor(),
        #transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
      ])
    if train == True:
      split = 'train'
    else:
      split = 'test'
    dataset = torchvision.datasets.SVHN(root=local_directory, split=split, download=True, transform=transform)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
  elif name == 'caltech256':
    path = local_directory + '256_ObjectCategories'
    if os.path.isdir(path) == False:
      os.makedirs(local_directory, exist_ok=True)
      with tarfile.open(dataset_drive_directory + '256_ObjectCategories.tar') as tar_ref:
        tar_ref.extractall(local_directory)
      import glob
      from shutil import copyfile
      for cat in range(1, 258):
        if cat == 257:
          continue
        cat_dirs = glob.glob(os.path.join(local_directory, "256_ObjectCategories", '%03d*' % cat))
        os.makedirs(os.path.join(local_directory + "Caltech256/train", os.path.basename(cat_dirs[0])), exist_ok=True)
        os.makedirs(os.path.join(local_directory + "Caltech256/test", os.path.basename(cat_dirs[0])), exist_ok=True)
        num_test = int(len(glob.glob1(cat_dirs[0],"*.jpg"))/5)
        for fdir in cat_dirs:
          test_added = 0 
          for fimg in sorted(glob.glob(os.path.join(fdir, '*.jpg'))):
            if test_added < num_test : 
              copyfile(fimg, os.path.join(local_directory + "Caltech256/test", os.path.basename(fdir),os.path.basename(fimg)))
              test_added = test_added + 1
            else :
              copyfile(fimg, os.path.join(local_directory + "/Caltech256/train", os.path.basename(fdir),os.path.basename(fimg)))
    else:
      print("caltech256 has been loaded!!!*!")
    
    if aug:
      transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
    else:
      transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
    # Load the Data
    if train:
      path = local_directory + 'Caltech256/train'
    else:
      path = local_directory + 'Caltech256/test'
    
    dataset = torchvision.datasets.ImageFolder(path, transform=transform)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

  elif name == 'cub':
    path = local_directory + 'CUB_200_2011'
    if os.path.isdir(path) == False:
      os.makedirs(local_directory, exist_ok=True)
      with tarfile.open(dataset_drive_directory + 'CUB_200_2011.tgz') as tar_ref:
        tar_ref.extractall(local_directory)
      import glob
      from shutil import copyfile
      tmp = pd.read_csv(os.path.join(local_directory + 'CUB_200_2011', 'train_test_split.txt'), sep=' ', names=['img_id', 'is_training_img'])
      tmp2 = pd.read_csv(os.path.join(local_directory + 'CUB_200_2011', 'images.txt'), sep=' ', names=['img_id', 'filepath'])
      all = tmp2.merge(tmp, on='img_id')
      paths = all.filepath
      for cat in range(1, 200):
        cat_dirs = glob.glob(os.path.join(local_directory + "CUB_200_2011/images/", '%03d*' % cat))
        os.makedirs(os.path.join(local_directory + "CUB/train", os.path.basename(cat_dirs[0])), exist_ok=True)
        os.makedirs(os.path.join(local_directory + "CUB/test", os.path.basename(cat_dirs[0])), exist_ok=True)
        for fdir in cat_dirs:
          for fimg in glob.glob(os.path.join(fdir, '*.jpg')):
            filepath=os.path.join(os.path.basename(os.path.basename(fdir)),os.path.basename(fimg))
            img_index = paths[paths == filepath].index[0]
            if all.iloc[img_index]['is_training_img'] == 0 : 
              copyfile(fimg, os.path.join(local_directory + "CUB/test", os.path.basename(fdir),os.path.basename(fimg)))
            else :
              copyfile(fimg, os.path.join(local_directory + "CUB/train", os.path.basename(fdir),os.path.basename(fimg)))
    else:
      print("CUB_200_2011 has been loaded!!!*!")

    if aug:
      transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
    else:
      transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
    # Load the Data
    if train:
      path = local_directory + 'CUB/train'
    else:
      path = local_directory + 'CUB/test'

    dataset = torchvision.datasets.ImageFolder(path, transform=transform)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

  elif name == 'imagenet':
    path = local_directory + 'imagenet/test'
    if os.path.isdir(path) == False:
      os.makedirs(local_directory + "imagenet", exist_ok=True)
      with tarfile.open(dataset_drive_directory + 'ILSVRC2012_img_test_v10102019.tar') as tar_ref:
        tar_ref.extractall(local_directory + 'imagenet')
    else:
      print("Imagenet has been loaded!!!")
    transform = transforms.Compose([
            transforms.Resize(size=256),
            transforms.CenterCrop(size=224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
        ])
    path = local_directory + 'imagenet'
    dataset = torchvision.datasets.ImageFolder(path, transform=transform)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
  else:
    print('Dataset', name, 'is not supported!!!*!')
    return None, None

  return dataset, loader

class dataset_class:
  def __init__(self, name, clean= False, valid = False, shuffle = False, aug = False, train_bs = 128, valid_bs = 100, local_directory = None, dataset_drive_directory=None):
    if name is not None:
      self.trainset, self.trainloader = get_dataset(name, train=True, shuffle=shuffle, aug=aug, batch_size = train_bs, local_directory = local_directory, dataset_drive_directory=dataset_drive_directory)
    else:
      print("Empty dataset!")
      self.trainset, self.trainloader = None, None
    self.trainset_clean, self.trainloader_clean = None, None
    self.validset, self.validloader = None, None
    if clean:
      self.trainset_clean, self.trainloader_clean = get_dataset(name, train=True, shuffle=False, aug=False, batch_size = valid_bs, local_directory = local_directory, dataset_drive_directory=dataset_drive_directory) 
    if valid:
      self.validset, self.validloader = get_dataset(name, train=False, shuffle=False, aug=False, batch_size = valid_bs, local_directory = local_directory, dataset_drive_directory=dataset_drive_directory)
