from torch.utils.data import Dataset, DataLoader, TensorDataset, Subset
from hoda_utils import SoftLabelLoss, imshow, get_model
from hoda_training import model_class
import os
import time
import math
import torch
import numpy as np
ver = 1
n_digits = 2
class knockoff_attack:
  def __init__(self, target_model, budget = 50000, num_classes = 10, classes = None, device = 'cuda'):
    self.bb_model = target_model
    self.num_classes = num_classes
    self.budget = budget
    self.device = device
    self.classes = classes 
    self.soft_loss = SoftLabelLoss()
    self.softmax = torch.nn.Softmax(dim=1)

  def make_one_hot(self,labels, K):
    return torch.zeros(labels.shape[0], K, device=labels.device).scatter(1, labels.unsqueeze(1), 1)

  def select_rand_data(self,dataset,trg_ds_name,att_ds_name,adv_access='prob',imagenet_flag = False):
      if not os.path.isdir('knockoff'):
        os.mkdir('knockoff')
      path = './knockoff/' + trg_ds_name + '_'+ att_ds_name +'_'+str(self.budget) + '_rand_index_set'
      print('==> check for seed', path)
      assert os.path.isdir('knockoff'), 'Error: no checkpoint directory found!'
      if os.path.isfile(path + '.npz'):
        print(path + '.npz is loading!')
        index_file = np.load(path + '.npz')
        rand_index_set = index_file['rand_index_set']
      else:
        print("len_dataset:",len(dataset.trainset))
        rand_index_set = np.arange(len(dataset.trainset))
        np.random.shuffle(rand_index_set)
        path = './knockoff/' + trg_ds_name + '_'+ att_ds_name +'_'+str(self.budget) + '_rand_index_set'
        np.savez(path + '.npz',rand_index_set=rand_index_set)
        print(path + '.npz','is saved')
      if imagenet_flag == False: 
        new_dataset = torch.utils.data.Subset(dataset.trainset,rand_index_set[:self.budget])
        new_loader = torch.utils.data.DataLoader(new_dataset, batch_size=self.budget, shuffle=False)
        X_selected,Y = next(iter(new_loader))
        X_selected = X_selected.to(self.device)
        new_loader = torch.utils.data.DataLoader(new_dataset, batch_size=128, shuffle=False)
        Y_hat = []
        with torch.no_grad():
          for X,Y in new_loader:
            X, Y = X.to(self.device), Y.to(self.device)
            Y_hat_b = self.softmax(self.bb_model(X))
            Y_hat.extend(Y_hat_b.cpu())
          Y_hat = torch.stack(Y_hat)
        if adv_access == 'label':
          labels = torch.argmax(Y_hat, dim=1)
          Y_hat = self.make_one_hot(labels, self.num_classes)
        print(X_selected.shape,Y_hat.shape,X_selected.type(),Y_hat.type())
        transfer_set = TensorDataset(X_selected.detach().cpu(), Y_hat.detach().cpu())
        return transfer_set
      else:
        print("Imagenet like dataset")
        new_loader = torch.utils.data.DataLoader(dataset.trainset, batch_size=128, shuffle=False)
        Y_hat = []
        counter = 0
        with torch.no_grad():
          for X,Y in new_loader:
            if counter % 10 == 0:
              print(counter)
            X, Y = X.to(self.device), Y.to(self.device)
            Y_hat_b = self.softmax(self.bb_model(X))
            Y_hat.extend(Y_hat_b.cpu())
            counter += 1
          Y_hat = torch.stack(Y_hat)
        if adv_access == 'label':
          labels = torch.argmax(Y_hat, dim=1)
          Y_hat = self.make_one_hot(labels, self.num_classes)
        print("labled")
        dataset.trainset.samples = [(d,l)for (d, s), l in zip(dataset.trainset.samples,Y_hat)]
        print(Y_hat.shape,Y_hat.type())
        return torch.utils.data.Subset(dataset.trainset,rand_index_set[:self.budget])
		

from torch.utils.data import Dataset, DataLoader, TensorDataset, Subset
ver = 1
n_digits = 2
class jb_based_attacks:
  def __init__(self, target_model, attack_data, aux_classifier_name, trg_ds_name, budget = 50000, num_classes = 10, classes = None, device = 'cuda'):
    self.bb_model = target_model
    self.num_classes = num_classes
    self.aux_classifier_name = aux_classifier_name
    self.budget = budget
    self.device = device
    self.attack_data = attack_data
    self.classes = classes 
    self.soft_loss = SoftLabelLoss()
    self.softmax = torch.nn.Softmax(dim=1)
    self.trg_ds_name = trg_ds_name

  def make_one_hot(self,labels, K):
    return torch.zeros(labels.shape[0], K, device=labels.device).scatter(1, labels.unsqueeze(1), 1)

  def select_seed_data(self,dataset,num_seed_data,adv_access='prob'):
      print("Selecting",self.budget ,"seed data with access:",adv_access)
      self.num_sample_each_class = int(num_seed_data / self.num_classes)
      print("Num of samples in each class:",self.num_sample_each_class)
      if not os.path.isdir('jb'):
        os.mkdir('jb')
      path = './jb/' + self.trg_ds_name + '_' + str(self.num_sample_each_class) + 'seed_ind'
      print('==> check for seed', path)
      assert os.path.isdir('jb'), 'Error: no checkpoint directory found!'
      if os.path.isfile(path + '.npz'):
        print(path + '.npz is loading!')
        index_file = np.load(path + '.npz')
        seed_index_set = index_file['seed_index_set']
      else:
        seed_index_set = []
        valid_index_set = []
        for i in range(self.num_classes):
          rand_ind = np.where(np.array(dataset.validset.targets) == i)[0]
          np.random.shuffle(rand_ind)
          print(len(rand_ind[:self.num_sample_each_class]),len(rand_ind[self.num_sample_each_class:]))
          seed_index_set.extend(list(rand_ind[:self.num_sample_each_class]))
          valid_index_set.extend(list(rand_ind[self.num_sample_each_class:]))
        print(len(seed_index_set))
        print(len(valid_index_set))
        seed_index_set.extend(valid_index_set)
        print(seed_index_set)
        path = './jb/' + self.trg_ds_name + '_' + str(self.num_sample_each_class) + 'seed_ind'
        np.savez(path + '.npz',seed_index_set=seed_index_set)
        print(path + '.npz','is saved')
      print(len(seed_index_set),seed_index_set)
      valid_dataset = torch.utils.data.Subset(dataset.validset,seed_index_set[num_seed_data:])
      seed_dataset = torch.utils.data.Subset(dataset.validset,seed_index_set[:num_seed_data])
      seed_loader = torch.utils.data.DataLoader(seed_dataset, batch_size=num_seed_data, shuffle=False)
      X,Y = next(iter(seed_loader))
      X, Y = X.to(self.device), Y.to(self.device)
      Y_hat = self.softmax(self.bb_model(X))
      if adv_access == 'label':
        labels = torch.argmax(Y_hat, dim=1)
        Y_hat = self.make_one_hot(labels, self.num_classes)
      print(X.shape,Y_hat.shape,X.type(),Y_hat.type())
      print(Y_hat[:2].detach().cpu().numpy())
      seed_dataset = TensorDataset(X.detach().cpu(), Y_hat.detach().cpu())
      return seed_dataset,valid_dataset


  
  def jbda(self,dataloader,aux_classifier,eps,range=[0,1],adv_access='prob'): #https://github.com/tribhuvanesh/prediction-poisoning/blob/1ddff8bb4ac2a5903aded6dde43dcd94207c451f/defenses/adversary/jacobian.py#L302
    print("adv_access:",adv_access)
    aug_X = []
    aug_Y = []
    bad_count = 0
    for i, (X, Y) in enumerate(dataloader):
      if True: #with torch.enable_grad():
          X, Y = X.to(self.device), Y.to(self.device)
          X.requires_grad = True
          Y_hat = aux_classifier(X)
          if len(list(Y.shape)) == 1:
            Y = self.make_one_hot(Y, self.num_classes)
          loss = self.soft_loss(Y_hat, Y).to(self.device)
          loss.backward()

          delta = eps * X.grad.detach().sign().to(self.device)
          X_new = torch.clip(X + delta,0,1)
          assert torch.max(X_new) <= 1 and torch.min(X_new) >= 0, 'Clip Error in Aug Data'
          Y_new = self.softmax(self.bb_model(X_new))
          if adv_access == 'label':
            labels = torch.argmax(Y_new, dim=1)
            Y_new = self.make_one_hot(labels, self.num_classes)

          aug_X.extend(X_new.detach())
          aug_Y.extend(Y_new.detach())
    print( "len aug data", len(aug_X), len(aug_Y) )
    return aug_X,aug_Y

  
  def pgd(self,aux_classifier, inputs, y_targ, epsilon, alpha, num_iter=5): #https://github.com/tribhuvanesh/prediction-poisoning/blob/1ddff8bb4ac2a5903aded6dde43dcd94207c451f/defenses/adversary/jacobian.py#L322
    """ Construct targeted adversarial examples on the examples X"""
    if epsilon == 0:
      return torch.zeros_like(inputs)
    y_targ = self.make_one_hot(torch.tensor(np.array([y_targ])), self.num_classes).to(self.device)
    with torch.enable_grad():
      inputs = inputs.to(self.device)
      delta = torch.zeros_like(inputs, requires_grad=True).to(self.device)
      for t in range(num_iter):
        y_hat = aux_classifier(inputs + delta)
        loss = self.soft_loss(y_hat, y_targ).to(self.device)
        loss.backward()
        delta.data = (delta - alpha * delta.grad.detach().sign()).clamp(-epsilon, epsilon)
        delta.grad.zero_()
      
      return delta.detach()



  def jbrand(self,dataloader,aux_classifier,eps,range=[0,1],adv_access='prob'): 
    print("adv_access:",adv_access)
    aug_X = []
    aug_Y = []
    for i, (X, Y) in enumerate(dataloader):
      if i % 200 == 0:
        print(i)
      with torch.enable_grad():
          assert X.shape[0] == Y.shape[0] == 1, 'Only supports batch_size = 1'
          X, Y = X.to(self.device), Y.to(self.device)
          rand_target = np.arange(self.num_classes)
          rand_target = rand_target[rand_target != Y[0].argmax().cpu().numpy()]  # Remove gt class
          rand_target = torch.tensor(np.random.choice(rand_target, self.randk, replace=False))

          for target in rand_target:
            delta = self.pgd(aux_classifier, X, target, epsilon=eps, alpha=eps/self.randk)
            X_new = torch.clamp(X + delta,0,1)
            assert torch.max(X_new) <= 1 and torch.min(X_new) >= 0, 'Clip Error in Aug Data'
            Y_new = self.softmax(self.bb_model(X_new))
            if adv_access == 'label':
              labels = torch.argmax(Y_new, dim=1)
              Y_new = self.make_one_hot(labels, self.num_classes)
            aug_X.extend(X_new.detach())
            aug_Y.extend(Y_new.detach())
    return aug_X,aug_Y


  def concat_two_dataset(self,ds,aug_x,aug_y):
    new_x = torch.cat((ds.tensors[0].to(self.device),torch.stack(aug_x)),dim=0)
    new_y = torch.cat((ds.tensors[1].to(self.device),torch.stack(aug_y)),dim=0)
    print(new_x.shape,new_y.shape,new_x.type(),new_y.type())
    new_dataset = TensorDataset(new_x, new_y)
    return new_dataset

#eps jbda -> 0.1
#eps jbrand3 -> 64/255
  def jb_attack(self,attack_type='jbda',eps=None,num_seed_data = 500, aux_train_epochs=20,data_aug_round=100,tau = 3,kapa = 0,batch_size=128,adv_access= 'prob'): #https://github.com/cleverhans-lab/cleverhans/blob/3ee7ffa5af4f1de324dede2d26aa9f6b9b59f438/cleverhans_v3.1.0/cleverhans_tutorials/mnist_blackbox.py
    ds, self.valid_ds = self.select_seed_data(self.attack_data,num_seed_data,adv_access=adv_access)
    num_sample_in_ds = num_seed_data
    self.rho = 0
    self.tau = tau
    self.kapa = kapa
    self.eps = eps
    self.batch_size = batch_size
    self.loader_list = [torch.utils.data.DataLoader(self.valid_ds, batch_size=self.batch_size, shuffle=False)]
    self.loader_name = ['hoda_' + self.aux_classifier_name + '_' + self.trg_ds_name + '_valid_info_'+attack_type]

    aug_dataset_name = "target_dataset_"+self.trg_ds_name+"_attack_"+attack_type+"_target_model_"+ self.aux_classifier_name + '_dataset_' + str(self.budget) +'_'+ adv_access
    if not os.path.isdir('jb'):
      os.mkdir('jb')
    dataset_path = './jb/' + aug_dataset_name
    print('==> check for dataset', dataset_path)
    assert os.path.isdir('jb'), 'Error: no checkpoint directory found!'
    if os.path.isfile(dataset_path):
      print(dataset_path + ' is loading!')
      loaded = torch.load(dataset_path)
      ds = TensorDataset(loaded['x'], loaded['y'])
      return ds
    else:
      print("Dataset does not exist!!!")

    if attack_type == 'jbda':
      print('Attack: JBDA')
      self.aug_factor = 2
    elif 'jbrand' in attack_type:
      # extract k from "jbrand<k>"
      self.randk = int(attack_type.replace('jbrand', ''))
      print("Attack: JBRAND",self.randk)
      self.aug_factor = self.randk + 1
      self.batch_size = 1
    else:
      print("Attack",attack_type,'is not supported')
      assert False
      
    

    while num_sample_in_ds < self.budget - self.aug_factor  and self.rho < data_aug_round:
      print("***************************** NEW AUG ROUND*************************************")
      if self.kapa > 0:
        num_sample = self.kapa
      else:
        num_sample = num_sample_in_ds
      if num_sample_in_ds + (num_sample * (self.aug_factor -1)) > self.budget:
        num_sample = self.budget - num_sample_in_ds
        num_sample  = int(num_sample / (self.aug_factor -1))
      print("num_sample_in_ds",num_sample_in_ds,"num_src_sample",num_sample,"rho",self.rho)

      if num_sample != num_sample_in_ds:
        print("Sample",num_sample,'data from',len(ds),'data')
        sample_index = np.arange(len(ds))
        np.random.shuffle(sample_index)
        selected_ds = torch.utils.data.Subset(ds,sample_index[:num_sample])
      else:
        selected_ds = ds

      ds_loader = torch.utils.data.DataLoader(ds, batch_size=128, shuffle=True)
      file_name = 'dummy_model'
      print("Num samples in aux classifier training set",num_sample_in_ds)
      self.aux_classifier = get_model(target_dataset=self.trg_ds_name,model_name=self.aux_classifier_name)
      self.aux_model = model_class(self.aux_classifier,num_classes=len(self.classes),classes=self.classes)
      self.aux_classifier, inf = self.aux_model.train(ds_loader,loader_list=self.loader_list,loader_name=self.loader_name,lr=0.01,epoch=aux_train_epochs,save_model=False,save_model_each_epoch=False,file_name=file_name,return_learning_index=False,soft_label=True,resume=None)
      self.aux_classifier.eval()
      
      start = time.time()
      print("Aug_data_number",num_sample)

      selected_ds_loader = torch.utils.data.DataLoader(selected_ds, batch_size=self.batch_size, shuffle=False)
      if attack_type == 'jbda':
        eps_coef = 2 * int(int(self.rho / self.tau) % 2 == 0) - 1
        print("jbda - eps_coef: ",eps_coef,"eps", self.eps)
        aug_X,aug_Y = self.jbda(selected_ds_loader,self.aux_classifier,eps_coef * self.eps,adv_access=adv_access)
      elif 'jbrand' in attack_type:
        eps_coef = 1 #2 * int(int(self.rho / self.tau) % 2 == 0) - 1
        print("jbrand - eps_coef: ",eps_coef,"eps", self.eps)
        aug_X,aug_Y = self.jbrand(selected_ds_loader,self.aux_classifier,eps_coef * self.eps,adv_access=adv_access)

      end = time.time()
      print("Elapsed time:",end - start)

      ds = self.concat_two_dataset(ds,aug_X,aug_Y)
      num_sample_in_ds += len(aug_X)
      self.rho += 1
    print(dataset_path,"is saved!")
    db = {'x': ds.tensors[0], 'y': ds.tensors[1]}
    torch.save(db, dataset_path)
    return ds
