from typing import Optional
import torch
import torch.nn.functional as F
import torch.nn as nn
import os
import numpy as np
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import importlib
import copy
import argparse
from torchvision import transforms, datasets
from matplotlib import pyplot as plt
import torch.nn.functional as F
from torch.autograd import Variable, grad
from numpy.linalg import eig as eig
from torch.distributions.multivariate_normal import MultivariateNormal
import backpack
from backpack import backpack, extend

import time

def create_path(model_name, args, num_true, num_random, dataset):

    path = model_name
    for item in args:
        path += "_" + str(item)

    path += "_" + str(num_true) + "_" + str(num_random) + "_" + dataset

    path = os.path.join("./", path, "")

    return path


def mkdir(path):

    isfolder = os.path.exists(path)
    if not isfolder:
        os.makedirs(path)

parser = argparse.ArgumentParser()

parser.add_argument("--gpu_device", type=str,
                    default='cuda:0',
                    help="gpu device")

parser.add_argument("--num_neurons", type=int,
                    default=600,
                    help="number of neurons of fc net")

parser.add_argument("--num_layers", type=int,
                    default=2,
                    help="Number of layers of fc net")

def split(dataset, n1, n2):

    data1 = dataset.data[:n1]
    targets1 = dataset.targets[:n1]
    data2 = dataset.data[-n2:]
    targets2 = dataset.targets[-n2:]

    return data1, targets1, data2, targets2

#use the first available gpu else use cpu
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
kwargs = {'num_workers': 1, 'pin_memory': True} if device == 'cuda' else {}

num_true = 55000
num_prior = 5000
num_random = 0
num_approx = 10000
num_classes = 10
dataset = "mnist"
num_inplanes = 1
transform = transforms.Compose([transforms.ToTensor()])

train_set = datasets.MNIST('./data',
                    train=True,
                    download=True,
                    transform=transform)

train_set_approx = datasets.MNIST('./data',
                    train=True,
                    download=True,
                    transform=transform)


train_set_prior = datasets.MNIST('./data',
                    train=True,
                    download=True,
                    transform=transform)

test_set = datasets.MNIST('./data',
                    train=False,
                    download=True,
                    transform=transform)



arg = parser.parse_args()


num_neurons = arg.num_neurons
num_layers = arg.num_layers

model_name = "fc"
args = (num_classes, num_layers, num_neurons)
path = create_path(model_name, args, num_true, num_random, dataset)
print(path)
mkdir(path)

data1, targets1, data2, targets2 = split(train_set, num_true, num_prior)

data3, targets3, _, _ = split(train_set, num_approx, 0)

train_set.data = data1
train_set.targets = targets1    
train_set_prior.data = data2
train_set_prior.targets = targets2
train_set_approx.data = data3
train_set_approx.targets = targets3

train_loader = torch.utils.data.DataLoader(train_set,
                                          batch_size=500,
                                          shuffle = True,
                                          **kwargs)

train_loader_approx = torch.utils.data.DataLoader(train_set_approx,
                                          batch_size=len(train_set_approx.data),
                                          shuffle = True,
                                          **kwargs)

train_loader_FIM = torch.utils.data.DataLoader(train_set_approx,
                                          batch_size=1,
                                          shuffle = True,
                                          **kwargs)

train_loader_prior = torch.utils.data.DataLoader(train_set_prior,
                                          batch_size=len(train_set_prior.data),
                                          shuffle = True,
                                          **kwargs)

test_loader = torch.utils.data.DataLoader(test_set,
                                         batch_size=500,
                                         shuffle = True,
                                         **kwargs)


## fully connected layers class
class fcn(nn.Module):
    def __init__(self, num_classes, num_layers, num_neurons):
        super(fcn, self).__init__()
        layers = []
        layers.append(nn.Linear(28*28, num_neurons))
        layers.append(nn.ReLU())
        for i in range(num_layers-1):
            layers.append(nn.Linear(num_neurons, num_neurons))
            layers.append(nn.ReLU())

        layers.append(nn.Linear(num_neurons, num_classes))
        self.classifier = nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
    
def get_names_params(mod):
    '''
    mod: model with nn.Parameters, cannot use functionalized model
    return:
        names_all: a list of all names of mod.paramters, [a1.b1.c1, a2.b2.c2, ...]
        orig_params: tuple of parameters of type nn.Parameter
    '''

    orig_params = tuple(mod.parameters())
    names_all = []
    for name, p in list(mod.named_parameters()):
        names_all.append(name)
    return orig_params, names_all

def del_attr(obj, names):
    '''
    names: one name in the list names_all, a.b.c, splited by ".", list of format names = [a,b,c]
    
    delete the attribute obj.a.b.c
    '''
    if len(names) == 1:
        delattr(obj, names[0])
    else:
        del_attr(getattr(obj, names[0]), names[1:])



def set_attr(obj, names, val):

    '''
    names: one name in the list names_all, a.b.c, splited by ".", list of format names = [a,b,c]
    
    set the attribute obj.a.b.c to val

    if obj.a.b.c is nn.Parameter, cannot directly use set_attr, need to first use del_attr
    '''
    if len(names) == 1:
        setattr(obj, names[0], val)
    else:
        set_attr(getattr(obj, names[0]), names[1:], val)



##Bayesian neural networks class
class bnn(nn.Module):
    def __init__(self, c, args, ns=150):
        super().__init__()
        self.w = c(*args).to(device)
        self.mu_std = nn.ModuleList([c(*args).to(device), c(*args).to(device)])
        self.ns, self.args = ns, args
        orig_params_w, names_all_w = get_names_params(self.w)
        self.names_all_w = names_all_w
        self.c = c



        

    def forward(self, x):
        ys = []
        for _ in range(self.ns):
            for name, m, v in zip(self.names_all_w, list(self.mu_std[0].parameters()), list(self.mu_std[1].parameters())):

                
                r = torch.randn_like(m).mul(torch.sqrt(torch.exp(2*v)))
                del_attr(self.w, name.split("."))
                set_attr(self.w, name.split("."), r+m)  
                


            y = self.w(x)
            ys.append(y)

        self.w = self.c(*args).to(device)
        return torch.stack(ys)
    
def sec(model, model_init, rho, num_samples, device, b = 100, c = 0.1, delta = 0.025):

    epsilon = torch.exp(2*rho)
    pi = torch.tensor(np.pi)
    kl_1, kl_2 = 0, 0

    test = 0

    for m0, m, xi in zip(
        model_init.parameters(), 
        model.mu_std[0].parameters(), 
        model.mu_std[1].parameters(), 
    ):

        # print(m0-m)

        q = torch.exp(2*xi)
        p = epsilon
        kl_1 += (1/p) * torch.sum((m0-m)**2)
        kl_2 += torch.sum(q / p) + torch.sum(torch.log(p / q))
        kl_2 += -m.numel()


    kl = (kl_1 + kl_2) / 2
    penalty = 2*torch.log(2*torch.abs(b*torch.log(c / epsilon))) + torch.log(pi**2*num_samples / 6*delta)


    # penalty = torch.tensor(0)
    
    sec = torch.sqrt((kl + penalty) / (2*(num_samples - 1)))

    return sec, kl, kl_1, kl_2, penalty


def train(model,model_init, num_samples, device, train_loader, criterion, optimizer, rho, num_classes):

    model.train()
    for (data, targets) in train_loader:
        # print(len(data))

        loss2, kl, kl_1, kl_2, penalty = sec(model, model_init, rho ,num_samples, device)
        

        data, targets = data.to(device), targets.to(device)
        output = model(data)
        output = output.reshape(model.ns * len(data), num_classes)
        targets = targets.repeat(model.ns)
        loss = criterion(output, targets) * (1/np.log(2))

        optimizer.zero_grad()
        (loss2 + loss).backward()

        optimizer.step()

    print("loss2, kl, kl1, kl2, p", loss2.item(), kl.item(), kl_1.item(), kl_2.item(), penalty.item())




def val(model, device, val_loader, criterion, num_classes):
    
    model.eval()
    sum_loss, sum_corr = 0, 0

    

    for (data, targets) in val_loader:
        data, targets = data.to(device), targets.to(device)
        output = model(data)
        output = output.reshape(model.ns * len(data), num_classes)
        targets = targets.repeat(model.ns)
        loss = criterion(output, targets)
        pred = output.max(1)[1]
        sum_loss += loss.item()
        sum_corr += pred.eq(targets).sum().item() / len(targets)

    err_avg = 1 - (sum_corr/len(val_loader))
    loss_avg = sum_loss / len(val_loader)
    


    return err_avg, loss_avg

def val_d(model, device, val_loader, criterion, num_classes):
    
    model.eval()
    sum_loss, sum_corr = 0, 0

    

    for (data, targets) in val_loader:
        data, targets = data.to(device), targets.to(device)
        output = model(data)
        loss = criterion(output, targets)
        pred = output.max(1)[1]
        sum_loss += loss.item()
        sum_corr += pred.eq(targets).sum().item() / len(targets)

    err_avg = 1 - (sum_corr/len(val_loader))
    loss_avg = sum_loss / len(val_loader)
    
    return err_avg, loss_avg






def initial1(model, model_trained):

    state_dict = model_trained.state_dict()
    model.w.load_state_dict(state_dict)
    model.mu_std[0].load_state_dict(state_dict)

    for v, w in zip(model.mu_std[1].parameters(), model_trained.parameters()):
        v.data = 0.5*torch.log(torch.abs(w) / 10)

def KLdiv(pbar,p):
    return pbar * np.log(pbar/p) + (1-pbar) * np.log((1-pbar)/(1-p))


def KLdiv_prime(pbar,p):
    return (1-pbar)/(1-p) - pbar/p


def Newt(p,q,c):
    newp = p - (KLdiv(q,p) - c)/KLdiv_prime(q,p)
    return newp


def approximate_BPAC_bound(train_accur, B_init, niter=5):

    '''
    train_accur : training accuracy
    B_init: the second term of pac-bayes bound
    return: approximated pac bayes bound using inverse of kl
    eg: err = approximate_BPAC_bound(0.9716, 0.2292)
    '''
    B_RE = 2* B_init **2
    A = 1-train_accur
    B_next = B_init+A
    if B_next>1.0:
        return 1.0
    for i in range(niter):
        B_next = Newt(B_next,A,B_RE)
    return B_next




def list_to_vec(param_list):

    '''
    transfer a iterable (can be tuple or list) of tensors to a tensor of shape (num_param, )
    
    gradient can pass through this operation.

    if param_list is leaf variable, p_vector is not a leaf variable
    '''

    cnt = 0
    for p in param_list:
        if cnt == 0:
            p_vector = p.contiguous().view(-1)
        else:
            p_vector = torch.cat([p_vector, p.contiguous().view(-1)])
        cnt += 1

    return p_vector


def main():

    c = fcn
    rho = torch.tensor(-3).to(device).float()
    model = bnn(c,args)

    
    model_trained = fcn(*args)
    model_trained.load_state_dict(torch.load("./fc_2_2_600_55000_0_mnist/model.pt", map_location='cpu'))



    model_trained = model_trained.to(device)
    model_init = fcn(*args)
    model_init.load_state_dict(torch.load("./fc_2_2_600_55000_0_mnist/model.pt", map_location='cpu'))
    model_init = model_init.to(device)
    

    num_params = sum(p.numel() for p in model_trained.parameters())
    print(num_params)
    

    initial1(model, model_trained)

    epochs = 200
    rho.requires_grad = True
    param = list(model.parameters()) + [rho]
    criterion = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(param, lr = 1e-3, weight_decay=0)

    dt = val_d(model_trained, device, train_loader, criterion, num_classes)
    bt = val(model, device, train_loader, criterion, num_classes)
    dv = val_d(model_trained, device, test_loader, criterion, num_classes)
    bv = val(model, device, test_loader, criterion, num_classes)

    print('deterministic train', dt)
    print('bayes train', bt)
    print('deterministic test', dv)
    print('bayes test', bv)

    loss2, kl, kl_1, kl_2, penalty = sec(model, model_init, rho, num_true, device)
    print("loss2, kl, kl1, kl2, p", loss2.item(), kl.item(), kl_1.item(), kl_2.item(), penalty.item())
    print("rho", rho.item())
    

    bd = approximate_BPAC_bound(1-bt[0], loss2.item())
    print("bd", bd)

    for epoch in range(epochs):
        if epoch >= 100:
            if epoch%5 == 0:
                for g in optimizer.param_groups:
                    g['lr'] = g['lr']*0.95


        time_start = time.time()

        train(model,model_init, num_true, device, train_loader, criterion, optimizer, rho, num_classes)
        time_end = time.time()

        if epoch%20 == 0:
            val_err, val_loss = val(model,device, test_loader, criterion, num_classes)
            train_err, train_loss = val(model,device, train_loader, criterion, num_classes)

            loss2, kl, kl_1, kl_2, penalty = sec(model, model_init, rho, num_true, device)
            bd1 = train_err + loss2
            bd2 = train_loss * (1/np.log(2)) + loss2



            print('epoch', epoch)
            print('train_err, train_loss', train_err, train_loss)
            print('val_err, val_loss', val_err, val_loss)
            print('bd1, bd2, rho', bd1.item(), bd2.item(), rho.item())
            print("loss2, kl, kl1, kl2, p", loss2.item(), kl.item(), kl_1.item(), kl_2.item(), penalty.item())
            for g in optimizer.param_groups:
                    print(g['lr'])
            print('time', time_end - time_start)

            if epoch != 0:
                torch.save((model.state_dict(), rho.item()), path + "model_bayes1.pt")

    dt = val_d(model_trained, device, train_loader, criterion, num_classes)
    bt = val(model, device, train_loader, criterion, num_classes)
    dv = val_d(model_trained, device, test_loader, criterion, num_classes)
    bv = val(model, device, test_loader, criterion, num_classes)

    print('deterministic train', dt)
    print('bayes train', bt)
    print('deterministic test', dv)
    print('bayes test', bv)

    loss2, kl, kl_1, kl_2, penalty = sec(model, model_init, rho, num_true, device)
    print("loss2, kl, kl1, kl2, p", loss2.item(), kl.item(), kl_1.item(), kl_2.item(), penalty.item())
    print("rho", rho.item())


    bd = approximate_BPAC_bound(1-bt[0], loss2.item())
    print("bd", bd)


    stat1 = dict({"dt": dt, "bt":bt, "dv":dv, "bv":bv, "bd":bd ,"loss2":loss2.item(), "kl":kl.item(), "kl_1":kl_1.item(), "kl_2":kl_2.item(), "rho":rho.item()})

    print(stat1)





main()









        
        
    
