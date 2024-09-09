import time
import os
import shutil
import time
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn. functional as F
import torch.backends.cudnn as cudnn
import argparse
import sys
from autoattack import AutoAttack

sys.path.append('../utils_pseudoLab/')
from PIL import Image
import torchvision
from torchvision import models, transforms
from torchvision.utils import save_image
from torchvision import datasets, transforms
import torch.utils.data as data
from ssl_networks import CNN as MT_Net
from wideArchitectures import WRN28_5_wn
from train import *

class PGD_L2():
    def __init__(self, model, epsilon=20/255, step_size=4/255, num_steps=20, random_start=True, target_mode= False, criterion='ce', bn_mode='eval', train=True):

        self.model = model
        self.epsilon = epsilon
        self.step_size = step_size
        self.num_steps = num_steps
        self.random_start = random_start
        self.target_mode= target_mode
        self.bn_mode = bn_mode
        self.train = train
        self.criterion = criterion
        self.criterion_ce = nn.CrossEntropyLoss()
        self.criterion_kl = nn.KLDivLoss(reduction='sum')

    def perturb(self, x_nat, targets):
        
        if self.bn_mode == 'eval':
            self.model.eval()
        
        if self.random_start:
            x_adv = x_nat.detach() + torch.empty_like(x_nat).uniform_(-self.epsilon, self.epsilon).cuda().detach()
            x_adv = torch.clamp(x_adv, min=0, max=1)
        else:
            x_adv = x_nat.clone().detach()

        for _ in range(self.num_steps):
            
            x_adv.requires_grad_()
            outputs = self.model(x_adv)
            self.model.zero_grad()
            if self.criterion == "ce":
                loss = self.criterion_ce(outputs, targets)
                loss.backward()
                grad = x_adv.grad
            elif self.criterion == "kl":
                loss = self.criterion_kl(F.log_softmax(outputs, dim=1), F.softmax(self.model(x_nat), dim = 1))
                grad = torch.autograd.grad(loss, [x_adv])[0]
            elif self.criterion == "revkl":
                loss = self.criterion_kl(F.log_softmax(self.model(x_nat), dim=1), F.softmax(outputs, dim = 1))
                grad = torch.autograd.grad(loss, [x_adv])[0]
                
            grad_norm = grad.abs().pow(2).view(x_nat.shape[0], -1).sum(1).pow(1./2)
            grad_norm = grad_norm.view(x_nat.shape[0], 1, 1, 1).expand_as(x_nat)
            d_adv = grad/grad_norm
            
            if self.target_mode:
                x_adv= x_adv - self.step_size * d_adv
            else:
                x_adv= x_adv + self.step_size * d_adv
            
            d_adv = (x_adv - x_nat).view(x_nat.shape[0], -1).detach()
            d_adv = d_adv.view(x_nat.shape)
            d_adv = torch.clamp(d_adv, min=-self.epsilon, max=self.epsilon)
            
            x_adv = torch.clamp(x_nat + d_adv, min=0, max=1).detach()
            
        if self.train:
            self.model.train()
        
        return x_adv, d_adv


class PGD_Linf():

    def __init__(self, model, epsilon=8*4/255, step_size=4/255, num_steps=10, random_start=True, target_mode= False, criterion= 'ce', bn_mode='eval', train=True, vat=False):

        self.model = model
        self.epsilon = epsilon
        self.step_size = step_size
        self.num_steps = num_steps
        self.random_start = random_start
        self.target_mode = target_mode
        self.bn_mode = bn_mode
        self.train = train
        self.criterion = criterion
        self.criterion_ce = nn.CrossEntropyLoss()
        self.criterion_kl = nn.KLDivLoss(reduction='sum')
        self.vat = vat

    def perturb(self, x_nat, targets=None):
        if self.bn_mode == 'eval':
            self.model.eval()
            
        if self.random_start:
            x_adv = x_nat.detach() + torch.empty_like(x_nat).uniform_(-self.epsilon, self.epsilon).cuda().detach()
            x_adv = torch.clamp(x_adv, min=0, max=1)
        else:
            x_adv = x_nat.clone().detach()

        for _ in range(self.num_steps):
            x_adv.requires_grad_()
            outputs = self.model(x_adv)
            #self.model.zero_grad()
            if self.criterion == "ce":
                loss = self.criterion_ce(outputs, targets)
                loss.backward()
                grad = x_adv.grad
            elif self.criterion == "kl":
                if self.vat:
                    loss = self.criterion_kl(F.log_softmax(outputs, dim=1), F.softmax(self.model(x_nat).detach(), dim = 1))
                    grad = torch.autograd.grad(loss, [x_adv])[0]
                else:
                    loss = self.criterion_kl(F.log_softmax(outputs, dim=1), F.softmax(self.model(x_nat), dim = 1))
                    grad = torch.autograd.grad(loss, [x_adv])[0]
            elif self.criterion == "revkl":
                loss = self.criterion_kl(F.log_softmax(self.model(x_nat), dim=1), F.softmax(outputs, dim = 1))
                grad = torch.autograd.grad(loss, [x_adv])[0]
            elif self.criterion == "js":
                nat_probs = F.softmax(self.model(x_nat), dim=1)
                adv_probs = F.softmax(outputs, dim=1)
                mean_probs = (nat_probs + adv_probs)/2
                loss =  (self.criterion_kl(mean_probs.log(), nat_probs) + self.criterion_kl(mean_probs.log(), adv_probs))/2
                grad = torch.autograd.grad(loss, [x_adv])[0]
            if self.target_mode:
                x_adv = x_adv - self.step_size * grad.sign()
            else:
                x_adv = x_adv + self.step_size * grad.sign()
            
            x_adv = torch.min(torch.max(x_adv, x_nat - self.epsilon), x_nat + self.epsilon)
            x_adv = torch.clamp(x_adv, min=0, max=1).detach()
            d_adv = torch.clamp(x_adv - x_nat, min=-self.epsilon, max=self.epsilon).detach()
            
        if self.train:
            self.model.train()
        
        
        return x_adv, d_adv



def validate(valloader, model, criterion, use_cuda, mode, pgd_attack=None, autoattack=None):

    model.eval()
    loss_per_batch = []
    total_correct = 0
    total_samples = 0

    for batch_idx, (inputs, targets) in enumerate(valloader):

        if use_cuda:
            inputs, targets = inputs.to(device), targets.to(device)
            inputs.requires_grad = True

        if not autoattack and pgd_attack:
            adv_inputs, _ = pgd_attack.perturb(inputs, targets)
            outputs = model(adv_inputs)
        elif not autoattack and not pgd_attack:
            outputs = model(inputs)
        elif autoattack and not pgd_attack:
            adv_inputs = autoattack.run_standard_evaluation(inputs, targets, bs=128)
            outputs = model(adv_inputs)
        else:
            raise ValueError("You should select one method.")

        loss = criterion(outputs, targets)
        loss_per_batch.append(loss.item())

        _, pred = outputs.max(1)
        correct = pred.eq(targets).sum().item()
        total_correct += correct
        total_samples += inputs.size(0)

    accuracy = 100.0 * total_correct / total_samples
    loss_per_epoch = [np.average(loss_per_batch)]
    acc_val_per_epoch = [accuracy]

    return loss_per_epoch, acc_val_per_epoch


def test(model, test_loader, device):
    criterion = nn.CrossEntropyLoss()
    #test_attack = PGD_L2(model=model, epsilon=8/255, step_size=(8)/255, num_steps=20, random_start='store_false', train=False)
    test_attack = PGD_Linf(model=model, epsilon=8/255, step_size=(8/4)/255, num_steps=20, random_start='store_false',            criterion='ce',bn_mode = 'eval', train = False)
    #test_attack = FGSM(model=model, epsilon=8/255)
    # auto_attack = AutoAttack(model, norm='Linf', eps=8/255, version='standard', verbose=False)
    # auto_attack.attacks_to_run = ['apgd-ce', 'apgd-t', 'fab', 'square']
    model.to(device)
    _, pgd_test_acc = validate(test_loader, model, criterion, use_cuda = True, mode='PGD_attack', pgd_attack=test_attack)
    # _, aa_test_acc  = validate(test_loader, model, criterion, use_cuda = True, mode='Autoattack', pgd_attack=None,                         autoattack=auto_attack)
    print(pgd_test_acc)
    # print(aa_test_acc)


# mean = [0.4914, 0.4822, 0.4465]
# std = [0.2023, 0.1994, 0.2010]

transform_test = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize(mean, std),
    ])

testset = datasets.CIFAR10(root='./data', train=False, download=False, transform=transform_test)
test_loader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False, num_workers=os.cpu_count(), pin_memory=True)




# model = WRN28_5_wn(num_classes = 10, dropout = 0.0)
model = MT_Net(num_classes = 10, dropRatio = 0.0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


path = 'ssl_models_M_SOTA_CIFAR10/1000/trades_attack_wrn_seed_38/last_20_epochs_seed_38_labels_1000/epoch_100_RobAcc_0.58510_NatAcc_0.79520_labels_1000_bestValLoss_0.79780.pth'


checkpoint = torch.load(path)
checkpoint = {k.replace('module.', ''): v for k, v in checkpoint.items()}
# print("Keys in the loaded checkpoint:", checkpoint.keys())
print("Path loaded: ", path)
model.load_state_dict(checkpoint)

test(model, test_loader, device)











