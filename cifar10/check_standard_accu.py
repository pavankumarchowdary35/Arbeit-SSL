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

from wideArchitectures import WRN28_5_wn
from train import *

def calculate_standard_accuracy(test_loader, model, criterion, device):
    model.eval()  # Set the model to evaluation mode
    correct = 0
    total = 0
    model.to(device)
    with torch.no_grad():  # Disable gradient calculation during evaluation
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    standard_accuracy = 100 * correct / total
    return standard_accuracy



mean = [0.4914, 0.4822, 0.4465]
std = [0.2023, 0.1994, 0.2010]

transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

testset = datasets.CIFAR10(root='./data', train=False, download=False, transform=transform_test)
test_loader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False, num_workers=os.cpu_count(), pin_memory=True)


model = WRN28_5_wn(num_classes = 10, dropout = 0.0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


path = 'ssl_models_M_SOTA_CIFAR10/4000/trades_attack_wrn_seed_501/best_epoch_22_valLoss_1.05853_valAcc_64.99000_labels_4000_bestAccVal_64.99000.pth'
checkpoint = torch.load(path)
checkpoint = {k.replace('module.', ''): v for k, v in checkpoint.items()}
# print("Keys in the loaded checkpoint:", checkpoint.keys())
print("Path loaded: ", path)
model.load_state_dict(checkpoint)
criterion = torch.nn.CrossEntropyLoss()

standard_accuracy = calculate_standard_accuracy(test_loader, model, criterion, device)
print("Standard Accuracy:", standard_accuracy)