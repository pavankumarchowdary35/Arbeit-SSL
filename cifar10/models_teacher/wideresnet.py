import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    def __init__(self, in_planes, out_planes, stride, dropRate=0.0, activate_before_residual=False, activation = 'relu'):
        super(BasicBlock, self).__init__()
        #self.bn1 = nn.BatchNorm2d(in_planes, momentum=0.001)
        self.bn1 = nn.BatchNorm2d(in_planes, momentum=0.1)
        self.activation = activation
        if self.activation == 'relu':
            self.relu1 = nn.ReLU(inplace=True)
            self.relu2 = nn.ReLU(inplace=True)
        elif self.activation == 'leaky':
            self.relu1 = nn.LeakyReLU(negative_slope=0.1, inplace=True)
            self.relu2 = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        elif self.activation == 'silu':
            self.relu1 = nn.SiLU(inplace=True)
            self.relu2 = nn.SiLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)
        #self.bn2 = nn.BatchNorm2d(in_planes, momentum=0.001)
        self.bn2 = nn.BatchNorm2d(out_planes, momentum=0.1)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.droprate = dropRate
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                               padding=0, bias=False) or None
        self.activate_before_residual = activate_before_residual
    def forward(self, x):
        if not self.equalInOut and self.activate_before_residual == True:
            x = self.relu1(self.bn1(x))
        else:
            out = self.relu1(self.bn1(x))
        out = self.relu2(self.bn2(self.conv1(out if self.equalInOut else x)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.conv2(out)
        return torch.add(x if self.equalInOut else self.convShortcut(x), out)

class NetworkBlock(nn.Module):
    def __init__(self, nb_layers, in_planes, out_planes, block, stride, dropRate=0.0, activate_before_residual=False, activation='relu'):
        super(NetworkBlock, self).__init__()
        self.activation = activation
        self.layer = self._make_layer(block, in_planes, out_planes, nb_layers, stride, dropRate, activate_before_residual)
        
    def _make_layer(self, block, in_planes, out_planes, nb_layers, stride, dropRate, activate_before_residual):
        layers = []
        for i in range(int(nb_layers)):
            layers.append(block(i == 0 and in_planes or out_planes, out_planes, i == 0 and stride or 1, dropRate, activate_before_residual, self.activation))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        return self.layer(x)

class WideResNet(nn.Module):
    def __init__(self, first_stride=1, num_classes=10, depth=28, widen_factor=2, dropRate=0.0, activation='relu'):
        super(WideResNet, self).__init__()
        nChannels = [16, 16*widen_factor, 32*widen_factor, 64*widen_factor]
        assert((depth - 4) % 6 == 0)
        n = (depth - 4) / 6
        block = BasicBlock
        # 1st conv before any network block
        self.conv1 = nn.Conv2d(3, nChannels[0], kernel_size=3, stride=1,
                               padding=1, bias=False)
        # 1st block
        # CIFAR, SVHN
        self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, first_stride, dropRate, activation=activation)
        # STl
        #self.block1 = NetworkBlock(n, nChannels[0], nChannels[1], block, 2, dropRate, activation=activation)
        # 2nd block
        self.block2 = NetworkBlock(n, nChannels[1], nChannels[2], block, 2, dropRate, activation=activation)
        # 3rd block
        self.block3 = NetworkBlock(n, nChannels[2], nChannels[3], block, 2, dropRate, activation=activation)
        # global average pooling and classifier
        #self.bn1 = nn.BatchNorm2d(nChannels[3], momentum=0.001)
        self.bn1 = nn.BatchNorm2d(nChannels[3], momentum=0.1)
        self.activation = activation
        if self.activation == 'relu':
            self.relu = nn.ReLU(inplace=True)
        elif self.activation == 'leaky':
            self.relu = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        elif self.activation == 'silu':
            self.relu = nn.SiLU(inplace=True)
        self.fc = nn.Linear(nChannels[3], num_classes)
        self.nChannels = nChannels[3]

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.zero_()

    def forward(self, x):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        #16 is Feature map size
        #out = F.avg_pool2d(out, 16)
        out = F.avg_pool2d(out, 8)
        # feature vector
        out = out.view(-1, self.nChannels)
        out = self.fc(out)
        return out