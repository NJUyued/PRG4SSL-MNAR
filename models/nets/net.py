import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm

from torchvision import models

class ReverseCLS(nn.Module):

    def __init__(self, in_dim, out_dim):
        super(ReverseCLS, self).__init__()
        self.fc = nn.Linear(in_dim, out_dim)
        self.main = nn.Sequential(self.fc, nn.Softmax(dim=-1))

    def forward(self, x):
        out = [x]
        for module in self.main.children():
            x = module(x)
            out.append(x)
        return out


class ResNet18(nn.Module):
    def __init__(self, num_classes, base_net):
       super(ResNet18, self).__init__() 
       self.backbone = nn.Sequential(*list(base_net.children())[:-1])
       self.__in_features=base_net.fc.in_features
       self.classifier = nn.Linear(self.__in_features, num_classes)

    def forward(self, x, ood_test=False, sel=1):
        feature = self.backbone(x).squeeze()
        output = self.classifier(feature)

        if ood_test:
            return output, feature
        else:
            return output
    def output_num(self):
        return self.__in_features


class CNN13(nn.Module):
       
    def __init__(self, num_classes=10, dropout=0.5):
        super(CNN13, self).__init__()

        #self.gn = GaussianNoise(0.15)
        self.activation = nn.LeakyReLU(0.1)
        self.conv1a = weight_norm(nn.Conv2d(3, 128, 3, padding=1))
        self.bn1a = nn.BatchNorm2d(128)
        self.conv1b = weight_norm(nn.Conv2d(128, 128, 3, padding=1))
        self.bn1b = nn.BatchNorm2d(128)
        self.conv1c = weight_norm(nn.Conv2d(128, 128, 3, padding=1))
        self.bn1c = nn.BatchNorm2d(128)
        self.mp1 = nn.MaxPool2d(2, stride=2, padding=0)
        self.drop1  = nn.Dropout(dropout)
        # self.drop1  = nn.Dropout2d(dropout)

        self.conv2a = weight_norm(nn.Conv2d(128, 256, 3, padding=1))
        self.bn2a = nn.BatchNorm2d(256)
        self.conv2b = weight_norm(nn.Conv2d(256, 256, 3, padding=1))
        self.bn2b = nn.BatchNorm2d(256)
        self.conv2c = weight_norm(nn.Conv2d(256, 256, 3, padding=1))
        self.bn2c = nn.BatchNorm2d(256)
        self.mp2 = nn.MaxPool2d(2, stride=2, padding=0)
        self.drop2  = nn.Dropout(dropout)
        # self.drop2  = nn.Dropout2d(dropout)
        
        self.conv3a = weight_norm(nn.Conv2d(256, 512, 3, padding=0))
        self.bn3a = nn.BatchNorm2d(512)
        self.conv3b = weight_norm(nn.Conv2d(512, 256, 1, padding=0))
        self.bn3b = nn.BatchNorm2d(256)
        self.conv3c = weight_norm(nn.Conv2d(256, 128, 1, padding=0))
        self.bn3c = nn.BatchNorm2d(128)
        self.ap3 = nn.AvgPool2d(6, stride=2, padding=0)
        
        self.fc1 =  weight_norm(nn.Linear(128, num_classes))
        self.__in_features=self.fc1.in_features
    def output_num(self):
        
        return self.__in_features    
    
    def forward(self, x, ood_test=False, sel=1):
        out = x
        ## layer 1-a###
        out = self.conv1a(out)
        out = self.bn1a(out)
        out = self.activation(out)
        
        ## layer 1-b###
        out = self.conv1b(out)
        out = self.bn1b(out)
        out = self.activation(out)
        
        ## layer 1-c###
        out = self.conv1c(out)
        out = self.bn1c(out)
        out = self.activation(out)
        
        out = self.mp1(out)
        out = self.drop1(out)
        
        
        ## layer 2-a###
        out = self.conv2a(out)
        out = self.bn2a(out)
        out = self.activation(out)
        
        ## layer 2-b###
        out = self.conv2b(out)
        out = self.bn2b(out)
        out = self.activation(out)
        
        ## layer 2-c###
        out = self.conv2c(out)
        out = self.bn2c(out)
        out = self.activation(out)
        
        
        out = self.mp2(out)
        out = self.drop2(out)
        
        
        ## layer 3-a###
        out = self.conv3a(out)
        out = self.bn3a(out)
        out = self.activation(out)
        
        ## layer 3-b###
        out = self.conv3b(out)
        out = self.bn3b(out)
        out = self.activation(out)
        
        ## layer 3-c###
        out = self.conv3c(out)
        out = self.bn3c(out)
        out = self.activation(out)
        
        out = self.ap3(out)
    
        feature = out.view(-1, 128)
        out = self.fc1(feature)
        if ood_test:
            return out, feature
        else:
            return out

            
    
def cnn13(num_classes=10, dropout = 0.5):
    model = CNN13(num_classes = num_classes, dropout=dropout)
    return model
