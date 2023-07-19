import torch.nn as nn

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

    def forward(self, x, ood_test=False):
        feature = self.backbone(x).squeeze()
        output = self.classifier(feature)

        if ood_test:
            return output, feature
        else:
            return output
    def output_num(self):
        return self.__in_features

class build_ResNet18:
    def __init__(self, base_net = None):
        self.base_net = base_net
    
    def build(self, num_classes):
        return ResNet18(num_classes = num_classes, base_net = self.base_net()) 