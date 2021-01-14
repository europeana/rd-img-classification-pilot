import torch.nn as nn
import torchvision



class ConvNet(nn.Module):
    def __init__(self,output_size):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer3 = nn.Sequential(
            nn.Linear(32*53*53, 512),
            nn.ReLU())
        self.layer4 = nn.Sequential(
            nn.Linear(512, output_size)
            )
        
        self.dropout = nn.Dropout(p=0.0)

        self.sm = nn.Softmax(dim=1)
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        #print(out.shape)
        out = out.reshape(out.size(0), -1)
        out = self.layer3(out)
        out = self.dropout(out)
        out = self.layer4(out)
        out = self.sm(out)
        return out


class ResNet(nn.Module):
    def __init__(self,size, output_size):
        super(ResNet, self).__init__()

        if size not in [18,34,50,101,152]:
            raise Exception('Wrong size for resnet')
        if size == 18:
            self.net = torchvision.models.resnet18(pretrained=True)
        elif size == 34:
            self.net = torchvision.models.resnet34(pretrained=True)
        elif size == 50:
            self.net = torchvision.models.resnet50(pretrained=True)
        elif size == 101:
            self.net = torchvision.models.resnet101(pretrained=True)
        elif size == 152:
            self.net = torchvision.models.resnet152(pretrained=True)

        #initialize the fully connected layer
        self.net.fc = nn.Linear(self.net.fc.in_features, output_size)
        self.sm = nn.Softmax(dim=1)

    def forward(self, x):
        out = self.net(x)
        out = self.sm(out)
        return out


class ResNetExperimental(nn.Module):
    def __init__(self,size, output_size):
        super(ResNetExperimental, self).__init__()

        if size not in [18,34,50,101,152]:
            raise Exception('Wrong size for resnet')
        if size == 18:
            self.net = torchvision.models.resnet18(pretrained=True)
        elif size == 34:
            self.net = torchvision.models.resnet34(pretrained=True)
        elif size == 50:
            self.net = torchvision.models.resnet50(pretrained=True)
        elif size == 101:
            self.net = torchvision.models.resnet101(pretrained=True)
        elif size == 152:
            self.net = torchvision.models.resnet152(pretrained=True)

        #initialize the fully connected layer
        hidden_size = 100
        self.net.fc = nn.Linear(self.net.fc.in_features, hidden_size)
        self.dropout = nn.Dropout(p=0.25)
        self.fc_out = nn.Linear(hidden_size, output_size)
        self.sm = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.net(x)
        x = self.dropout(x)
        x = self.fc_out(x)
        out = self.sm(x)
        return out