import math
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import torch.optim as optim

def call_bn(bn, x):
    return bn(x)

class CNN(nn.Module):
    def __init__(self, input_channel=3, n_outputs=10, dropout_rate=0.25, top_bn=False):
        self.dropout_rate = dropout_rate
        self.top_bn = top_bn
        super(CNN, self).__init__()
        self.c1=nn.Conv2d(input_channel,128,kernel_size=3,stride=1, padding=1)
        self.c2=nn.Conv2d(128,128,kernel_size=3,stride=1, padding=1)
        self.c3=nn.Conv2d(128,128,kernel_size=3,stride=1, padding=1)
        self.c4=nn.Conv2d(128,256,kernel_size=3,stride=1, padding=1)
        self.c5=nn.Conv2d(256,256,kernel_size=3,stride=1, padding=1)
        self.c6=nn.Conv2d(256,256,kernel_size=3,stride=1, padding=1)
        self.c7=nn.Conv2d(256,512,kernel_size=3,stride=1, padding=0)
        self.c8=nn.Conv2d(512,256,kernel_size=3,stride=1, padding=0)
        self.c9=nn.Conv2d(256,128,kernel_size=3,stride=1, padding=0)
        self.l_c1=nn.Linear(128,n_outputs)
        self.bn1=nn.BatchNorm2d(128)
        self.bn2=nn.BatchNorm2d(128)
        self.bn3=nn.BatchNorm2d(128)
        self.bn4=nn.BatchNorm2d(256)
        self.bn5=nn.BatchNorm2d(256)
        self.bn6=nn.BatchNorm2d(256)
        self.bn7=nn.BatchNorm2d(512)
        self.bn8=nn.BatchNorm2d(256)
        self.bn9=nn.BatchNorm2d(128)

    def forward(self, x,):
        h=x
        h=self.c1(h)
        h=F.leaky_relu(call_bn(self.bn1, h), negative_slope=0.01)
        h=self.c2(h)
        h=F.leaky_relu(call_bn(self.bn2, h), negative_slope=0.01)
        h=self.c3(h)
        h=F.leaky_relu(call_bn(self.bn3, h), negative_slope=0.01)
        h=F.max_pool2d(h, kernel_size=2, stride=2)
        h=F.dropout2d(h, p=self.dropout_rate)

        h=self.c4(h)
        h=F.leaky_relu(call_bn(self.bn4, h), negative_slope=0.01)
        h=self.c5(h)
        h=F.leaky_relu(call_bn(self.bn5, h), negative_slope=0.01)
        h=self.c6(h)
        h=F.leaky_relu(call_bn(self.bn6, h), negative_slope=0.01)
        h=F.max_pool2d(h, kernel_size=2, stride=2)
        h=F.dropout2d(h, p=self.dropout_rate)

        h=self.c7(h)
        h=F.leaky_relu(call_bn(self.bn7, h), negative_slope=0.01)
        h=self.c8(h)
        h=F.leaky_relu(call_bn(self.bn8, h), negative_slope=0.01)
        h=self.c9(h)
        h=F.leaky_relu(call_bn(self.bn9, h), negative_slope=0.01)
        h=F.avg_pool2d(h, kernel_size=h.data.shape[2])

        h = h.view(h.size(0), h.size(1))
        logit=self.l_c1(h)
        if self.top_bn:
            logit=call_bn(self.bn_c1, logit)
        return logit

class CNN7(nn.Module):
    def __init__(self, input_channel=3, n_outputs=10, dropout_rate=0.25, momentum=0.1):
        self.dropout_rate = dropout_rate
        self.momentum = momentum
        super(CNN7, self).__init__()
        self.c1=nn.Conv2d(input_channel, 64,kernel_size=3,stride=1, padding=1)
        self.c2=nn.Conv2d(64,64,kernel_size=3,stride=1, padding=1)
        self.c3=nn.Conv2d(64,128,kernel_size=3,stride=1, padding=1)
        self.c4=nn.Conv2d(128,128,kernel_size=3,stride=1, padding=1)
        self.c5=nn.Conv2d(128,196,kernel_size=3,stride=1, padding=1)
        self.c6=nn.Conv2d(196,16,kernel_size=3,stride=1, padding=1)
        self.linear1=nn.Linear(16, n_outputs)
        self.bn1=nn.BatchNorm2d(64, momentum=self.momentum)
        self.bn2=nn.BatchNorm2d(64, momentum=self.momentum)
        self.bn3=nn.BatchNorm2d(128, momentum=self.momentum)
        self.bn4=nn.BatchNorm2d(128, momentum=self.momentum)
        self.bn5=nn.BatchNorm2d(196, momentum=self.momentum)
        self.bn6=nn.BatchNorm2d(16, momentum=self.momentum)

    def forward(self, x,):
        h=x
        h=self.c1(h)
        h=F.relu(call_bn(self.bn1, h))
        h=self.c2(h)
        h=F.relu(call_bn(self.bn2, h))
        h=F.max_pool2d(h, kernel_size=2, stride=2)

        h=self.c3(h)
        h=F.relu(call_bn(self.bn3, h))
        h=self.c4(h)
        h=F.relu(call_bn(self.bn4, h))
        h=F.max_pool2d(h, kernel_size=2, stride=2)

        h=self.c5(h)
        h=F.relu(call_bn(self.bn5, h))
        h=self.c6(h)
        h=F.relu(call_bn(self.bn6, h))
        h=F.max_pool2d(h, kernel_size=2, stride=2)

        h=F.adaptive_avg_pool2d(h, (1,1))
        h = h.view(h.size(0), -1)
        logit=self.linear1(h)
        return logit

class MLPNet(nn.Module):
    def __init__(self):
        super(MLPNet, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 256)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


import torch
import torch.nn as nn
import torchvision.models as models


class ModifiedVGG19(nn.Module):
    def __init__(self,num_class):
        super(ModifiedVGG19, self).__init__()
        # Load pretrained VGG-19 model
        vgg19 = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1)

        # Remove the last two fully connected layers and add Global Average Pooling and new layers
        self.features = vgg19.features  # Keep the convolutional part of VGG-19
        self.fc = nn.Sequential(
            nn.Linear(512, 4096),  # First new FC layer
            nn.ReLU(inplace=True),
            nn.Linear(4096, 2048),  # Second new FC layer
            nn.ReLU(inplace=True),
            nn.Linear(2048, num_class)  # Output layer for regression (predicting a single number)
        )

    def forward(self, x):
        # print("1",x.shape)
        x = self.features(x)
        # print("2",x.shape)
        x = torch.mean(x, dim=[2, 3])
        # print("3",x.shape)
        x = torch.flatten(x, 1)
        # print("4",x.shape)
        x = self.fc(x)
        # x=x.view(16)
        # print("5",x.shape)
        return x


class ModifiedResNet34(nn.Module):
    def __init__(self,num_class):
        super(ModifiedResNet34, self).__init__()
        # Load pretrained VGG-19 model
        resnet34 = models.resnet34(pretrained=True)

        self.features = nn.Sequential(
            resnet34.conv1,
            resnet34.bn1,
            resnet34.relu,
            resnet34.maxpool,
            resnet34.layer1,
            resnet34.layer2,
            resnet34.layer3,
            resnet34.layer4,
            resnet34.avgpool  # 注意：ResNet 已有全局平均池化！
        )

        # 自定义全连接层（类似你的 VGG 修改）
        self.fc = nn.Sequential(
            nn.Linear(512, num_class)  # 输出 (batch_size, 1)
        )

    def forward(self, x):
        # print("1",x.shape)
        x = self.features(x)
        # print("2",x.shape)
        x = torch.mean(x, dim=[2, 3])
        # print("3",x.shape)
        x = torch.flatten(x, 1)
        # print("4",x.shape)
        x = self.fc(x)
        # x = x.view(16)
        # print("5",x.shape)
        return x