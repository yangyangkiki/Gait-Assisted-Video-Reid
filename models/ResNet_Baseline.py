from __future__ import absolute_import

import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
import torchvision

__all__ = ['ResNet50_2Branch_TP_KD', 'ResNet50_2Branch_TP']

# Bottleneck of standard ResNet50/101, with kernel size equal to 1
class Bottleneck1x1(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck1x1, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=1, stride=stride,
                               padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class ResNet50_2Branch_TP_KD(nn.Module):
    def __init__(self, num_classes, loss={'xent'}, **kwargs):
        super(ResNet50_2Branch_TP_KD, self).__init__()
        self.loss = loss
        resnet50_appearance = torchvision.models.resnet50(pretrained=True)
        resnet50_appearance.layer4[0].conv2.stride = 1
        resnet50_appearance.layer4[0].downsample[0].stride = 1
        self.base_appearance = nn.Sequential(*list(resnet50_appearance.children())[:-2])
        self.feat_appearance_dim = 2048

        resnet50_gait = torchvision.models.resnet50(pretrained=True)
        resnet50_gait.layer4[0].conv2.stride = 1
        resnet50_gait.layer4[0].downsample[0].stride = 1
        self.base_gait = nn.Sequential(*list(resnet50_gait.children())[:-2])
        self.feat_gait_dim = 2048

        self.feat_dim = 2048
        self.bn = nn.BatchNorm1d(self.feat_dim)
        self.classifier = nn.Linear(self.feat_dim, num_classes)

        self.attconv = nn.Sequential(
                nn.Conv2d(2048, 512, kernel_size=1, stride=1, padding=0, bias=True),
                nn.BatchNorm2d(512),
                nn.ReLU(),
                nn.Conv2d(512, 64, kernel_size=1, stride=1, padding=0, bias=True),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.Conv2d(64, 1, kernel_size=1, stride=1, padding=0, bias=True),
                nn.BatchNorm2d(1),
                nn.ReLU(),
                )

        self.groupingbn = nn.BatchNorm2d(512 * 4)
        self.nonlinear = Bottleneck1x1(512 * 4, 512, stride=1)

        self.head = nn.Sequential(nn.Linear(2048, 512),
                                  nn.BatchNorm1d(512),
                                  nn.ReLU(inplace=True),
                                  nn.Linear(512, 2048),
                                  nn.ReLU(inplace=True)
                                  )

    def forward(self, x, x_g):  # 32 3 4 224 112
        # appearance
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        b = x.size(0)
        t = x.size(1)
        x = x.contiguous().view(b*t, x.size(2), x.size(3), x.size(4))
        x = self.base_appearance(x)
        x = F.avg_pool2d(x, x.size()[2:])
        x = x.view(b,t,-1)
        x = x.permute(0,2,1)  # 16 2048 4

        if not self.training:
            return x

        # gait
        x_g = x_g.permute(0, 2, 1, 3, 4).contiguous()
        b = x_g.size(0)
        t = x_g.size(1)
        x_g = x_g.contiguous().view(b * t, x_g.size(2), x_g.size(3), x_g.size(4))
        x_g = self.base_gait(x_g)
        x_g = F.avg_pool2d(x_g, x_g.size()[2:])
        x_g = x_g.view(b, t, -1)
        x_g = x_g.permute(0, 2, 1)  # 16 2048 4

        f_a = F.avg_pool1d(x, t)  # 16 2048 1  # mean time pooling
        f_g = F.avg_pool1d(x_g, t)  # 16 2048 1  # mean time pooling

        f_cat = torch.cat((f_a,f_g), 2).contiguous().unsqueeze(3)

        # MLP projection head:
        f_a_head = f_a.clone().view(b, self.feat_appearance_dim)  # 16 2048
        head = self.head(f_a_head)
        f_g_head = f_g.clone().view(b, self.feat_appearance_dim)

        att = self.attconv(f_cat)
        att = F.softmax(att, dim=2)

        out = f_cat * att
        out = out.contiguous().squeeze(3)

        f_a = out[:,:,0]
        f_g = out[:,:,1]

        # classification
        f = f_a + f_g  # sum

        f = f.contiguous().unsqueeze(2).unsqueeze(3)
        f = self.groupingbn(f)
        f = self.nonlinear(f)

        f = f.contiguous().view(f.size(0), -1)
        f = self.bn(f)  # zhaoyang

        y = self.classifier(f)

        if self.loss == {'xent'}:
            return y
        elif self.loss == {'xent', 'htri'}:
            return y, f, f_a, f_g, head, f_g_head
        elif self.loss == {'cent'}:
            return y, f, f_a, f_g,head, f_g_head
        else:
            raise KeyError("Unsupported loss: {}".format(self.loss))


class ResNet50_2Branch_TP(nn.Module):
    def __init__(self, num_classes, loss={'xent'}, **kwargs):
        super(ResNet50_2Branch_TP, self).__init__()
        self.loss = loss
        resnet50_appearance = torchvision.models.resnet50(pretrained=True)
        resnet50_appearance.layer4[0].conv2.stride = 1
        resnet50_appearance.layer4[0].downsample[0].stride = 1
        self.base_appearance = nn.Sequential(*list(resnet50_appearance.children())[:-2])
        self.feat_appearance_dim = 2048

        resnet50_gait = torchvision.models.resnet50(pretrained=True)
        resnet50_gait.layer4[0].conv2.stride = 1
        resnet50_gait.layer4[0].downsample[0].stride = 1
        self.base_gait = nn.Sequential(*list(resnet50_gait.children())[:-2])
        self.feat_gait_dim = 2048

        self.feat_dim = 2048
        self.bn = nn.BatchNorm1d(self.feat_dim)
        self.classifier = nn.Linear(self.feat_dim, num_classes)

        self.attconv = nn.Sequential(
                nn.Conv2d(2048, 512, kernel_size=1, stride=1, padding=0, bias=True),
                nn.BatchNorm2d(512),
                nn.ReLU(),
                nn.Conv2d(512, 64, kernel_size=1, stride=1, padding=0, bias=True),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.Conv2d(64, 1, kernel_size=1, stride=1, padding=0, bias=True),
                nn.BatchNorm2d(1),
                nn.ReLU(),
                )

        self.groupingbn = nn.BatchNorm2d(512 * 4)
        self.nonlinear = Bottleneck1x1(512 * 4, 512, stride=1)

    def forward(self, x, x_g):  # 32 3 4 224 112
        # appearance
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        b = x.size(0)
        t = x.size(1)
        x = x.contiguous().view(b*t, x.size(2), x.size(3), x.size(4))
        x = self.base_appearance(x)
        x = F.avg_pool2d(x, x.size()[2:])
        x = x.view(b,t,-1)
        x = x.permute(0,2,1)  # 16 2048 4

        # gait
        x_g = x_g.permute(0, 2, 1, 3, 4).contiguous()
        b = x_g.size(0)
        t = x_g.size(1)
        x_g = x_g.contiguous().view(b * t, x_g.size(2), x_g.size(3), x_g.size(4))
        x_g = self.base_gait(x_g)
        x_g = F.avg_pool2d(x_g, x_g.size()[2:])
        x_g = x_g.view(b, t, -1)
        x_g = x_g.permute(0, 2, 1)  # 16 2048 4

        if not self.training:
            return x, x_g

        f_a = F.avg_pool1d(x, t)  # 16 2048 1  # mean time pooling

        f_g = F.avg_pool1d(x_g, t)  # 16 2048 1  # mean time pooling

        f_cat = torch.cat((f_a,f_g), 2).contiguous().unsqueeze(3)

        att = self.attconv(f_cat)
        att = F.softmax(att, dim=2)

        out = f_cat * att
        out = out.contiguous().squeeze(3)

        f_a = out[:,:,0]
        f_g = out[:,:,1]

        # classification
        f = f_a + f_g  # sum

        f = f.contiguous().unsqueeze(2).unsqueeze(3)
        f = self.groupingbn(f)
        f = self.nonlinear(f)

        f = f.contiguous().view(f.size(0), -1)
        f = self.bn(f)

        y = self.classifier(f)

        if self.loss == {'xent'}:
            return y
        elif self.loss == {'xent', 'htri'}:
            return y, f, f_a, f_g
        elif self.loss == {'cent'}:
            return y, f, f_a, f_g
        else:
            raise KeyError("Unsupported loss: {}".format(self.loss))
