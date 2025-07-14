from torchvision import models
import torch.nn.functional as F
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.init as init
from torch.autograd import Function


class GradReverse(Function):
    def __init__(self, lambd):
        self.lambd = lambd

    def forward(self, x):
        return x.view_as(x)

    def backward(self, grad_output):
        return grad_output * -self.lambd


def grad_reverse(x, lambd=1.0):
    return GradReverse(lambd)(x)


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, use_dropout):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout):
        conv_block = []
        p = 0
        # TODO: support padding types
        assert (padding_type == 'zero')
        p = 1

        # TODO: InstanceNorm
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p),
                       norm_layer(dim, affine=True),
                       nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p),
                       norm_layer(dim, affine=True)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out


class ResBottle(nn.Module):
    def __init__(self, option='resnet18', pret=True):
        super(ResBottle, self).__init__()
        if option == 'resnet18':
            model_ft = models.resnet18(pretrained=pret)
        if option == 'resnet50':
            model_ft = models.resnet50(pretrained=pret)
        if option == 'resnet101':
            model_ft = models.resnet101(pretrained=pret)
        if option == 'resnet152':
            model_ft = models.resnet152(pretrained=pret)
        mod = list(model_ft.children())
        mod.pop()
        # self.model_ft =model_ft
        self.features = nn.Sequential(*mod)

        self.bottleneck = nn.Linear(model_ft.fc.in_features, 256)
        nn.init.normal_(self.bottleneck.weight.data, 0, 0.005)
        nn.init.constant_(self.bottleneck.bias.data, 0.1)

        self.dim = 256

    def forward(self, x):

        x = self.features(x)

        x = x.view(x.size(0), -1)

        x = self.bottleneck(x)

        x = x.view(x.size(0), self.dim)
        return x

    def output_num(self):
        return self.dim


class ResNet_all(nn.Module):
    def __init__(self, option='resnet18', pret=True):
        super(ResNet_all, self).__init__()
        self.dim = 2048
        if option == 'resnet18':
            model_ft = models.resnet18(pretrained=pret)
            self.dim = 512
        if option == 'resnet50':
            model_ft = models.resnet50(pretrained=pret)
        if option == 'resnet101':
            model_ft = models.resnet101(pretrained=pret)
        if option == 'resnet152':
            model_ft = models.resnet152(pretrained=pret)
        # mod = list(model_ft.children())
        # mod.pop()
        # self.model_ft =model_ft
        self.conv1 = model_ft.conv1
        self.bn0 = model_ft.bn1
        self.relu = model_ft.relu
        self.maxpool = model_ft.maxpool
        self.layer1 = model_ft.layer1
        self.layer2 = model_ft.layer2
        self.layer3 = model_ft.layer3
        self.layer4 = model_ft.layer4
        self.pool = model_ft.avgpool
        self.fc = nn.Linear(2048, 12)

    def forward(self, x, layer_return=False, input_mask=False, mask=None, mask2=None):
        if input_mask:
            x = self.conv1(x)
            x = self.bn0(x)
            x = self.relu(x)
            conv_x = x
            x = self.maxpool(x)
            fm1 = mask * self.layer1(x)
            fm2 = mask2 * self.layer2(fm1)
            fm3 = self.layer3(fm2)
            fm4 = self.pool(self.layer4(fm3))
            x = fm4.view(fm4.size(0), self.dim)
            x = self.fc(x)
            return x  # ,fm1
        else:
            x = self.conv1(x)
            x = self.bn0(x)
            x = self.relu(x)
            conv_x = x
            x = self.maxpool(x)
            fm1 = self.layer1(x)
            fm2 = self.layer2(fm1)
            fm3 = self.layer3(fm2)
            fm4 = self.pool(self.layer4(fm3))
            x = fm4.view(fm4.size(0), self.dim)
            x = self.fc(x)
            if layer_return:
                return x, fm1, fm2
            else:
                return x

class ClassifierMLP(nn.Module):

    def __init__(self,
                 input_size,
                 output_size,
                 dropout,
                 last='tanh'):
        super(ClassifierMLP, self).__init__()

        self.last = last
        self.net = nn.Sequential(
            nn.Dropout(p=dropout),

            nn.Linear(input_size, int(input_size / 4)),
            nn.ReLU(),

            nn.Linear(int(input_size / 4), int(input_size / 16)),
            nn.ReLU(),

            nn.Linear(int(input_size / 16), output_size))

        if last == 'logsm':
            self.last_layer = nn.LogSoftmax(dim=-1)
        elif last == 'sm':
            self.last_layer = nn.Softmax(dim=-1)
        elif last == 'tanh':
            self.last_layer = nn.Tanh()
        elif last == 'sigmoid':
            self.last_layer = nn.Sigmoid()
        elif last == 'relu':
            self.last_layer = nn.ReLU()

    def forward(self, input):
        y = self.net(input)
        if self.last != None:
            y = self.last_layer(y)

        return y


class CNNlayer(nn.Module):

    def __init__(self,
                 in_channel=1, kernel_size=8, stride=1, padding=1,
                 mp_kernel_size=2, mp_stride=2, dropout=0.):
        super(CNNlayer, self).__init__()

        layer1 = nn.Sequential(
            nn.Conv1d(in_channel, 4, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm1d(4),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=mp_kernel_size, stride=mp_stride))

        layer2 = nn.Sequential(
            nn.Conv1d(4, 16, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=mp_kernel_size, stride=mp_stride))

        layer3 = nn.Sequential(
            nn.Conv1d(16, 32, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=mp_kernel_size, stride=mp_stride))

        layer4 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=mp_kernel_size, stride=mp_stride))

        layer5 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveMaxPool1d(4),
            nn.Flatten())

        dp = nn.Dropout(dropout)

        self.fs = nn.Sequential(
            layer1,
            layer2,
            layer3,
            layer4,
            dp,
            layer5)

    def forward(self, tar, x=None, y=None):
        h = self.fs(tar)

        return h


class FeatureExtractor(nn.Module):

    def __init__(self, in_channel, window_sizes=[4, 8, 16, 24, 32], block=CNNlayer, dropout=0.5):
        super(FeatureExtractor, self).__init__()

        self.convs = nn.ModuleList([
            block(in_channel=in_channel, kernel_size=h, dropout=dropout)
            for h in window_sizes])
        self.fl = nn.Flatten()
        self.fc = nn.Linear(2560,256)
        self.dim = 256
    def forward(self, input):
        out = [conv(input) for conv in self.convs]
        out = torch.cat(out, dim=1)
        out = self.fl(out)
        out = self.fc(out)
        return out
    def output_num(self):
        return self.dim


class BaseModel(nn.Module):

    def __init__(self,
                 input_size,
                 num_classes,
                 dropout):
        super(BaseModel, self).__init__()

        self.G = FeatureExtractor(in_channel=input_size, dropout=dropout)

        self.C = ClassifierMLP(2560, num_classes, dropout, last=None)

    def forward(self, input):
        f = self.G(input)
        predictions = self.C(f)
        if self.training:
            return predictions, f
        else:
            return predictions

class ResClassifier(nn.Module):
    def __init__(self, num_classes=12, num_layer=2, num_unit=2048, prob=0.5, middle=1000):
        super(ResClassifier, self).__init__()
        layers = []
        # currently 10000 units
        layers.append(nn.Dropout(p=prob))
        layers.append(nn.Linear(num_unit, middle))
        layers.append(nn.BatchNorm1d(middle, affine=True))
        layers.append(nn.ReLU(inplace=True))

        for i in range(num_layer - 1):
            layers.append(nn.Dropout(p=prob))
            layers.append(nn.Linear(middle, middle))
            layers.append(nn.BatchNorm1d(middle, affine=True))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(middle, num_classes))
        self.classifier = nn.Sequential(*layers)

    def set_lambda(self, lambd):
        self.lambd = lambd

    def forward(self, x, reverse=False):
        if reverse:
            x = grad_reverse(x, self.lambd)
        x = self.classifier(x)
        return x
