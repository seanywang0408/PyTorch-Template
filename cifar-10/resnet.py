import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']

# conv = nn.Conv2d
# bn = nn.BatchNorm2d
# maxpool = nn.MaxPool2d
# avgpool = nn.AvgPool2d


from torch.nn import Conv3d
from configs.default_config import LIDCConfig as cfg

class Conv2_5d(nn.Conv3d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        kernel_size = (1,kernel_size,kernel_size)
        padding = (0,padding,padding)
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)


conv = locals()[cfg.GLOBAL_CONV]

bn = nn.BatchNorm3d
maxpool = nn.MaxPool3d
avgpool = nn.AvgPool3d

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}



def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        state_dict = model_zoo.load_url(model_urls['resnet18'])
        for key in list(state_dict.keys()):
            # print(key)
            if 'fc' in key:
                del state_dict[key]
        model.load_state_dict(state_dict,strict=False)
    return model




def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return conv(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = bn(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = bn(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = conv(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = bn(planes)
        self.conv2 = conv(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = bn(planes)
        self.conv3 = conv(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = bn(planes * 4)
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


class ResNet(nn.Module):

    def __init__(self, block, layers):#, num_classes=1000):
        self.inplanes = cfg.channels[0]
        super(ResNet, self).__init__()
        self.conv1 = conv(3, cfg.channels[0], kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = bn(cfg.channels[0])
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = maxpool(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, cfg.channels[0], layers[0])
        self.layer2 = self._make_layer(block, cfg.channels[1], layers[1], stride=2)
        self.layer3 = self._make_layer(block, cfg.channels[2], layers[2])
        self.layer4 = self._make_layer(block, cfg.channels[3], layers[3], stride=2)
        # self.avgpool = avgpool(7, stride=1)
        # self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, conv):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, bn):
                m.weight.data.fill_(1)
                m.bias.data.zero_()



    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                bn(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)



    def forward(self, x):
        # print(x.shape)
        # print(self.conv1)
        x = self.conv1(x)
        # print(x.shape)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x2 = x.clone()
        # print(x.shape)
        x = self.layer1(x)
        x = self.layer2(x)

        x1 = x.clone()

        x = self.layer3(x)
        x = self.layer4(x)
        
        # x = self.avgpool(x)
        # x = x.view(x.size(0), -1)
        # x = self.fc(x)

        return x, x1, x2

    def load_state_dict(self, state_dict, strict=True):
        if cfg.GLOBAL_CONV == 'Conv2_5d':
            for key in list(state_dict.keys()):
                if state_dict[key].dim()==4:
                    state_dict[key] = state_dict[key].unsqueeze(2)
        super().load_state_dict(state_dict, strict=strict)