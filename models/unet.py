import torch
import torch.nn.functional as F
from torch import nn
from configs.default_config import LIDCConfig as cfg

from torch.nn import Conv3d
class Conv2_5d(nn.Conv3d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        kernel_size = (1,kernel_size,kernel_size)
        padding = (0,padding,padding)
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)


conv = locals()[cfg.GLOBAL_CONV]

class _EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=False):
        super(_EncoderBlock, self).__init__()
        layers = [
            conv(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            conv(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
        ]
        if dropout:
            layers.append(nn.Dropout())
        layers.append(nn.MaxPool3d(kernel_size=2, stride=2))
        self.encode = nn.Sequential(*layers)

    def forward(self, x):
        return self.encode(x)


class _DecoderBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super(_DecoderBlock, self).__init__()
        self.decode = nn.Sequential(
            conv(in_channels, middle_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(middle_channels),
            nn.ReLU(inplace=True),
            conv(middle_channels, middle_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(middle_channels),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2),
            conv(middle_channels, out_channels, kernel_size=1)
        )

    def forward(self, x):
        return self.decode(x)


class UNet(nn.Module):
    def __init__(self, num_classes):
        super(UNet, self).__init__()
        self.enc1 = _EncoderBlock(3, cfg.channels[0])
        self.enc2 = _EncoderBlock(cfg.channels[0], cfg.channels[1])
        self.enc3 = _EncoderBlock(cfg.channels[1], cfg.channels[2])
        self.enc4 = _EncoderBlock(cfg.channels[2], cfg.channels[3])
        self.center = _DecoderBlock(cfg.channels[3], cfg.channels[3], cfg.channels[3])
        self.dec4 = _DecoderBlock(2*cfg.channels[3], cfg.channels[3], cfg.channels[2])
#         self.center = _DecoderBlock(cfg.channels[2], cfg.channels[3], cfg.channels[2])
        self.dec3 = _DecoderBlock(2*cfg.channels[2], cfg.channels[2], cfg.channels[1])
        self.dec2 = _DecoderBlock(2*cfg.channels[1], cfg.channels[1], cfg.channels[0])
        self.dec1 = nn.Sequential(
            conv(2*cfg.channels[0], cfg.channels[0], kernel_size=3, padding=1),
            nn.BatchNorm3d(cfg.channels[0]),
            nn.ReLU(inplace=True),
            conv(cfg.channels[0], cfg.channels[0], kernel_size=3, padding=1),
            nn.BatchNorm3d(cfg.channels[0]),
            nn.ReLU(inplace=True),
        )
        self.final = conv(cfg.channels[0], num_classes, kernel_size=1)

    def forward(self, x):
        enc1 = self.enc1(x)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        enc4 = self.enc4(enc3)
        center = self.center(enc4)
        dec4 = self.dec4(torch.cat([center, F.interpolate(enc4, center.size()[2:], mode='trilinear')], 1))
        dec3 = self.dec3(torch.cat([dec4, F.interpolate(enc3, dec4.size()[2:], mode='trilinear')], 1))
        dec2 = self.dec2(torch.cat([dec3, F.interpolate(enc2, dec3.size()[2:], mode='trilinear')], 1))
        dec1 = self.dec1(torch.cat([dec2, F.interpolate(enc1, dec2.size()[2:], mode='trilinear')], 1))
        final = self.final(dec1)
        return F.interpolate(final, x.size()[2:], mode='trilinear')
    