from modeling.unet_part import *
import torch
import torch.nn as nn
from modeling.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, BatchNorm):
        super(UNet, self).__init__()
        self.inc = inconv(n_channels, 16, BatchNorm)
        self.down1 = down(16, 32, BatchNorm)
        self.down2 = down(32, 64, BatchNorm)
        self.down3 = down(64, 128, BatchNorm)
        self.down4 = down(128, 256, BatchNorm)
        self.up1 = up(384, 128, BatchNorm)
        self.up2 = up(192, 64, BatchNorm)
        self.up3 = up(96, 32, BatchNorm)
        self.up4 = up(48, 16, BatchNorm)
        self.outc = outconv(16, n_classes)
        self._init_weight()

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        return x
    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, SynchronizedBatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


def build_preprocessor(in_ch, out_ch, BatchNorm):
    return UNet(in_ch, out_ch, BatchNorm)

