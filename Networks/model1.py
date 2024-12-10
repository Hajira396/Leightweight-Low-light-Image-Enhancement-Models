import torch
import torch.nn as nn
import numpy as np

def conv(in_channels, out_channels, kernel_size, bias=False, padding=1, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=bias, stride=stride)

class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, relu=True, bn=False, bias=False):
        super(BasicConv, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)

class SpatialAttn(nn.Module):
    def __init__(self, in_planes):
        super(SpatialAttn, self).__init__()
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size=5, stride=1, padding=2, relu=False)

    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = torch.sigmoid(x_out)  # broadcasting
        return x * scale

class ChannelAttn(nn.Module):
    def __init__(self, channels, reduction=8):
        super(ChannelAttn, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y

class DAU(nn.Module):
    def __init__(self, n_feat, reduction=8, bias=False):
        super(DAU, self).__init__()
        self.body = nn.Sequential(
            conv(n_feat, n_feat, kernel_size=3, bias=bias),
            nn.PReLU(),
            conv(n_feat, n_feat, kernel_size=3, bias=bias),
        )
        self.sa = SpatialAttn(n_feat)
        self.ca = ChannelAttn(n_feat, reduction)

    def forward(self, x):
        res = self.body(x)
        sa_branch = self.sa(res)
        ca_branch = self.ca(res)
        res = sa_branch + ca_branch
        return res

class ResidualGroup(nn.Module):
    def __init__(self, n_feat, n_resblocks=2, reduction=8):
        super(ResidualGroup, self).__init__()
        self.resblocks = nn.ModuleList([DAU(n_feat, reduction) for _ in range(n_resblocks)])
        self.conv = conv(n_feat, n_feat, kernel_size=3)

    def forward(self, x):
        res = x
        for block in self.resblocks:
            res = block(res)
        res = self.conv(res)
        res += x
        return res

class Model1(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, n_feat=16, n_resgroups=2, n_resblocks=2, reduction=8):
        super(Model1, self).__init__()
        self.conv_in = conv(in_channels, n_feat, kernel_size=3)
        self.resgroups = nn.ModuleList([ResidualGroup(n_feat, n_resblocks, reduction) for _ in range(n_resgroups)])
        self.conv_out = conv(n_feat, out_channels, kernel_size=3)

    def forward(self, x):
        h = self.conv_in(x)
        for group in self.resgroups:
            h = group(h)
        h = self.conv_out(h)
        h += x
        return h




