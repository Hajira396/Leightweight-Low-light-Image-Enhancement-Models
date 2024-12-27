import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F

def conv(in_channels, out_channels, kernel_size=3, bias=False, padding=1, stride=1):
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
        scale = torch.sigmoid(x_out)  
        return x * scale


class ChannelAttn(nn.Module):
    def __init__(self, channels, reduction=8):
        super(ChannelAttn, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


class CAB(nn.Module):
    def __init__(self, n_feat, reduction=8, bias=False):
        super(CAB, self).__init__()
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


class UNetWithAttention(nn.Module):
    def __init__(self, input_channels=3, output_channels=3, num_filters=64):
        super(UNetWithAttention, self).__init__()

        # Encoder
        self.enc1 = self.conv_block(input_channels, num_filters)          # 3 -> 64
        self.enc2 = self.conv_block(num_filters, num_filters * 2)        # 64 -> 128
        self.enc3 = self.conv_block(num_filters * 2, num_filters * 4)    # 128 -> 256
        self.enc4 = self.conv_block(num_filters * 4, num_filters * 8)    # 256 -> 512
        self.bottleneck = self.conv_block(num_filters * 8, num_filters * 16)  # 512 -> 1024
        self.dec4 = self.conv_block(num_filters * 16 + num_filters * 8, num_filters * 8)   # 1024 + 512 = 1536 -> 512
        self.dec3 = self.conv_block(num_filters * 8 + num_filters * 4, num_filters * 4)     # 512 + 256 = 768 -> 256
        self.dec2 = self.conv_block(num_filters * 4 + num_filters * 2, num_filters * 2)     # 256 + 128 = 384 -> 128
        self.dec1 = self.conv_block(num_filters * 2 + num_filters, num_filters)             # 128 + 64 = 192 -> 64

        # Attention layers
        self.attn1 = CAB(num_filters * 2) 
        self.attn2 = CAB(num_filters * 4) 
        self.attn3 = CAB(num_filters * 8) 

        # Final layer
        self.final = nn.Conv2d(num_filters, output_channels, kernel_size=1)  # 64 -> 3

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)  # [B, 64, H, W]
        enc2 = self.enc2(F.max_pool2d(enc1, 2))  # [B, 128, H/2, W/2]
        enc2 = self.attn1(enc2)  # Apply attention
        enc3 = self.enc3(F.max_pool2d(enc2, 2))  # [B, 256, H/4, W/4]
        enc3 = self.attn2(enc3)  # Apply attention
        enc4 = self.enc4(F.max_pool2d(enc3, 2))  # [B, 512, H/8, W/8]
        enc4 = self.attn3(enc4)  # Apply attention

        bottleneck = self.bottleneck(F.max_pool2d(enc4, 2))  # [B, 1024, H/16, W/16]

        # Decoder
        dec4 = F.interpolate(bottleneck, scale_factor=2, mode='bilinear', align_corners=True)  # [B, 1024, H/8, W/8]
        dec4 = torch.cat((dec4, enc4), dim=1)  # [B, 1024 + 512, H/8, W/8] = [B, 1536, H/8, W/8]
        dec4 = self.dec4(dec4)  # [B, 512, H/8, W/8]

        dec3 = F.interpolate(dec4, scale_factor=2, mode='bilinear', align_corners=True)  # [B, 512, H/4, W/4]
        dec3 = torch.cat((dec3, enc3), dim=1)  # [B, 512 + 256, H/4, W/4] = [B, 768, H/4, W/4]
        dec3 = self.dec3(dec3)  # [B, 256, H/4, W/4]

        dec2 = F.interpolate(dec3, scale_factor=2, mode='bilinear', align_corners=True)  # [B, 256, H/2, W/2]
        dec2 = torch.cat((dec2, enc2), dim=1)  # [B, 256 + 128, H/2, W/2] = [B, 384, H/2, W/2]
        dec2 = self.dec2(dec2)  # [B, 128, H/2, W/2]

        dec1 = F.interpolate(dec2, scale_factor=2, mode='bilinear', align_corners=True)  # [B, 128, H, W]
        dec1 = torch.cat((dec1, enc1), dim=1)  # [B, 128 + 64, H, W] = [B, 192, H, W]
        dec1 = self.dec1(dec1)  # [B, 64, H, W]

        # Final output
        out = self.final(dec1)  # [B, 3, H, W]

        return out




