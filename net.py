import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from t2t_vit import Channel, Spatial
from function import adaptive_instance_normalization


# Convolution operation
class ConvLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, is_last=False):
        super(ConvLayer, self).__init__()
        reflection_padding = int(np.floor(kernel_size / 2))
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)
        # self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        # self.bn = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout2d(p=0.5)
        self.is_last = is_last

    def forward(self, x):
        x = x.cuda()

        # out = self.conv2d(x)
        # out = self.bn(out)
        out = self.reflection_pad(x)

        out = self.conv2d(out)

        if self.is_last is False:
            out = F.leaky_relu(out, inplace=True)
        return out

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = ConvLayer(channels, channels, kernel_size=3, stride=1)
        self.bn1 = nn.BatchNorm2d(channels, affine=True)
        self.relu = nn.ReLU()
        self.conv2 = ConvLayer(channels, channels, kernel_size=3, stride=1)
        self.bn2 = nn.BatchNorm2d(channels, affine=True)

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + residual
        out = self.relu(out)
        return out


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class self_a(nn.Module):
    """ Self attention Layer"""

    def __init__(self, in_dim):
        super(self_a, self).__init__()
        self.chanel_in = in_dim
        # self.activation = activation

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.value_conv_x = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.value_conv_y = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)

        self.s_conv = nn.Conv2d(in_channels=2*in_dim, out_channels=in_dim, kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)
        self.softmax2 = nn.Softmax(dim=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.beta = nn.Parameter(torch.zeros(1))

    def forward(self, x, y):
        """
            inputs :
                x : input feature maps( B * C * W * H)
            returns :
                out : self attention value + input feature
                attention: B * N * N (N is Width*Height)
        """
        m_batchsize, C, width, height = x.size()

        proj_query = self.query_conv(y).view(m_batchsize, -1, width * height).permute(0, 2, 1)  # B*N*C

        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)  # B*C*N

        energy = torch.bmm(proj_query, proj_key)  # batch的matmul B*N*N
        attention = self.softmax(energy)  # B * (N) * (N)

        proj_value_x = self.value_conv_x(x).view(m_batchsize, -1, width * height)  # B * C * N
        proj_value_y = self.value_conv_y(y).view(m_batchsize, -1, width * height)

        out_x = torch.bmm(proj_value_x, attention.permute(0, 2, 1))  # B*C*N
        out_y = torch.bmm(proj_value_y, attention.permute(0, 2, 1))

        out_x = out_x.view(m_batchsize, C, width, height)
        out_y = out_y.view(m_batchsize, C, width, height)

        x_att = self.gamma * out_x
        y_att = self.beta * out_y

        return x_att, y_att

class Encoder(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super().__init__()
        self.conv1 = ConvLayer(in_channels, out_channels, kernel_size, stride)
        self.res = ResidualBlock(out_channels)
        self.conv2 = ConvLayer(out_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        x = self.conv1(x)
        x = self.res(x)
        x = self.conv2(x)
        return x

class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super().__init__()
        self.conv1 = ConvLayer(in_channels, in_channels//2, kernel_size, stride)
        self.conv2 = ConvLayer(in_channels//2, out_channels, kernel_size, stride)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x


#1
class net(nn.Module):
    def __init__(self, input_nc=2, output_nc=1):
        super(net, self).__init__()
        kernel_size = 1
        stride = 1

        self.down1 = nn.AvgPool2d(2)
        self.down2 = nn.AvgPool2d(4)
        self.down3 = nn.AvgPool2d(8)

        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up2 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.up3 = nn.Upsample(scale_factor=8, mode='bilinear', align_corners=True)



        self.conv_in1 = ConvLayer(input_nc, input_nc, kernel_size, stride)
        self.conv_out = ConvLayer(64, 1, kernel_size, stride, is_last=True)
        # self.conv_t3 = ConvLayer(128, 64, kernel_size=1, stride=1)
        # self.conv_t2 = ConvLayer(64, 32, kernel_size=1, stride=1)
        # self.conv_t0 = ConvLayer(3, 3, kernel_size, stride)

        self.en0 = Encoder(2, 64, kernel_size, stride)
        self.en1 = Encoder(64, 64, kernel_size, stride)
        self.en2 = Encoder(64, 64, kernel_size, stride)
        self.en3 = Encoder(64, 64, kernel_size, stride)

        # self.de3 = Decoder(96, 32, kernel_size, stride)
        # self.de2 = Decoder(48, 16, kernel_size, stride)
        # self.de1 = Decoder(19, 3, kernel_size, stride)
        # self.de0 = Decoder(3, 3, kernel_size, stride)

        # self.f1 = ConvLayer(6, 3, kernel_size, stride)
        # self.f2 = ConvLayer(32, 16, kernel_size, stride)
        # self.f3 = ConvLayer(64, 32, kernel_size, stride)

        # self.ctrans0 = Channel(size=256, embed_dim=128, patch_size=16, channel=3)
        # self.ctrans1 = Channel(size=128, embed_dim=128, patch_size=16, channel=16)
        # self.ctrans2 = Channel(size=64, embed_dim=128, patch_size=16, channel=32)
        self.ctrans3 = Channel(size=32, embed_dim=128, patch_size=16, channel=64)

        #self.strans0 = Spatial(size=256, embed_dim=128*2, patch_size=8, channel=3)
        #self.strans1 = Spatial(size=128, embed_dim=256*2, patch_size=8, channel=16)
        # self.strans2 = Spatial(size=256, embed_dim=512*2, patch_size=8, channel=32)
        self.strans3 = Spatial(size=256, embed_dim=1024*2, patch_size=4, channel=64)


    def en(self, vi, ir):
        f = torch.cat([vi, ir], dim=1)
        x = self.conv_in1(f)
        x0 = self.en0(x)
        x1 = self.en1(self.down1(x0))
        x2 = self.en2(self.down1(x1))
        x3 = self.en3(self.down1(x2))

        return [x0, x1, x2, x3]

    # def de(self, f):
    #     x0, x1, x2, x3 = f
    #     o3 = self.de3(torch.cat([self.up1(x3), x2], dim=1))
    #     o2 = self.de2(torch.cat([self.up1(o3), x1], dim=1))
    #     o1 = self.de1(torch.cat([self.up1(o2), x0], dim=1))
    #     o0 = self.de0(o1)
    #     out = self.conv_out1(o0)
    #     return out

    def forward(self, vi, ir):
        # w = ir / (torch.max(ir) - torch.min(ir))
        # f_pre = w * ir + (1-w) * vi
        f0 = torch.cat([vi, ir], dim=1)
        x = self.conv_in1(f0)
        x0 = self.en0(x)
        x1 = self.en1(self.down1(x0))
        x2 = self.en2(self.down1(x1))
        x3 = self.en3(self.down1(x2))

        x3t = self.strans3(self.ctrans3(x3))
        # x2r = self.ctrans2(x2)
        # x1r = self.ctrans1(x1)
        # x0r = self.ctrans0(x0)
        # x3m = torch.clamp(x3r, 0, 1)
        x3m = x3t
        x3r = x3 * x3m
        x2m = self.up1(x3m)
        x2r = x2 * x2m
        x1m = self.up1(x2m) + self.up2(x3m)
        x1r = x1 * x1m
        x0m = self.up1(x1m) + self.up2(x2m) + self.up3(x3m)
        x0r = x0 * x0m

        other =self.up3(x3r) + self.up2(x2r) + self.up1(x1r) + x0r
        f1 = self.conv_out(other) 
        # out = self.conv_out(f1)

        return f1



