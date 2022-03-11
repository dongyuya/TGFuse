import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F




def calc_mean_std(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std


def adaptive_instance_normalization(content_feat, style_feat):
    assert (content_feat.size()[:2] == style_feat.size()[:2])
    size = content_feat.size()
    style_mean, style_std = calc_mean_std(style_feat)
    content_mean, content_std = calc_mean_std(content_feat)

    normalized_feat = (content_feat - content_mean.expand(
        size)) / content_std.expand(size)
    return normalized_feat * style_std.expand(size) + style_mean.expand(size)


def _calc_feat_flatten_mean_std(feat):
    # takes 3D feat (C, H, W), return mean and std of array within channels
    assert (feat.size()[0] == 3)
    assert (isinstance(feat, torch.FloatTensor))
    feat_flatten = feat.view(3, -1)
    mean = feat_flatten.mean(dim=-1, keepdim=True)
    std = feat_flatten.std(dim=-1, keepdim=True)
    return feat_flatten, mean, std


import torch
import torch.nn as nn
from torchvision import models


class Vgg16(nn.Module):
    def __init__(self):
        super(Vgg16, self).__init__()
        features = models.vgg16(pretrained=True).features
        self.to_relu_1_2 = nn.Sequential()
        self.to_relu_2_2 = nn.Sequential()
        self.to_relu_3_3 = nn.Sequential()
        self.to_relu_4_3 = nn.Sequential()
        self.conv0 = nn.Conv2d(1, 3, 1, 1)
        
        for x in range(4):
            self.to_relu_1_2.add_module(str(x), features[x])
        for x in range(4, 9):
            self.to_relu_2_2.add_module(str(x), features[x])
        for x in range(9, 16):
            self.to_relu_3_3.add_module(str(x), features[x])
        for x in range(16, 23):
            self.to_relu_4_3.add_module(str(x), features[x])
        
        # don't need the gradients, just want the features
        # for param in self.parameters():
        #     param.requires_grad = False
    
    def forward(self, x):
        x = self.conv0(x)
        h = self.to_relu_1_2(x)
        h_relu_1_2 = h
        h = self.to_relu_2_2(h)
        h_relu_2_2 = h
        h = self.to_relu_3_3(h)
        h_relu_3_3 = h
        h = self.to_relu_4_3(h)
        h_relu_4_3 = h
        out = [h_relu_1_2, h_relu_2_2, h_relu_3_3, h_relu_4_3]
        # out = [h_relu_2_2, h_relu_4_3]
        return out


class Vgg_l(nn.Module):
    def __init__(self):
        super(Vgg_l, self).__init__()
        features = models.vgg16(pretrained=True).features
        self.to_relu_1_2 = nn.Sequential()
        self.to_relu_2_2 = nn.Sequential()
        self.to_relu_3_3 = nn.Sequential()
        self.to_relu_4_3 = nn.Sequential()
        self.conv0 = nn.Conv2d(1, 3, 1, 1)

        for x in range(4):
            self.to_relu_1_2.add_module(str(x), features[x])
        # for x in range(4, 9):
        #     self.to_relu_2_2.add_module(str(x), features[x])
        # for x in range(9, 16):
        #     self.to_relu_3_3.add_module(str(x), features[x])
        # for x in range(16, 23):
        #     self.to_relu_4_3.add_module(str(x), features[x])

        # don't need the gradients, just want the features
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        x = self.conv0(x)
        h = self.to_relu_1_2(x)
        h_relu_1_2 = h
        # h = self.to_relu_2_2(h)
        # h_relu_2_2 = h
        # h = self.to_relu_3_3(h)
        # h_relu_3_3 = h
        # h = self.to_relu_4_3(h)
        # h_relu_4_3 = h
        # out = [h_relu_1_2, h_relu_2_2, h_relu_3_3, h_relu_4_3]
        out = [h_relu_1_2]
        return out

def gram(x):
    (bs, ch, h, w) = x.size()
    f = x.view(bs, ch, w*h)
    f_T = f.transpose(1, 2)
    G = f.bmm(f_T) / (ch * h * w)
    return G

def p_loss(ir, out):
    mse_loss = torch.nn.MSELoss()
    vgg = Vgg16().cuda()
    ir_f = vgg(ir)
    # vi_f = vgg(vi)
    out_f = vgg(out)
    # vi_loss = 0
    ir_loss = mse_loss(ir_f[3], out_f[3])
    # for j in range(4):
    #     # vi_loss += mse_loss(vi_f[j], out_f[j])
    #     ir_loss += mse_loss(ir_f[j], out_f[j])

    p_loss = ir_loss
    return p_loss

def c_loss(vi, out):
    mse_loss = torch.nn.MSELoss()
    vgg = Vgg_l().cuda()

    vi_f = vgg(vi)
    out_f = vgg(out)
    # vi_loss = 0
    vi_loss = mse_loss(vi_f[0], out_f[0])
    # for j in range(4):
    #     # vi_loss += mse_loss(vi_f[j], out_f[j])
    #     ir_loss += mse_loss(ir_f[j], out_f[j])
    p_loss = vi_loss
    return p_loss
    
    
def perceptual_loss(output, sty):
    mse_loss = torch.nn.MSELoss()
    vgg = Vgg16().cuda()
    style_features = vgg(sty)
    out_features = vgg(output)
    style_gram = [gram(sty_map) for sty_map in style_features]
    out_gram = [gram(out_map) for out_map in out_features]
    p_loss = 0
    for j in range(2):
        p_loss += mse_loss(out_gram[j], style_gram[j])
    # p_loss = mse_loss(gram(out_features),gram(style_features))
    return p_loss


class TVLoss(nn.Module):
    def __init__(self,TVLoss_weight=1):
        super(TVLoss,self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self,x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self._tensor_size(x[:,:,1:,:])
        count_w = self._tensor_size(x[:,:,:,1:])
        h_tv = torch.pow((x[:,:,1:,:]-x[:,:,:h_x-1,:]),2).sum()
        w_tv = torch.pow((x[:,:,:,1:]-x[:,:,:,:w_x-1]),2).sum()
        return self.TVLoss_weight*2*(h_tv/count_h+w_tv/count_w)/batch_size

    def _tensor_size(self,t):
        return t.size()[1]*t.size()[2]*t.size()[3]

def grad(img):
    kernel = torch.FloatTensor([[0,1,0],[1,-4,1],[0,1,0]]).cuda()
    kernel = kernel.unsqueeze(0).unsqueeze(0)
    g = F.conv2d(img, kernel, stride=1, padding=0)
    return g


def grad_loss(ir, out):
    # mse_loss = torch.nn.MSELoss()
    L1loss = torch.nn.L1Loss()
    grad_ir = grad(ir)
    # grad_vi = grad(vi)
    grad_out = grad(out)
    loss = L1loss(grad_ir, grad_out)
    # loss2 = L1loss(grad_vi, grad_out)
    # grad_loss = torch.mean(loss)
    return loss