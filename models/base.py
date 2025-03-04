# -*- coding: utf-8 -*-
"""
@author: caigentan@AnHui University
@software: PyCharm
@file: SwinTransformer.py
@time: 2021/5/6 6:13
"""

import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import numpy as np
import torch.nn.functional as F
import os
from TSE import TSE
from back.Swin import SwinTransformer


def conv3x3(in_planes, out_planes, stride=1, has_bias=False):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=has_bias)


def conv3x3_bn_relu(in_planes, out_planes, stride=1):
    return nn.Sequential(
        conv3x3(in_planes, out_planes, stride),
        nn.BatchNorm2d(out_planes),
        nn.ReLU(inplace=True),
    )


class SwinNet(nn.Module):
    def __init__(self):
        super(SwinNet, self).__init__()

        self.rgb_swin = SwinTransformer(embed_dim=128, depths=[2, 2, 18, 2], num_heads=[4, 8, 16, 32])
        self.depth_swin = SwinTransformer(embed_dim=128, depths=[2, 2, 18, 2], num_heads=[4, 8, 16, 32])

        self.up2 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.up4 = nn.UpsamplingBilinear2d(scale_factor=4)
        self.conv1024 = conv3x3_bn_relu(1024, 512)
        self.conv512 = conv3x3_bn_relu(512, 256)
        self.conv256 = conv3x3_bn_relu(256, 128)
        self.conv128 = conv3x3_bn_relu(128, 64)
        self.conv64_1 = conv3x3(64, 1)

        self.edge_layer = Edge_Module()
        self.edge_feature = conv3x3_bn_relu(1, 32)
        self.fuse_edge_sal = conv3x3(32, 1)
        self.up_edge = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=4),
            conv3x3(32, 1)
        )
        self.pyramidpooling = PyramidPooling(1024, 1024)
        self.prt1 = PRT(1024, 1024)
        self.prt2 = PRT(512, 1024)
        self.prt3 = PRT(256, 512)
        self.prt4 = PRT(128, 256)

        self.tse1 = TSE(1024, use_gpu=True)
        self.tse1_1 = TSE(1024, use_gpu=True)
        self.tse2 = TSE(512, use_gpu=True)
        self.tse2_1 = TSE(512, use_gpu=True)
        self.tse3 = TSE(256, use_gpu=True)
        self.tse3_1 = TSE(256, use_gpu=True)
        self.tse4 = TSE(128, use_gpu=True)
        self.tse4_1 = TSE(128, use_gpu=True)

        self.fuse1 = fuse(1024, 1024)
        self.fuse2 = fuse(512, 1024)
        self.fuse3 = fuse(256, 512)
        self.fuse4 = fuse(128, 256)

        self.conv128_32 = nn.Conv2d(128, 32, 1, 1, 0)
        self.conv64_32 = nn.Conv2d(64, 1, 1, 1, 0)
        self.conv32_1 = nn.Conv2d(32, 1, 1, 1, 0)

        self.score_1 = nn.Conv2d(128, 1, 1, 1, 0)
        self.score_ppm = nn.Conv2d(1024, 1, 1, 1, 0)

        self.relu = nn.ReLU(True)

    def forward(self, x, d):
        rgb_list = self.rgb_swin(x)
        depth_list = self.depth_swin(d)

 
        r1 = rgb_list[3] + depth_list[3]
        r2 = rgb_list[2] + depth_list[2]
        r3 = rgb_list[1] + depth_list[1]
        r4 = rgb_list[0] + depth_list[0]


        # print('pre1 de size: '+pre1.shape)
        r1 = self.conv1024(self.up2(r1))  # [512,24,24]
        r2 = self.conv512(self.up2(r2 + r1))  # self.up2(r3+r4) = [512,48,48]
        r3 = self.conv256(self.up2(r2 + r3))  # self.up2(r2+r3) = [256,96,96]
        r4 = self.conv128(r4 + r3 )  # r1+r2 = [128,96,96]

        out = self.up4(r4)
        out = self.conv64_1(out)


        return out


    def load_pre(self, pre_model):
        self.rgb_swin.load_state_dict(torch.load(pre_model)['model'], strict=False)
        print(f"RGB SwinTransformer loading pre_model ${pre_model}")
        self.depth_swin.load_state_dict(torch.load(pre_model)['model'], strict=False)
        print(f"Depth SwinTransformer loading pre_model ${pre_model}")


def convblock(in_, out_, ks, st, pad):
    return nn.Sequential(
        nn.Conv2d(in_, out_, ks, st, pad),
        nn.BatchNorm2d(out_),
        nn.ReLU(inplace=True)
    )


class fuse(nn.Module):
    def __init__(self, in_1, in_2):
        super(fuse, self).__init__()
        self.ca = ChannelAttention(in_1)
        self.conv1 = convblock(2 * in_1, 128, 3, 1, 1)
        self.conv2 = convblock(in_2, 128, 3, 1, 1)
        self.conv3 = convblock(128, in_1, 3, 1, 1)
        self.conv4 = convblock(in_1, 128, 3, 1, 1)

    def forward(self, x, r, d, pre):
        size = x.size()[2:]
        rd = torch.cat((r, d), dim=1)
        rd = F.interpolate(self.conv1(rd), size, mode='bilinear', align_corners=True)
        pre = F.interpolate(self.conv2(pre), size, mode='bilinear', align_corners=True)
        x = F.interpolate(self.conv4(x), size, mode='bilinear', align_corners=True)
        fuse = rd + x + pre
        fuse = fuse * x * pre
        fuse = self.conv3(fuse)
        # print(fuse.shape)
        return self.ca(fuse)


class PyramidPooling(nn.Module):

    def __init__(self, in_channels, out_channels, **kwargs):
        super(PyramidPooling, self).__init__()
        inter_channels = int(in_channels / 4)
        self.conv = nn.Conv2d(2 * in_channels, in_channels, 1, 1, 0, bias=False)
        self.conv1 = nn.Conv2d(in_channels, inter_channels, 1, 1, 0, bias=False)
        self.conv2 = nn.Conv2d(in_channels, inter_channels, 1, 1, 0, bias=False)
        self.conv3 = nn.Conv2d(in_channels, inter_channels, 1, 1, 0, bias=False)
        self.conv4 = nn.Conv2d(in_channels, inter_channels, 1, 1, 0, bias=False)
        self.out = nn.Conv2d(in_channels * 2, out_channels, 1, 1, 0, bias=False)

    def pool(self, x, size):
        avgpool = nn.AdaptiveAvgPool2d(size)
        return avgpool(x)

    def upsample(self, x, size):
        return F.interpolate(x, size, mode='bilinear', align_corners=True)

    def forward(self, x1, x2):
        # print(x1.shape)
        # print(x2.shape)
        size = x1.size()[2:]
        x = torch.cat((x1, x2), dim=1)
        x = self.conv(x)
        feat1 = self.upsample(self.conv1(self.pool(x, 1)), size)
        feat2 = self.upsample(self.conv2(self.pool(x, 2)), size)
        feat3 = self.upsample(self.conv3(self.pool(x, 3)), size)
        feat4 = self.upsample(self.conv4(self.pool(x, 6)), size)
        x = torch.cat([x, feat1, feat2, feat3, feat4], dim=1)
        x = self.out(x)
        return x

class NonLocal(nn.Module):
    def __init__(self, channel):
        super(NonLocal, self).__init__()
        self.inter_channel = channel // 2
        self.conv_phi = nn.Conv2d(channel, self.inter_channel, 1, 1, 0)
        self.conv_theta = nn.Conv2d(channel, self.inter_channel, 1, 1, 0)
        self.conv_g = nn.Conv2d(channel, self.inter_channel, 1, 1, 0)
        self.softmax = nn.Softmax(dim=1)
        self.conv_mask = nn.Conv2d(self.inter_channel, channel, 1, 1, 0)



    def forward(self, x):
        # [N, C, H , W]
        #       print(self.conv_sgate2(F.interpolate(t[3], size=x.shape[2:], mode='bilinear', align_corners=True)).shape)
        # z = t[3]
        b, c, h, w = x.size()
        # 获取phi特征，维度为[N, C/2, H * W]，注意是要保留batch和通道维度的，是在HW上进行的
        x_phi = self.conv_phi(x).view(b, c, -1)

        # 获取theta特征，维度为[N, H * W, C/2]
        x_theta = self.conv_theta(x).view(b, c, -1).permute(0, 2, 1).contiguous()

        # 获取g特征，维度为[N, H * W, C/2]
        x_g = self.conv_g(x).view(b, c, -1).permute(0, 2, 1).contiguous()
        # 对phi和theta进行矩阵乘，[N, H * W, H * W]
        mul_theta_phi = torch.matmul(x_theta, x_phi)
        # softmax拉到0~1之间
        mul_theta_phi = self.softmax(mul_theta_phi)
        # 与g特征进行矩阵乘运算，[N, H * W, C/2]
        mul_theta_phi_g = torch.matmul(mul_theta_phi, x_g)
        # [N, C/2, H, W]
        mul_theta_phi_g = mul_theta_phi_g.permute(0, 2, 1).contiguous().view(b, self.inter_channel, h, w)
        # 1X1卷积扩充通道数
        mask = self.conv_mask(mul_theta_phi_g)
        out = mask + x
        return out


class PRT(nn.Module):
    def __init__(self, in_1, in_2):
        super(PRT, self).__init__()
        self.in_1 = in_1
        self.in_2 = in_2
        self.conv2_split = nn.ModuleList()
        for i in range(4):
            self.conv2_split.append(nn.Conv2d(in_1 // 4, 32, 3, 1, 1))
        self.conv2 = convblock(128, 128, 3, 1, 1)

        self.conv3_split = nn.ModuleList()
        for i in range(4):
            self.conv3_split.append(nn.Conv2d(in_1 // 4, 32, 3, 1, 1))
        self.conv3 = convblock(128, 128, 1, 1, 1)

        self.conv1 = nn.Conv2d(in_2, in_1 // 4, 1, 1, 1)
        self.conv1_1 = convblock(in_1 // 4, in_1 // 4, 3, 1, 1)

        self.conv_1 = nn.Conv2d(in_2, in_1 // 4, 1, 1, 1)
        self.conv_1_1 = convblock(in_1 // 4, in_1 // 4, 3, 1, 1)
        # self.conv1_1 = nn.Conv2d(128, in_1, 3, 1, 1)
        self.sigmoid = nn.Sigmoid()
        # self.conv2 = convblock(in_1, 128, 3, 1, 1)
        # self.conv3 = convblock(in_2, 128, 3, 1, 1)
        self.conv4 = convblock(128, in_1, 3, 1, 1)
        self.CA = ChannelAttention(in_1)
        self.CA1 = ChannelAttention(128)
        self.CA2 = ChannelAttention(128)

        self.conv5 = convblock(1024, 128, 3, 1, 1)

        self.conv6 = convblock(in_1 + in_2, in_1, 3, 1, 1)
        self.CA_ = ChannelAttention(in_1)

        self.attention1 = NonLocal(in_1)
        self.attention2 = NonLocal(in_1)
        self.erase1 = convblock(in_1*2,in_1,3,1,1)
        self.erase2 = convblock(in_1*2, in_1, 3, 1, 1)

    def forward(self, x, erase_x, pre, mid,erase_mid, ppm):
        size = x.size()[2:]
        # print(size)

        pre1 = self.conv1(pre)
        pre1 = self.conv1_1(F.interpolate(pre1, size, mode='bilinear', align_corners=True))
        # print(pre1.shape)
        # mask1 = self.sigmoid(pre1)

        pre2 = self.conv_1(pre)
        pre2 = self.conv_1_1(F.interpolate(pre2, size, mode='bilinear', align_corners=True))
        # print(pre2.shape)
        # mask2 = self.sigmoid(pre2)

        x = self.attention1(self.erase1( torch.cat([x,erase_x],dim=1)))
        mid = self.attention2(self.erase2(torch.cat([mid,erase_mid],dim=1)))

        xx = torch.split(x, (self.in_1 // 4, self.in_1 // 4, self.in_1 // 4, self.in_1 // 4), dim=1)


        aa = []
        xx_ = []
        # print(x.shape)
      #  print("xx",xx[1].shape)
        pre1 = F.interpolate(pre1, xx[1].size()[2:], mode='bilinear', align_corners=True)
        mask1 = torch.sigmoid(pre1)
        for i in range(len(xx)):
            aa.append(xx[i] * mask1)
        for i in range(len(xx)):
            xx_.append(self.conv2_split[i](aa[i]))

        xx1 = torch.cat((xx_[0], xx_[1], xx_[2], xx_[3]), dim=1)
        x = self.conv2(xx1)
        x = F.interpolate(x, size, mode='bilinear', align_corners=True)
        x = self.CA1(x)
        # print(x.shape)
        # print(mask.shape)
        # x1 = x * mask

        midd = torch.split(mid, (self.in_1 // 4, self.in_1 // 4, self.in_1 // 4, self.in_1 // 4), dim=1)
        bb = []
        midd_ = []
        pre2 = F.interpolate(pre2, midd[1].size()[2:], mode='bilinear', align_corners=True)
        mask2 = torch.sigmoid(pre2)
        for i in range(len(midd)):
            bb.append(midd[i] * mask2)
        for i in range(len(midd)):
            midd_.append(self.conv3_split[i](bb[i]))

        midd1 = torch.cat((midd_[0], midd_[1], midd_[2], midd_[3]), dim=1)
        mid = self.conv3(midd1)
        mid = F.interpolate(mid, size, mode='bilinear', align_corners=True)
        mid = self.CA2(mid)
        ppm = self.conv5(F.interpolate(ppm, size, mode='bilinear', align_corners=True))
        out = x + mid + ppm
        out = self.conv4(out)
        return out
        # dil = F.interpolate(dil, size, mode='bilinear', align_corners=True)
        # out1 = torch.cat((out, dil), dim=1)
        # out1 = self.conv6(out1)
        # out1 = F.interpolate(out1, size, mode='bilinear', align_corners=True)
        # out2 = out1 + out

        # return self.CA(out), self.CA_(out2)



class fuse_enhance(nn.Module):
    def __init__(self, infeature):
        super(fuse_enhance, self).__init__()
        self.depth_channel_attention = ChannelAttention(infeature)
        self.rgb_channel_attention = ChannelAttention(infeature)
        self.rd_spatial_attention = SpatialAttention()
        self.rgb_spatial_attention = SpatialAttention()
        self.depth_spatial_attention = SpatialAttention()

    def forward(self, r, d):
        assert r.shape == d.shape, "rgb and depth should have same size"
        mul_fuse = r * d
        sa = self.rd_spatial_attention(mul_fuse)
        r_f = r * sa
        d_f = d * sa
        r_ca = self.rgb_channel_attention(r_f)
        d_ca = self.depth_channel_attention(d_f)

        r_out = r * r_ca
        d_out = d * d_ca
        return r_out, d_out


class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


## Residual Channel Attention Block (RCAB)
class RCAB(nn.Module):
    def __init__(
            self, n_feat, kernel_size=3, reduction=16,
            bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        super(RCAB, self).__init__()
        modules_body = []
        for i in range(2):
            modules_body.append(self.default_conv(n_feat, n_feat, kernel_size, bias=bias))
            if bn: modules_body.append(nn.BatchNorm2d(n_feat))
            if i == 0: modules_body.append(act)
        modules_body.append(CALayer(n_feat, reduction))
        self.body = nn.Sequential(*modules_body)
        self.res_scale = res_scale

    def default_conv(self, in_channels, out_channels, kernel_size, bias=True):
        return nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size // 2), bias=bias)

    def forward(self, x):
        res = self.body(x)
        # res = self.body(x).mul(self.res_scale)
        res += x
        return res


class Edge_Module(nn.Module):
    def __init__(self, in_fea=[128, 256, 512], mid_fea=32):
        super(Edge_Module, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_fea[0], mid_fea, 1)
        self.conv4 = nn.Conv2d(in_fea[1], mid_fea, 1)
        self.conv5 = nn.Conv2d(in_fea[2], mid_fea, 1)
        self.conv5_2 = nn.Conv2d(mid_fea, mid_fea, 3, padding=1)
        self.conv5_4 = nn.Conv2d(mid_fea, mid_fea, 3, padding=1)
        self.conv5_5 = nn.Conv2d(mid_fea, mid_fea, 3, padding=1)
        self.up2 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.classifer = nn.Conv2d(mid_fea * 3, 1, kernel_size=3, padding=1)
        self.rcab = RCAB(mid_fea * 3)

    def forward(self, x2, x4, x5):
        _, _, h, w = x2.size()
        edge2_fea = self.relu(self.conv2(x2))
        edge2 = self.relu(self.conv5_2(edge2_fea))
        edge4_fea = self.relu(self.conv4(x4))
        edge4 = self.relu(self.conv5_4(edge4_fea))
        edge5_fea = self.relu(self.conv5(x5))
        edge5 = self.relu(self.conv5_5(edge5_fea))

        edge4 = F.interpolate(edge4, size=(h, w), mode='bilinear', align_corners=True)
        edge5 = F.interpolate(edge5, size=(h, w), mode='bilinear', align_corners=True)

        edge = torch.cat([edge2, edge4, edge5], dim=1)
        edge = self.rcab(edge)
        edge = self.classifer(edge)
        return edge


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        # self.fc2 = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)
        self.fc2 = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        out = self.sigmoid(out)
        return out * x


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(1, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = max_out
        x = self.conv1(x)
        return self.sigmoid(x)



