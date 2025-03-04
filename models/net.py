# -*- coding: utf-8 -*-
"""
@author: caigentan@AnHui University
@software: PyCharm
@file: SwinTransformer.py
@time: 2021/5/6 6:13
"""
from torch import nn, Tensor
import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import numpy as np
import torch.nn.functional as F
import os
from TSE import TSE
from back.Swin import SwinTransformer
from torchvision.ops import DeformConv2d
from typing import Optional
import  math
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
        self.conv2048_1024 = conv3x3_bn_relu(2048, 1024)
        self.conv1024_512 = conv3x3_bn_relu(1024, 512)
        self.conv512_256 = conv3x3_bn_relu(512, 256)
        self.conv256_32 = conv3x3_bn_relu(256, 32)
        self.conv128_64 = conv3x3_bn_relu(128,64)
        self.conv64_1 = conv3x3(64, 1)


        self.edge_feature = conv3x3_bn_relu(1, 32)
        self.fuse_edge_sal = conv3x3(32, 1)
        self.up_edge = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=4),
            conv3x3(32, 1)
        )


        self.tse1 = TSE(1024, use_gpu=True)
        self.tse1_1 = TSE(1024, use_gpu=True)
        self.tse2 = TSE(512, use_gpu=True)
        self.tse2_1 = TSE(512, use_gpu=True)
        self.tse3 = TSE(256, use_gpu=True)
        self.tse3_1 = TSE(256, use_gpu=True)
        self.tse4 = TSE(128, use_gpu=True)
        self.tse4_1 = TSE(128, use_gpu=True)

        self.rgb_fus1 = Fuse_Feature_Module(128)
        self.rgb_fus2 = Fuse_Feature_Module(256)
        self.rgb_fus3 = Fuse_Feature_Module(512)


        self.t_fus1 = Fuse_Feature_Module(128)
        self.t_fus2 = Fuse_Feature_Module(256)
        self.t_fus3 = Fuse_Feature_Module(512)

        self.msa_1 = MSA(1024,1024,d_model=1024)
        self.msa_2 = MSA(1024,1024,d_model=1024)

        self.conv128_32 = nn.Conv2d(128, 32, 1, 1, 0)
        self.conv64_32 = nn.Conv2d(64, 1, 1, 1, 0)
        self.conv32_1 = nn.Conv2d(32, 1, 1, 1, 0)

        self.score_1 = nn.Conv2d(128, 1, 1, 1, 0)
        self.score_ppm = nn.Conv2d(1024, 1, 1, 1, 0)
        self.score_ppm2 = nn.Conv2d(1024, 1, 1, 1, 0)

        self.relu = nn.ReLU(True)

    def forward(self, x, d):
        rgb_list = self.rgb_swin(x)
        depth_list = self.depth_swin(d)

        r4 = rgb_list[0]
        r3 = rgb_list[1]
        r2 = rgb_list[2]
        r1 = rgb_list[3]
        d4 = depth_list[0]
        d3 = depth_list[1]
        d2 = depth_list[2]
        d1 = depth_list[3]
        # for i in range(4):
        #     print(rgb_list[i].shape)

        erase_r1 = self.tse1(d1, r1)
        erase_d1 = self.tse1_1(r1, d1)   #1024,12,12
        erase_r2 = self.tse2(d2, r2)
        erase_d2 = self.tse2_1(r2, d2)   #512,24,24
        erase_r3 = self.tse3(d3, r3)
        erase_d3 = self.tse3_1(r3, d3)   #256,48,48
        erase_r4 = self.tse4(d4, r4)
        erase_d4 = self.tse4_1(r4, d4)   #128,96,96

        d_ppm = self.msa_1 (r1,d1,erase_r1)
        r_ppm = self.msa_2(d1,r1, erase_d1)
        ppm = d_ppm + r_ppm

        rf2 = self.rgb_fus3(r2,erase_r2,ppm)
        tf2 = self.t_fus3(d2,erase_d2,ppm)
        rf3 = self.rgb_fus2(r3,erase_r3,tf2)
        tf3 = self.t_fus2(d3,erase_d3,rf2)
        rf4 = self.rgb_fus1(r4,erase_r4,tf3)
        tf4 = self.t_fus1(d4,erase_d4,rf3)

        out = self.conv128_64(rf4+tf4)
        out = self.up4(out)

        out =self.conv64_1(out)
        score_g = self.score_ppm(ppm)

        return out,score_g



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

class defomableConv(nn.Module):
    def __init__(self, inC, outC, kH = 3, kW = 3, num_deformable_groups = 4):
        super(defomableConv, self).__init__()
        self.num_deformable_groups = num_deformable_groups
        self.inC = inC
        self.outC = outC
        self.kH = kH
        self.kW = kW
        self.offset = nn.Conv2d(inC, num_deformable_groups * 2 * kH * kW, kernel_size=(kH, kW), stride=(1, 1), padding=(1, 1), bias=False)
        self.deform = DeformConv2d(inC, outC, (kH, kW), stride=1, padding=1, groups=num_deformable_groups)

    def forward(self, x):
        offset = self.offset(x)
        out = self.deform(x, offset)
        return out

# gated MSA layer
def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


class MSA_layer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu"):
        super(MSA_layer, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        self.linear0 = nn.Linear(d_model,d_model)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_model, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.df_c = defomableConv(d_model,d_model)

        self.bn1 = nn.BatchNorm2d(d_model)
        self.bn2 = nn.BatchNorm2d(d_model)


    def forward(self, rgb, ert ,  pos: Optional[Tensor] = None, query_pos: Optional[Tensor] = None):

        r_f = self.multihead_attn(query=self.with_pos_embed(ert, query_pos).transpose(0, 1),#hw b c
                                   key=self.with_pos_embed(rgb, pos).transpose(0, 1),
                                   value=rgb.transpose(0, 1))[0].transpose(0, 1)#b hw c


       # fr2 = self.linear2(self.dropout(self.activation(self.linear1(fr))))  #FFN
        B,N,C= rgb.shape

        fr = r_f   #add&norm
        index = r_f
        fr = self.norm2(fr)
        fr = self.linear0(fr)
        fr_dc = self.df_c(rgb.transpose(1, 2).contiguous().view(B, C, int(math.sqrt(N)), int(math.sqrt(N))))
        fr = fr_dc.flatten(2).transpose(1, 2) + fr
        fr2 = self.linear2(self.dropout(self.activation(fr)))  #DCFFN
        fr = self.dropout3(fr2) + index
        fr = self.norm3(fr)


        return fr

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos


class MSA(nn.Module):
    def __init__(self, inC, outC, d_model=256,  decoder_layer=None):
        super(MSA, self).__init__()

        self.decoder_layer = MSA_layer(d_model=d_model, nhead=8)
        self.up2 = nn.UpsamplingBilinear2d(scale_factor=2)

        self.con3_3 = conv3x3_bn_relu(inC, outC)
    def forward(self, rgb, t, erase_r):  #r3 t3 [512,24,24] ar4 at4 [1024,12,12]

        index_rgb = rgb


        rgb = rgb.flatten(2).transpose(1, 2)  # b hw c
        t = t.flatten(2).transpose(1, 2)
        erase_r = erase_r.flatten(2).transpose(1, 2)
        ert = t + erase_r

        fr = self.decoder_layer(rgb,ert)

        fr = fr.transpose(1,2).contiguous().view(index_rgb.shape[0],index_rgb.shape[1],index_rgb.shape[2],index_rgb.shape[2])


        return fr


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



class NonLocal(nn.Module):
    def __init__(self, channel):
        super(NonLocal, self).__init__()
        self.inter_channel = channel // 2
        self.conv_phi = nn.Conv2d(channel, self.inter_channel, 1, 1, 0)
        self.conv_theta = nn.Conv2d(channel, self.inter_channel, 1, 1, 0)
        self.conv_g = nn.Conv2d(channel, self.inter_channel, 1, 1, 0)
        self.softmax = nn.Softmax(dim=1)
        self.conv_mask = nn.Conv2d(self.inter_channel, channel, 1, 1, 0)

        self.conv_sgate2 = convblock(1024, 128, 1, 1, 0)

        self.conv_sgate3 = convblock(128, 4, 3, 1, 1)

    def forward(self, x, y):
        # [N, C, H , W]
        #       print(self.conv_sgate2(F.interpolate(t[3], size=x.shape[2:], mode='bilinear', align_corners=True)).shape)
        # z = t[3]
        b, c, h, w = x.size()
        # 获取phi特征，维度为[N, C/2, H * W]，注意是要保留batch和通道维度的，是在HW上进行的
        x_phi = self.conv_phi(y).view(b, c, -1)

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

class Fuse_Feature_Module(nn.Module):
    def __init__(self,in_1):
        super(Fuse_Feature_Module, self).__init__()
        self.conv1 = nn.Conv2d(in_1*2,in_1,1,1,0)
        self.conv2 = nn.Conv2d(in_1,in_1,3,1,1)
        self.conv3 = convblock(in_1*2,in_1,3,1,1)
        self.conv4 = convblock(in_1*2,in_1*2,3,1,1)
        self.conv5 = convblock(in_1*4,in_1,3,1,1)
        self.conv6 = convblock(in_1*2,in_1,3,1,1)
        self.non_local = NonLocal(in_1)
    def forward(self,rgb,erase_r,high_t):
        e_r = self.conv2 ((rgb + erase_r )*rgb)
        ht1 = self.conv1(F.interpolate(high_t,rgb.shape[2:],mode='bilinear', align_corners=True))
        ht2 = self.conv6(F.interpolate(high_t,rgb.shape[2:],mode='bilinear', align_corners=True))
        fert = self.non_local(e_r,ht1)

        fer = self.conv4(torch.cat([fert ,ht2], dim=1))
        high_t = F.interpolate(high_t,rgb.shape[2:],mode='bilinear', align_corners=True)
        result = self.conv3(fer+high_t)

        return  result




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


if __name__ == '__main__':

    a = torch.randn(1,3,384,384)
    b = torch.randn(1,3,384,384)

    model = SwinNet()
    x1,x2= model(a, b)
    print(x1.shape)
    # model.load_state_dict(torch.load(r"D:\tanyacheng\Experiments\SOD\Transformer_Saliency\Swin\Saliency\Swin-Transformer-Saliency_v19\SwinTransNet_RGBD_cpts\SwinTransNet_epoch_best.pth"), strict=True)

