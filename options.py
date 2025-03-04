# -*- coding: utf-8 -*-
"""
@author: caigentan@AnHui University
@software: PyCharm
@file: options.py
@time: 2021/5/16 14:52
"""

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=300, help='epoch number')
parser.add_argument('--lr', type=float, default=5e-5, help='learning rate')
parser.add_argument('--batchsize', type=int, default=4, help='training batch size')
parser.add_argument('--trainsize', type=int, default=384, help='training dataset size')
parser.add_argument('--clip', type=float, default=0.5, help='gradient clipping margin')
parser.add_argument('--decay_rate', type=float, default=0.1, help='decay rate of learning rate')
parser.add_argument('--decay_epoch', type=int, default=30, help='every n epochs decay learning rate')
# parser.add_argument('--load', type=str, default='./pre_train/swin_base_patch4_window7_224.pth', help='train from checkpoints')
parser.add_argument('--load', type=str, default='/media/data2/lcl_e/wkp/pre_models/swin_base_patch4_window12_384_22k.pth', help='train from checkpoints')
parser.add_argument('--gpu_id', type=str, default='1', help='train use gpu')

# RGB-D Datasets
# parser.add_argument('--rgb_root', type=str, default='./datasets/train/RGB/', help='the training RGB images root')
# parser.add_argument('--depth_root', type=str, default='./datasets/train/Depth/', help='the training Depth images root')
# parser.add_argument('--gt_root', type=str, default='./datasets/train/GT/', help='the training GT images root')
# parser.add_argument('--edge_root', type=str, default='./datasets/train/Edge/', help='the training Edge images root')
# parser.add_argument('--val_rgb_root', type=str, default='./datasets/RGB-D/validation/RGB/', help='the validation RGB images root')
# parser.add_argument('--val_depth_root', type=str, default='./datasets/RGB-D/validation/Depth/', help='the validation Depth images root')
# parser.add_argument('--val_gt_root', type=str, default='./datasets/RGB-D/validation/GT/', help='the test validation GT images root')

# RGB-T Datasets
parser.add_argument('--rgb_root', type=str, default='/media/data2/lcl_e/wkp/datasets/SOD/RGBT/VT5000/Train/RGB/', help='the training RGB images root')
parser.add_argument('--depth_root', type=str, default='/media/data2/lcl_e/wkp/datasets/SOD/RGBT/VT5000/Train/T/', help='the training Thermal images root')
parser.add_argument('--gt_root', type=str, default='/media/data2/lcl_e/wkp/datasets/SOD/RGBT/VT5000/Train/GT/', help='the training GT images root')
parser.add_argument('--edge_root', type=str, default='/media/data2/lcl_e/wkp/datasets/SOD/RGBT/VT5000/Train/T/', help='the training Edge images root')

parser.add_argument('--val_rgb_root', type=str, default='/media/data2/lcl_e/wkp/datasets/SOD/RGBT/VT5000/Test/RGB/', help='the validation RGB images root')
parser.add_argument('--val_depth_root', type=str, default='/media/data2/lcl_e/wkp/datasets/SOD/RGBT/VT5000/Test/T/', help='the validation Thermal images root')
parser.add_argument('--val_gt_root', type=str, default='/media/data2/lcl_e/wkp/datasets/SOD/RGBT/VT5000/Test/GT/', help='the validation GT images root')

parser.add_argument('--save_path', type=str, default='./cpts/ab_tse/', help='the path to save models and logs')

opt = parser.parse_args()