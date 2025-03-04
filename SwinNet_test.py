# -*- coding: utf-8 -*-
"""
@author: caigentan@AnHui University
@software: PyCharm
@file: SwinNet_test.py
@time: 2021/5/31 09:34
"""

import torch
import torch.nn.functional as F
import sys
sys.path.append('./models')
import numpy as np
import os, argparse
import cv2
from models.ab_tse import SwinNet
from data import test_dataset
import time
import torch.nn as nn

parser = argparse.ArgumentParser()
parser.add_argument('--testsize', type=int, default=384, help='testing size')
parser.add_argument('--gpu_id', type=str, default='7', help='select gpu id')
parser.add_argument('--test_path',type=str,default='/DATA/wanghongshun/',help='test dataset path')
# parser.add_argument('--test_path',type=str,default='./datasets/RGB-T/Test/',help='test dataset path')
opt = parser.parse_args()

dataset_path ="/media/data2/lcl_e/wkp/datasets/SOD/RGBT/VT5000-Test/"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"



#set device for test
#if opt.gpu_id=='7':
 #   os.environ["CUDA_VISIBLE_DEVICES"] = "7"
 #   print('USE GPU 5')
#elif opt.gpu_id=='3':
 #   os.environ["CUDA_VISIBLE_DEVICES"] = "3"
 #   print('USE GPU 3')

#load the model
model = SwinNet()
#Large epoch size may not generalize well. You can choose a good model to load according to the log file and pth files saved in ('./BBSNet_cpts/') when training.
model.load_state_dict(torch.load("./cpts/ab_tse/Swin_epoch_best.pth"))

model.cuda()
model.eval()
fps = 0
    
USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda:0" if USE_CUDA else "cpu")
device_ids=[0,1]
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model, device_ids)
        
if torch.cuda.is_available():
    model = model.to(device)
        
# RGBT-Test
test_datasets = ['VT5000']
for dataset in test_datasets:
    time_s = time.time()
    sal_save_path = './saliency_map/' + dataset + '/'
    #edge_save_path = './test_maps/SwinTransNet_RGBT2_best/Edge/' + dataset + '/'
    if not os.path.exists(sal_save_path):
        os.makedirs(sal_save_path)
        # os.makedirs(edge_save_path)
    image_root = dataset_path +  '/RGB/'
    gt_root = dataset_path + '/GT/'
    depth_root = dataset_path + '/T/'
    test_loader = test_dataset(image_root, gt_root, depth_root, opt.testsize)
    nums = test_loader.size
    for i in range(test_loader.size):
        image, gt, depth, name, image_for_post = test_loader.load_data()
        gt = np.asarray(gt, np.float32)
        gt /= (gt.max() + 1e-8)
        image = image.cuda()
        # depth = depth = depth.repeat(1,3,1,1).cuda()
        #depth = depth.cuda()
        depth = depth.repeat(1, 3, 1, 1).cuda()


        # print(depth.shape)
        res,e = model(image,depth)
        res = F.upsample(res, size=gt.shape, mode='bilinear', align_corners=False)
        # edge = F.upsample(edge, size=gt.shape, mode='bilinear', align_corners=False)
        res = res.sigmoid().data.cpu().numpy().squeeze()
        # edge = edge.sigmoid().data.cpu().numpy().squeeze()
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)
        # edge = (edge - edge.min()) / (edge.max() - edge.min() + 1e-8)
        print('save img to: ', sal_save_path + name)
        cv2.imwrite(sal_save_path + name, res * 255)
    time_e = time.time()
    fps += (nums / (time_e - time_s))
    print("FPS:%f" % (nums / (time_e - time_s)))
    print('Test Done!')
print("Total FPS %f" % fps) # this result include I/O cost

# test
'''
test_datasets = ['SIP','SSD','RedWeb','NJU2K','NLPR','STERE','DES','LFSD',]
# test_datasets = ['DUT-RGBD']
fps = 0
for dataset in test_datasets:
    time_s = time.time()
    save_path = './test_maps/' + dataset + '/'
    edge_save_path = './test_maps/' + dataset + '/edge/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if not os.path.exists(edge_save_path):
        os.makedirs(edge_save_path)
    image_root = dataset_path + dataset + '/RGB/'
    gt_root = dataset_path + dataset + '/GT/'
    depth_root = dataset_path + dataset + '/depth/'
    test_loader = test_dataset(image_root, gt_root, depth_root, opt.testsize)
    nums = test_loader.size
    for i in range(test_loader.size):
        image, gt, depth, name, image_for_post = test_loader.load_data()
        gt = np.asarray(gt, np.float32)
        gt /= (gt.max() + 1e-8)
        image = image.cuda()
        depth = depth.repeat(1,3,1,1).cuda()
        res, edge = model(image,depth)
        res = F.upsample(res, size=gt.shape, mode='bilinear', align_corners=False)
        edge = F.upsample(edge, size=gt.shape, mode='bilinear', align_corners=False)
        res = res.sigmoid().data.cpu().numpy().squeeze()
        edge = edge.sigmoid().data.cpu().numpy().squeeze()
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)
        edge = (edge - edge.min()) / (edge.max() - edge.min() + 1e-8)
        print('save img to: ',save_path+name)
        cv2.imwrite(save_path + name, res*255)
        cv2.imwrite(edge_save_path + name, edge * 255)
    time_e = time.time()
    fps += (nums / (time_e - time_s))
    print("FPS:%f" % (nums / (time_e - time_s)))
    print('Test Done!')
print("Total FPS %f" % fps)

'''