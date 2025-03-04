import os
#set the device for training
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
import torch.nn.functional as F
import sys
import torch.nn as nn
import numpy as np
import argparse
import cv2
from models.token import SwinNet
from data import test_dataset
import time
from visual_graph import visactmap
#from Code.utils.data import get_loader


parser = argparse.ArgumentParser()
parser.add_argument('--testsize', type=int, default=384, help='testing size')
# parser.add_argument('--gpu_id', type=str, default='3', help='select gpu id')
parser.add_argument('--test_path', type=str, default='/media/data2/lcl_e/wkp/datasets/SOD/RGBT/', help='test dataset path')
opt = parser.parse_args()

dataset_path = opt.test_path

# set device for test
#set device for test
#if opt.gpu_id=='0':
#    os.environ["CUDA_VISIBLE_DEVICES"] = "3"
#    print('USE GPU 0')

# load the model
model = SwinNet()
model.cuda()

model.load_state_dict({k.replace('module.',''):v for k,v in torch.load('./cpts/tokenNet/Swin_epoch_best.pth', map_location='cuda:0').items()})#torch.load('./Checkpoint/TransMSOD/SPNet_epoch_best.pth')
model.eval()

# test
test_datasets = ['VT1000']#'VT2000_unalign', , 'VT5000-Test_unalign', 'VT1000_unalign', 'VT821_unalign']#['VT5000/Test', 'VT1000', 'VT821']#['VT5000-Test_unalign', 'VT1000_unalign', 'VT821_unalign']#['NLPR', 'STERE1000', 'SIP', 'NJUD', 'NLPR', 'DUT-RGBD', 'SSD']

#test_datasets = ['NLPR', 'STERE1000', 'SIP', 'NJUD', 'NLPR', 'DUT-RGBD', 'SSD']#['STERE1000', 'SIP', 'NJUD', 'NLPR', 'DUT-RGBD', 'SSD']
for dataset in test_datasets:
    save_path = './vis/out_semantic/' + dataset + '/'#'./output/'#'./test_maps_RGBT/' + dataset + '/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    
    image_root = dataset_path + dataset + '/RGB/'
    print(image_root)
    gt_root = dataset_path + dataset + '/GT/'
    depth_root = dataset_path + dataset + '/T/'
    vis_loader = test_dataset(image_root, gt_root, depth_root, testsize=384)
    visactmap(model, vis_loader, save_dir=save_path, width=384, height=384, use_gpu=True)
    
    
#    test_loader = test_dataset(image_root, gt_root, depth_root, opt.testsize)
#    img_num = len(test_loader)
#    time_s = time.time()
#    for i in range(test_loader.size):
#        image, gt, depth, name, image_for_post = test_loader.load_data()
#
#        gt = np.asarray(gt, np.float32)
#        gt /= (gt.max() + 1e-8)
#        image = image.cuda()
#        depth = depth.cuda()
#        pre_res = model(image, depth)
#        res = pre_res
#        res = F.upsample(res, size=gt.shape, mode='bilinear', align_corners=False)
#        res = res.sigmoid().data.cpu().numpy().squeeze()
#        res = (res - res.min()) / (res.max() - res.min() + 1e-8)
#
#        print('save img to: ', save_path + name)
#        cv2.imwrite(save_path + name, res * 255)
#    time_e = time.time()
#    print('speed: %f FPS' % (img_num / (time_e - time_s)))
    print('Test Done!')