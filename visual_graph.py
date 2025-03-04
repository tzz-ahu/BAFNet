# from utils import *
import warnings
# from model_source import Generator, Dis, Class, feature_extractor
warnings.filterwarnings('ignore')
import torch
import torch.nn as nn
import numpy as np
import cv2
import torch.nn.functional as F
import os.path as osp


# IMAGENET_MEAN = [0.485, 0.456, 0.406]
# IMAGENET_STD = [0.229, 0.224, 0.225]
IMAGENET_MEAN = [0.5, 0.5, 0.5]
IMAGENET_STD = [0.5, 0.5, 0.5]
GRID_SPACING = 10


@torch.no_grad()
def visactmap(
        model,
        test_loader,
        save_dir,
        width,
        height,
        use_gpu,
        img_mean=None,
        img_std=None
):
    if img_mean is None or img_std is None:
        # use imagenet mean and std
        img_mean = IMAGENET_MEAN 
        img_std = IMAGENET_STD

    model.eval()

    batch = 0
    img_num = len(test_loader)
    print(img_num)
    for i in range(test_loader.size):#for batch_idx, data in enumerate(test_loader):
        batch = batch + 1
        images, gts, depths, name, _ = test_loader.load_data()#image, gt, depth, name, image_for_post = test_loader.load_data()
        #print(images.shape)
        #print(gts.shape)
        #print(depths.shape)
        #print("#########")
        if use_gpu:
            images = images.cuda()
            #gts = gts.cuda()
            #depths = depths.cuda()
            depths = depths.repeat(1, 3, 1, 1).cuda()
        # forward to get convolutional feature maps
        try:
            
            #outputs = model(audio, imgs, imgs2, return_featuremaps=True)
            _, _, out = model(images, depths)################out, fr[3], fuseed_rt4, aligned_t4, att_4_t, att_4_r, ft[3], visatt4r, visatt4t, our_r, out_t, semantic
        except TypeError:
            raise TypeError(
                'forward() got unexpected keyword argument "return_featuremaps". '
                'Please add return_featuremaps as an input argument to forward(). When '
                'return_featuremaps=True, return feature maps only.'
            )


        if out.dim() != 4:
            raise ValueError(
                'The model output is supposed to have '
                'shape of (b, c, h, w), i.e. 4 dimensions, but got {} dimensions. '
                'Please make sure you set the model output at eval mode '
                'to be the last convolutional feature maps'.format(
                    out.dim()
                )
            )

        # compute activation maps
        out = (out ** 2).sum(1)
        b, h, w = out.size()
        out = out.view(b, h * w)
        out = F.normalize(out, p=2, dim=1)
        out = out.view(b, h, w)
        

        if use_gpu:
            images, out = images.cpu(), out.cpu()
            #audio, outputs = imgs.cpu(), outputs.cpu()
            
        #print(out.shape)
        #print("###")
        for j in range(out.size(0)):
            #print("j:")
            #print(j)

            #imnameWri = str(batch) + '_' + str(j)
            # RGB image
            #print(images.shape)
            #print("@@@")
            images_j = images[j, ...]
            #print(images.shape)
            for t, m, s in zip(images_j, img_mean, img_std):
                t.mul_(s).add_(m).clamp_(0, 1)
            img_np = np.uint8(np.floor(images_j.numpy() * 255))
            #img_np = np.uint8(np.floor(img.numpy()))
            
            img_np = img_np.transpose((1, 2, 0))  # (c, h, w) -> (h, w, c)

            # activation map
            am = out[j, ...].numpy()
            am = cv2.resize(am, (width, height))
            am = 255 * (am - np.min(am)) / (
                    np.max(am) - np.min(am) + 1e-12
            )
            am = np.uint8(np.floor(am))
            am = cv2.applyColorMap(am, cv2.COLORMAP_JET)
            cv2.imwrite(osp.join(save_dir, name + 'am.png'), am)#vis_map_our #r"/media/data2/lcl_e/wkp
            # overlapped
            overlapped = img_np * 0.3 + am * 0.7
            overlapped[overlapped > 255] = 255
            overlapped = overlapped.astype(np.uint8)
            cv2.imwrite(osp.join(save_dir, name + 'overlapped.png'), overlapped)
            # save images in a single figure (add white spacing between images)
            # from left to right: original image, activation map, overlapped image
            grid_img = 255 * np.ones(
                (height, 3 * width + 2 * GRID_SPACING, 3), dtype=np.uint8
            )
            grid_img[:, :width, :] = img_np[:, :, ::-1]
            grid_img[:,
            width + GRID_SPACING:2 * width + GRID_SPACING, :] = am
            grid_img[:, 2 * width + 2 * GRID_SPACING:, :] = overlapped
            cv2.imwrite(osp.join(save_dir, name + '.png'), grid_img)#vis_map_our #r"/media/data2/lcl_e/wkp/code/SOD2/T/TransMSOD14_swin_1_T/vis/"
            
            # print(osp.join(r"E:\visCAM", imnameWri + '.jpg'))

        if (i + 1) % 10 == 0:
            print(
                '- done batch {}/{}'.format(
                    i + 1, len(test_loader)
                )
            )

def eval_CAM(feature, generator):
    # train_data = train_data_V2F_map()#
    train_data = test_data_V2F_map()
    data_loader = DataLoader(train_data, batch_size=250, shuffle=False, num_workers=0)


    # sampler_R = 'RandomIdentitySampler'
    # MyDataSet = val_vis_map_data_V2F()
    # faceroot= './train_retrival_8fv_label64.txt'#train_face_voice_gallery
    # train_dataset=[]
    # with open(faceroot, 'r') as f:
    #     for line in f:
    #         strr = line.split()
    #         item = {
    #             "image_path": strr[0],
    #             "voice_path": strr[1],
    #             "id": strr[2],
    #         }
    #         train_dataset.append(item)

    # sampler = getattr(samplers, sampler_R)(train_dataset, batch_size=128, num_instances=8)
    # data_loader = DataLoader(MyDataSet, sampler=sampler, batch_size=128,
    #                           num_workers=0, pin_memory=True)

    #val_map = val_vis_map_data_V2F()
    #data_loader = DataLoader(val_map, batch_size=128, shuffle=False, num_workers=2)
    feature.eval()
    generator.eval()

    # visactmap(
    #    #feature, data_loader, r"E:\Debug_code\V2F_focus\vis_map_audio", 224, 224, use_gpu=True
    #    feature, data_loader, r"C:\Users\wangjx\Desktop\V2F-baseline\vis-map_baseline_train", 224, 224, use_gpu=True)#vis_map_our

    visactmap_F(
        #feature, data_loader, r"E:\Debug_code\V2F_focus\vis_map_audio", 224, 224, use_gpu=True
        #feature, data_loader, r"E:\Debug_code\V2F_focus\vis_map_val", 224, 224, use_gpu=True)#Assignment
        #feature, data_loader, r"E:\Debug_code\V2F_focus\vis_Assignment", 224, 224, use_gpu=True)#Assignment
        #feature, generator, data_loader, r"E:\Debug_code\V2F_focus\tsne", 224, 224, use_gpu=True)#E:\Debug_code\V2F_focus\our_generator
        feature, generator, data_loader, r"C:\Users\wangjx\Desktop\V2F-baseline\graph_s", 224, 224, use_gpu=True)#our_generator

    raise RuntimeError

def showPointSingleModal_one(features, label, save_path):
    # label = self.relabel(label)
    N,C,H,W = features.size()
    feature = features.view(N,-1)
    tsne = manifold.TSNE(n_components=2, init='pca', random_state=501)
    feature = feature.cpu().numpy()
    features_tsne = tsne.fit_transform(feature)
    COLORS = ['SlateBlue', 'DeepPink', 'magenta', 'red', 'blue', 'black', 'Olive', 'SeaGreen',
            'yellow', 'Lime', 'pink', 'Orange', 'Orchid', 'Purple', 'SlateGray', 'Red', 'PaleTurquoise',
            'Teal', 'Honeydew', 'c']
    #COLORS = ['darkorange', 'limegreen', 'royalblue', 'red', 'darkviolet', 'black']
    MARKS = ['x', 'o', '+', '^', 's']
    features_min, features_max = features_tsne.min(0), features_tsne.max(0)
    features_norm = (features_tsne - features_min) / (features_max - features_min)
    plt.figure(figsize=(20, 20))
    for i in range(features_norm.shape[0]):
        plt.scatter(features_norm[i, 0], features_norm[i, 1], s=5, color=COLORS[label[i] % 5],
                    marker=MARKS[label[i] % 5])
        #plt.scatter(features_norm[i, 0], features_norm[i, 1], s=60, color=COLORS[label[i] % 6],
        #            marker=MARKS[label[i] % 5])
    plt.savefig(save_path)
    plt.show()
    plt.close()

def get_L2norm_loss_self_driven(x):
    #l = (x.norm(p=2, dim=1).mean() - 25) ** 2
    #x = x - x.mean(1).repeat(1,2)
    norm = (sum(sum(x**2,1)))**(1/2)/x.size
    x_new = x/norm
    return x_new

def showPointTwoModal(features1, features2, labels, batch_idx, save_path):
    #N,C,H,W = features1.size()
    #features1 = torch.squeeze(features1)
    #features2 = torch.squeeze(features2)
    #features1 = features1.view(C,-1)
    #features2 = features2.view(C,-1)
    N,C = features1.size()
    #features1 = features1.T
    #features2 = features2.T
    #label1 = label1.repeat([C, 1])
    tsne = manifold.TSNE(n_components=2, init='pca', random_state=501)
    features1 = features1.cpu().numpy()
    features2 = features2.cpu().numpy()
    #label = labels.cpu().numpy()
    unique_label = torch.unique(labels)
    #features1_norm = np.zeros_like(torch.zeros(N,2))
    #features2_norm = np.zeros_like(torch.zeros(N,2))
    features1_tsne = np.zeros_like(torch.zeros(N,2))
    features2_tsne = np.zeros_like(torch.zeros(N,2))
    for cls in unique_label:
        cls = cls.item()
        cls_inds = torch.nonzero(labels == cls).squeeze(1) 
        features1_tsne[cls_inds,:]  = tsne.fit_transform(features1[cls_inds])
        features2_tsne[cls_inds,:] = tsne.fit_transform(features2[cls_inds])
        #features1_norm = features1
        #features2_norm = features2
        #features1_norm[cls_inds,:] = get_L2norm_loss_self_driven(features1_tsne)
        #features2_norm[cls_inds,:] = get_L2norm_loss_self_driven(features2_tsne)
        features1_min, features1_max = features1_tsne.min(0), features1_tsne.max(0)
        features1_norm= (features1_tsne - features1_min) / (features1_max - features1_min)
        features2_min, features2_max = features2_tsne.min(0), features2_tsne.max(0)
        features2_norm = (features2_tsne - features2_min) / (features2_max - features2_min)
    grid_img = plt.figure(figsize=(20, 20))

    COLORS = ['SlateBlue', 'DeepPink', 'magenta', 'red', 'blue', 'black', 'Olive', 'SeaGreen', 'c',
              'yellow', 'Lime', 'pink', 'Orange', 'Orchid', 'Purple', 'SlateGray', 'Red', 'PaleTurquoise',
              'Teal', 'Honeydew']
    MARKS = [ 'o', '+','x', '^', 's']
    for i in range(len(features1_norm)):
        plt.scatter(features1_norm[i, 0], features1_norm[i, 1], s=960, color=COLORS[labels[i] % 20],
                    marker=MARKS[0])#int(label1[i][0])
        plt.scatter(features2_norm[i, 0], features2_norm[i, 1], s=960, color=COLORS[labels[i] % 20],
                    marker=MARKS[1])
        #plt.scatter(features1_norm[i, 0], features1_norm[i, 1], s=480, color=COLORS[(labels[i]) % 20],
        #            marker=MARKS[0])#int(label1[i][0])
        #plt.scatter(features2_norm[i, 0], features2_norm[i, 1], s=480, color=COLORS[labels[i] % 20],
        #            marker=MARKS[1])
    imnameWri = str(batch_idx) + '_' + str(labels[i])
    #cv2.imwrite(osp.join(r"E:\Debug_code\V2F_focus\vis_map_our", imnameWri + '.jpg'), grid_img)
    plt.savefig(osp.join(save_path, imnameWri + '.jpg'))#E:\Debug_code\V2F_focus\vis_Assignment
    plt.show()
    plt.close()


"""
def showPointSingleModal(features1, label1, features2, label2, save_path):
    # label = self.relabel(label)
    tsne = manifold.TSNE(n_components=2, init='pca', random_state=501)
    features1_tsne = tsne.fit_transform(features1)
    features2_tsne = tsne.fit_transform(features2)
    COLORS = ['darkorange', 'limegreen', 'royalblue', 'red', 'darkviolet', 'black']
    MARKS = ['x', 'o', '+', '^', 's']
    features1_min, features1_max = features1_tsne.min(0), features1_tsne.max(0)
    features1_norm = (features1_tsne - features1_min) / (features1_max - features1_min)
    features2_min, features2_max = features2_tsne.min(0), features2_tsne.max(0)
    features2_norm = (features2_tsne - features2_min) / (features2_max - features2_min)

    plt.figure(figsize=(20, 20))
    for i in range(features1_norm.shape[0]):
        plt.scatter(features1_norm[i, 0], features1_norm[i, 1], s=60, color=COLORS[label1[i] % 6],
                    marker=MARKS[label1[i] % 5])
    for i in range(features2_norm.shape[0]):
        plt.scatter(features2_norm[i, 0], features2_norm[i, 1], s=60, color=COLORS[label2[i] % 6],
                    marker=MARKS[label2[i] % 5])
    plt.savefig(save_path)
    plt.show()
    plt.close()
"""

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
GRID_SPACING = 10


@torch.no_grad()
def visactmap_F(
        model,
        generator,
        test_loader,
        save_dir,
        width,
        height,
        use_gpu,
        img_mean=None,
        img_std=None
):
    if img_mean is None or img_std is None:
        # use imagenet mean and std
        img_mean = IMAGENET_MEAN
        img_std = IMAGENET_STD

    model.eval()
    generator.eval()

    batch = 0
    #labels = torch.Tensor().float().to('cuda')
    #features = torch.FloatTensor().to('cuda')
    Similarity=[]
    dist = nn.CosineSimilarity(dim=0)
    #euc_dist = euclidean_dist()
    for batch_idx, data in enumerate(test_loader):
        batch = batch + 1
        audio, imgs, imgs2, label = data[0], data[1], data[2], data[3]
        if use_gpu:
            audio = audio.cuda()
            imgs = imgs.cuda()
            imgs2 = imgs2.cuda()
            label = label.cuda()
            

        # audio, imgs, labels = data[0], data[1], data[2]
        #imgs, labels = data[0], data[1]
        #label = label.float()
        # if use_gpu:
        #     imgs = imgs#.cuda()
            #audio = audio.cuda()

        # forward to get convolutional feature maps
        try:
            outputs2, outputs, outputs3 = model(audio, imgs, imgs2)
            outputs2, outputs, outputs3 = generator(outputs2, outputs, outputs3)
            # outputs = model(imgs)
            #outputs1, outputs= model(audio, imgs, return_featuremaps=True)
            #audio_F, imgs_F = model(audio, imgs)
            #outputs_V, outputs_F = generator(audio_F, imgs_F)
        except TypeError:
            raise TypeError(
                'forward() got unexpected keyword argument "return_featuremaps". '
                'Please add return_featuremaps as an input argument to forward(). When '
                'return_featuremaps=True, return feature maps only.'
            )
        #
        #simil = dist(outputs_V.T, outputs_F.T) #self.dist(center1, center2)
        #Similarity.append(simil)
        #dist = euclidean_dist(outputs_V, outputs_F)
        #Similarity.append(dist)
        #name = 'V2F' + '_'  + 'Similarity_our'  + '.pkl'
        #torch.save(Similarity, name)

        #features = torch.cat([features, outputs], 0)
        #label = torch.unsqueeze(label, 0)
        #labels = torch.cat([labels, label], 1) 
        #labels = torch.cat([label_V, label_F], 0) 
        #features = torch.cat([outputs_V, outputs_F], 0)
        showPointSingleModal_one(outputs, label, save_dir)
        #showPointTwoModal(outputs_V, outputs_F,labels, batch_idx, save_dir)
        #showPointSingleModal(features, labels, save_dir)




def load_f(feature, G):
    states = torch.load('V2F33_0.9594818652849741.pkl')#
    # states = torch.load('V2F20_0.9636269430051814.pkl')#V2F33_0.9594818652849741.pkl
    feature.load_state_dict(states['feature'])
    G.load_state_dict(states['G'])
    #D.load_state_dict(states['D'])
    #C.load_state_dict(states['C'])
    return feature, G#, D, C

def test():
    feature = feature_extractor()
    #feature = feature_only_F()
    #feature = feature_retrieval()#
    generator = Generator()
    #generator = Generator_our()
    cuda = True if torch.cuda.is_available() else False
    #feature = load_retrieval(feature)
    feature, generator = load_f(feature, generator)

    if cuda:
        feature = feature.to('cuda')
        generator =generator.to('cuda')

    acc_best = eval_CAM(feature,generator)

if __name__ == '__main__':
    seed = 25
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    #train()
    #eval()
    test()