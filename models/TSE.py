from __future__ import absolute_import
from __future__ import division

import torch
import math
import numpy as np
from torch import nn
from torch.nn import functional as F


class TSE(nn.Module):
    def __init__(self, channel, use_gpu=False):
        super(TSE, self).__init__()
        self.layer4 = nn.Conv2d(channel, 1024, 1)

        self.branch0 = nn.Conv2d(1024, channel, 1)
        self.branch1 = nn.Conv2d(1024, channel, 1)
        self.block_size = 3
        self.in_channels = channel
        self.use_gpu = use_gpu

        self.conv_reduce = nn.Conv2d(channel, self.in_channels,
                kernel_size=1, stride=1, padding=0, bias=False)

        self.conv_erase = nn.Sequential(
            nn.Conv2d(self.in_channels, self.in_channels,
                    kernel_size=1, stride=1, padding=0, bias=True),
            #nn.BatchNorm2d(self.in_channels)
        )

        # init
        for m in [self.conv_reduce]:
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))

        for m in self.conv_erase:
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(0)
                m.bias.data.zero_()


    def correlation(self, x1, x2):
        """calculate the correlation map of x1 to x2
        """
        b, c, h, w = x2.size()

        x1 = F.avg_pool2d(x1, x1.size()[2:])
        x1 = self.conv_reduce(x1)
        x1 = x1.view(b, -1)

        x2 = x2.view(b, c, -1)
        f = torch.matmul(x1.view(b, 1, c), x2)

        f = f / np.sqrt(c)

        f = f.view(b, 1, h, w)
        return f


    def block_binarization(self, f):
        """
        generate the binary masks
        """
        soft_masks = f
        bs, t, h, w = f.size()
        f = torch.mean(f, 3)
        weight = torch.ones(1, 1, self.block_size, 1)
        if self.use_gpu: weight = weight.cuda()
        f = F.conv2d(input=f.view(-1, 1, h, 1),
                weight=weight,
                padding=(self.block_size//2, 0))

        if self.block_size % 2 == 0:
            f = f[:, :, :-1]
        index = torch.argmax(f.view(bs * t, h), dim=1)
        # generate the masks
        masks = torch.zeros(bs * t, h)
        if self.use_gpu: masks = masks.cuda()
        index_b = torch.arange(0, bs * t, dtype=torch.long)
        masks[index_b, index] = 1
        block_masks = F.max_pool2d(input=masks[:, None, :, None],
                                kernel_size=(self.block_size, 1),
                                stride=(1, 1),
                                padding=(self.block_size//2, 0))
        if self.block_size % 2 == 0:
            block_masks = block_masks[:, :, 1:]

        block_masks = 1 - block_masks.view(bs, t, h, 1)
        return block_masks, soft_masks


    def erase_feature(self, x, masks, soft_masks):
        """erasing the x with the masks, the softmasks for gradient back-propagation.
        """
        b, c, h, w = x.size()
        soft_masks = soft_masks - (1 - masks) * 1e8
        soft_masks = F.softmax(soft_masks.view(b, h * w) , 1)

        inputs = x * masks.unsqueeze(1)

        res = torch.bmm(x.view(b, c, h * w), soft_masks.unsqueeze(-1))

        outputs = inputs + self.conv_erase(res.unsqueeze(-1))

        return outputs


    def forward(self, fus, rgb):
        b, c,  h, w = fus.size()                                                                                         # t?
        m = torch.ones(b, h, w)
        if self.use_gpu: m = m.cuda()

        # forward the first frame with no erasing pixels
        #x0 = x[:, :, 0]
        #print(x0.shape)
        x0 = self.erase_feature(fus, m, m) # [b, c, h, w]
        y0 = self.layer4(x0)
        y0 = self.branch0(y0)
        #print(y0.shape)

        # generate the erased attention maps for second frames.
        f = self.correlation(y0.detach(), rgb)
        #print(f.shape)
        masks, soft_masks = self.block_binarization(f)
        #print('mask',masks.shape)
        #print('softmask', soft_masks.shape)
        masks, soft_masks = masks[:, 0], soft_masks[:, 0]

        # forward the second frame with saliency erasing
        #x1 = x[:, :, 1]
        #print(x1.shape)
        x1 = self.erase_feature(rgb, masks, soft_masks)
        y1 = self.layer4(x1)
        y1 = self.branch1(y1)

        return y0+y1


