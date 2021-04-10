import random
import numpy as np
from cv2 import cv2
import torch
import torch.utils.data as data
import utils.utils as util
import utils.bic as bic

class LQGTDataset(data.Dataset):
    '''
    Read LQ (Low Quality, here is LR) and GT image pairs.
    If only GT image is provided, generate LQ image on-the-fly.
    The pair is ensured by 'sorted' function, so please check the name convention.
    '''

    def __init__(self, opt):
        super(LQGTDataset, self).__init__()
        self.opt = opt
        self.paths_LQ, self.paths_GT = None, None

        # 从目录下获取所有图像的路径
        self.paths_GT = util.get_image_paths(opt['dataroot_GT']) if opt['dataroot_GT'] else None
        self.paths_LQ = util.get_image_paths(opt['dataroot_LQ']) if opt['dataroot_LQ'] else None
        assert self.paths_GT, 'Error: GT path is empty.'
        if self.paths_LQ and self.paths_GT:
            assert len(self.paths_LQ) == len(
                self.paths_GT
            ), 'GT and LQ datasets have different number of images - {}, {}.'.format(
                len(self.paths_LQ), len(self.paths_GT))
        # self.random_scale_list = [1]
        self.imgs_GT = []
        self.imgs_LQ = []
        for i in range(len(self.paths_GT)):
            GT_path = self.paths_GT[i]
            img_GT = util.read_img(GT_path)
            self.imgs_GT.append(img_GT)
            # print(img_GT.shape)
            if self.paths_LQ:
                LQ_path = self.paths_LQ[i]
                img_LQ = util.read_img(LQ_path)
                self.imgs_LQ.append(img_LQ)
                # print(img_LQ.shape)
            # print(i)


    def __getitem__(self, index):
        GT_path, LQ_path = None, None
        scale = self.opt['scale']
        GT_size = self.opt['GT_size']

        # get GT image
        GT_path = self.paths_GT[index]
        # img_GT_old = util.read_img(GT_path)
        img_GT = self.imgs_GT[index]
        # assert img_GT == img_GT_old
        if self.opt['phase'] != 'train':  # 如果是测试集/验证集
            img_GT = util.modcrop(img_GT, scale)
        # change color space if necessary
        if self.opt['color']:
            img_GT = util.channel_convert(img_GT.shape[2], self.opt['color'], [img_GT])[0]

        
        if self.paths_LQ:
            LQ_path = self.paths_LQ[index]
            # img_LQ_old = util.read_img(LQ_path)
            img_LQ = self.imgs_LQ[index]
            # assert img_LQ == img_LQ_old
        else:  # 如果未指定LR图像，则自动将HR图像进行Bicubic下采样

            # randomly scale during training
            if self.opt['phase'] == 'train':
                # random_scale = random.choice(self.random_scale_list)
                H_s, W_s, _ = img_GT.shape

                def _mod(n, scale, thres):
                    rlt = int(n)
                    rlt = (rlt // scale) * scale
                    return thres if rlt < thres else rlt

                H_s = _mod(H_s, scale, GT_size)
                W_s = _mod(W_s, scale, GT_size)
                img_GT = cv2.resize(np.copy(img_GT), (W_s, H_s), interpolation=cv2.INTER_LINEAR)
                # force to 3 channels
                if img_GT.ndim == 2:
                    img_GT = cv2.cvtColor(img_GT, cv2.COLOR_GRAY2BGR)

            H, W, _ = img_GT.shape
            # using matlab imresize
            img_LQ = bic.imresize_np(img_GT, 1 / scale, True)
            if img_LQ.ndim == 2:
                img_LQ = np.expand_dims(img_LQ, axis=2)

        if self.opt['phase'] == 'train':
            # if the image size is too small
            H, W, _ = img_GT.shape
            if H < GT_size or W < GT_size:
                img_GT = cv2.resize(np.copy(img_GT), (GT_size, GT_size),
                                    interpolation=cv2.INTER_LINEAR)
                # using matlab imresize
                img_LQ = bic.imresize_np(img_GT, 1 / scale, True)
                if img_LQ.ndim == 2:
                    img_LQ = np.expand_dims(img_LQ, axis=2)

            H, W, C = img_LQ.shape
            LQ_size = GT_size // scale

            # 随机裁剪固定大小(LQ_size)
            rnd_h = random.randint(0, max(0, H - LQ_size))
            rnd_w = random.randint(0, max(0, W - LQ_size))
            img_LQ = img_LQ[rnd_h:rnd_h + LQ_size, rnd_w:rnd_w + LQ_size, :]
            rnd_h_GT, rnd_w_GT = int(rnd_h * scale), int(rnd_w * scale)
            img_GT = img_GT[rnd_h_GT:rnd_h_GT + GT_size, rnd_w_GT:rnd_w_GT + GT_size, :]

            # 数据增广. 水平翻转或者旋转图像
            img_LQ, img_GT = util.augment([img_LQ, img_GT], self.opt['use_flip'],
                                          self.opt['use_rot'])

        # change color space if necessary
        if self.opt['color']:
            img_LQ = util.channel_convert(3, self.opt['color'],
                                          [img_LQ])[0]  # TODO during val no definition

        # BGR to RGB, HWC to CHW, numpy to tensor
        if img_GT.shape[2] == 3:
            img_GT = img_GT[:, :, [2, 1, 0]]
            img_LQ = img_LQ[:, :, [2, 1, 0]]
        img_GT = torch.from_numpy(np.ascontiguousarray(np.transpose(img_GT, (2, 0, 1)))).float()
        img_LQ = torch.from_numpy(np.ascontiguousarray(np.transpose(img_LQ, (2, 0, 1)))).float()

        if LQ_path is None:
            LQ_path = GT_path
        return {'LQ': img_LQ, 'GT': img_GT, 'LQ_path': LQ_path, 'GT_path': GT_path}

    def __len__(self):
        return len(self.paths_GT)
