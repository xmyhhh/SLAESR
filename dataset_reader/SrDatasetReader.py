'''
超分辨率数据库
'''

import os
import glob
import imageio
import cv2
import numpy as np
from torch.utils.data import Dataset, DataLoader


# 来自 sr_utils
def im_to_batch(im, sub_im_hw=(640, 640)):
    '''
    输入一张图像，输出分块图像序列
    :param im: np.array [h, w, c]
    :param sub_im_hw: 分块大小
    :return:
    '''
    ori_hw = im.shape[:2]
    h_c = int(np.ceil(ori_hw[0] / sub_im_hw[0]))
    w_c = int(np.ceil(ori_hw[1] / sub_im_hw[1]))
    sub_ims = []
    for h_i in range(h_c):
        for w_i in range(w_c):
            nim = im[sub_im_hw[0] * h_i: sub_im_hw[0] * (h_i + 1), sub_im_hw[1] * w_i: sub_im_hw[1] * (w_i + 1)]
            sub_ims.append(nim)
    return sub_ims, (h_c, w_c)


def padding_im_with_multiples_of_n(im, n=32, borderType=cv2.BORDER_CONSTANT, value=None):
    h_pad = im.shape[0] % n
    w_pad = im.shape[1] % n
    if h_pad > 0:
        h_pad = n - h_pad
    if w_pad > 0:
        w_pad = n - w_pad
    im = cv2.copyMakeBorder(im, 0, h_pad, 0, w_pad, borderType, value=value)
    return im, [h_pad, w_pad]


class SrDatasetReader(Dataset):
    def __init__(self, path=r'../datasets/faces', img_hw=(128, 128), iter_count=1000000):
        # assert mini_dataset >= 0, 'mini_dataset must be equal or big than 0'
        # self.use_random = use_random

        suffix = {'.jpg', '.png', '.bmp'}
        self.imgs_path = []
        # for p in glob.iglob('%s/**' % path, recursive=True):
        #     if os.path.splitext(p)[1].lower() in suffix:
        #         self.imgs_path.append(p)
        for p in os.listdir(path):
            if os.path.splitext(p)[1].lower() in suffix:
                self.imgs_path.append(os.path.join(path, p))
        # if mini_dataset > 0:
        #     np.random.shuffle(self.imgs_path)
        #     self.imgs_path = self.imgs_path[:mini_dataset]
        self.img_hw = img_hw
        self.iter_count = iter_count
        self.cache = []
        # self.random_horizontal_flip = random_horizontal_flip

    def __getitem__(self, _):
        # if self.use_random:

        # im = center_crop(im)
        # im = cv2.resize(im, self.img_hw, interpolation=cv2.INTER_CUBIC)

        # if self.random_horizontal_flip and np.random.uniform() > 0.5:
        #     im = np.array(im[:, ::-1])

        if len(self.cache) == 0:
            item = np.random.randint(0, len(self.imgs_path))

            impath = self.imgs_path[item]
            im = imageio.imread(impath)

            if im.ndim == 2:
                im = np.tile(im[..., None], (1, 1, 3))
            elif im.shape[-1] == 4:
                im = im[..., :3]

            # 根据边长比例，自动缩放
            # 缩放最大比例为1
            h_scale_min = self.img_hw[0] / im.shape[0]
            w_scale_min = self.img_hw[1] / im.shape[1]

            scale_percent = np.random.uniform(min(h_scale_min, w_scale_min, 1.), 1.)
            dst_wh = (int(im.shape[1] * scale_percent), int(im.shape[0] * scale_percent))
            im = cv2.resize(im, dst_wh, interpolation=cv2.INTER_AREA)

            # 填充，确保图像边长为要求图像大小的倍数
            h_pad = im.shape[0] % self.img_hw[0]
            w_pad = im.shape[1] % self.img_hw[1]
            if h_pad > 0:
                h_pad = self.img_hw[0] - h_pad
            if w_pad > 0:
                w_pad = self.img_hw[1] - w_pad
            im = cv2.copyMakeBorder(im, 0, h_pad, 0, w_pad, cv2.BORDER_CONSTANT, value=0.)

            ca = im_to_batch(im, self.img_hw)[0]
            self.cache.extend(ca)

        im = self.cache.pop(0)
        return im

    def __len__(self):
        # if self.use_random:
        #     return self.iter_count
        # else:
        return self.iter_count


if __name__ == '__main__':
    data = SrDatasetReader(r'../datasets/绮丽')
    for i in range(len(data)):
        a = data[i]
        print(a.shape)
        if a.shape != (128, 128, 3):
            raise AssertionError('img shape is not equal (3, 128, 128)')
        cv2.imshow('test', cv2.cvtColor(a, cv2.COLOR_RGB2BGR))
        cv2.waitKey(16)
