import os
import glob
import imageio
import cv2
import numpy as np
from torch.utils.data import Dataset, DataLoader


def center_crop(im):
    h, w = im.shape[:2]
    s0 = max(h, w)
    s1 = min(h, w)
    s2 = s1 // 2
    center_yx = h // 2, w // 2
    start_yx = (max(center_yx[0] - s2, 0), max(center_yx[1] - s2, 0))
    end_yx = (start_yx[0] + s1, start_yx[1] + s1)
    im = im[start_yx[0]:end_yx[0], start_yx[1]:end_yx[1]]
    return im


class NoTagFaceDatasetReader(Dataset):
    def __init__(self, path=r'../datasets/faces', img_hw=(128, 128), iter_count=1000000, random_horizontal_flip=False, mini_dataset=0, use_random=True):
        assert mini_dataset >= 0, 'mini_dataset must be equal or big than 0'
        self.use_random = use_random

        suffix = {'.jpg', '.png', '.bmp'}
        self.imgs_path = []
        for p in glob.iglob('%s/**' % path, recursive=True):
            if os.path.splitext(p)[1].lower() in suffix:
                self.imgs_path.append(p)
        if mini_dataset > 0:
            np.random.shuffle(self.imgs_path)
            self.imgs_path = self.imgs_path[:mini_dataset]
        self.img_hw = img_hw
        self.iter_count = iter_count
        self.random_horizontal_flip = random_horizontal_flip

    def __getitem__(self, item):
        if self.use_random:
            item = np.random.randint(0, len(self.imgs_path))

        impath = self.imgs_path[item]
        im = imageio.imread(impath)
        im = center_crop(im)
        im = cv2.resize(im, self.img_hw, interpolation=cv2.INTER_CUBIC)

        if self.random_horizontal_flip and np.random.uniform() > 0.5:
            im = np.array(im[:, ::-1])

        if im.ndim == 2:
            im = np.tile(im[..., None], (1, 1, 3))
        elif im.shape[-1] == 4:
            im = im[..., :3]

        return im

    def __len__(self):
        if self.use_random:
            return self.iter_count
        else:
            return len(self.imgs_path)


if __name__ == '__main__':
    data = NoTagFaceDatasetReader(r'../datasets/moeimouto-faces', random_horizontal_flip=True)
    for i in range(len(data)):
        a = data[i]
        print(a.shape)
        if a.shape != (3, 128, 128):
            raise AssertionError('img shape is not equal (3, 128, 128)')
        cv2.imshow('test', ((a.transpose([1,2,0]) + 1) / 2 * 255).astype(np.uint8)[:, :, ::-1])
        cv2.waitKey(16)
