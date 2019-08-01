from model_utils_torch import *
import imageio


def im_arr_to_tensor(arr, low=-1., high=1.):
    arr = np.asarray(arr, np.float32)
    arr = (arr / 255.) * (high - low) + low
    arr = np.transpose(arr, [0, 3, 1, 2])
    arr = torch.tensor(arr)
    return arr


def tensor_to_im_arr(t: torch.Tensor, low=-1., high=1.):
    t = (t.clamp(low, high) - low) / (high - low)
    t = t * 255.
    t = t.permute(0, 2, 3, 1).cpu().numpy()
    t = np.asarray(t, np.uint8)
    return t


# 将一堆图像贴到一张大图上
def make_figure_img(imgs, hw=(9, 9), path=None):

        n = int(np.prod(hw))

        img_hw = imgs[0].shape[0:2]
        img_depth = imgs[0].shape[2]

        figure = np.zeros((img_hw[0] * hw[0], img_hw[1] * hw[1], img_depth), dtype=np.uint8)

        for i in range(n):
            if len(imgs) == i:
                break
            row = i // hw[1]
            col = i % hw[1]

            figure[row*img_hw[0]:(row+1)*img_hw[0], col*img_hw[1]:(col+1)*img_hw[1]] = imgs[i]

        imageio.imwrite(path, figure)
        return figure


# 渐变测试
def sample_with_interpolation(dnet: nn.Module, path, start_latent, end_latent, step, hw=(9, 9), batch_size=1):
    torch.backends.cudnn.benchmark = False
    with torch.no_grad():

        n = int(np.prod(hw))

        dnet.eval()

        imgs = []

        # for ori_img in ori_imgs:
        #     im = gnet(enet(ori_img.unsqueeze(0)) / 2).cpu().numpy().transpose((0, 2, 3, 1))[0]
        #     imgs.append(im)

        n_batch = int(np.ceil(len(ori_imgs) / batch_size))
        for b_id in range(n_batch):
            ori_imgs_batch = ori_imgs[b_id*batch_size: (b_id+1)*batch_size]
            im = dnet(enet(ori_imgs_batch))
            imgs.append(im)

        imgs = torch.cat(imgs, 0).permute(0, 2, 3, 1).cpu().numpy()

        img_hw = imgs[0].shape[0:2]
        img_depth = imgs[0].shape[2]

        figure = np.zeros((img_hw[0] * hw[0], img_hw[1] * hw[1], img_depth))

        for i in range(n):
            if len(imgs) == i:
                break
            row = i // hw[1]
            col = i % hw[1]

            figure[row*img_hw[0]:(row+1)*img_hw[0], col*img_hw[1]:(col+1)*img_hw[1]] = imgs[i]

        figure = (figure + 1) / 2 * 255
        figure = np.round(figure).clip(0, 255).astype(np.uint8)
        imageio.imwrite(path, figure)

    torch.backends.cudnn.benchmark = True


# 采样函数
def sample_with_label(enet: nn.Module, dnet: nn.Module, path, ori_imgs, hw=(9, 9), batch_size=1):
    torch.backends.cudnn.benchmark = False
    with torch.no_grad():

        n = int(np.prod(hw))

        enet.eval()
        dnet.eval()

        imgs = []

        # for ori_img in ori_imgs:
        #     im = gnet(enet(ori_img.unsqueeze(0)) / 2).cpu().numpy().transpose((0, 2, 3, 1))[0]
        #     imgs.append(im)

        n_batch = int(np.ceil(len(ori_imgs) / batch_size))
        for b_id in range(n_batch):
            ori_imgs_batch = ori_imgs[b_id*batch_size: (b_id+1)*batch_size]
            im = dnet(enet(ori_imgs_batch))
            imgs.append(im)

        imgs = torch.cat(imgs, 0).permute(0, 2, 3, 1).cpu().numpy()

        img_hw = imgs[0].shape[0:2]
        img_depth = imgs[0].shape[2]

        figure = np.zeros((img_hw[0] * hw[0], img_hw[1] * hw[1], img_depth))

        for i in range(n):
            if len(imgs) == i:
                break
            row = i // hw[1]
            col = i % hw[1]

            figure[row*img_hw[0]:(row+1)*img_hw[0], col*img_hw[1]:(col+1)*img_hw[1]] = imgs[i]

        figure = (figure + 1) / 2 * 255
        figure = np.round(figure).clip(0, 255).astype(np.uint8)
        imageio.imwrite(path, figure)

    torch.backends.cudnn.benchmark = True


# 采样函数
def sample_with_2label(enet: nn.Module, dnet: nn.Module, path, ori_imgs, hw=(9, 9), batch_size=1):
    torch.backends.cudnn.benchmark = False
    with torch.no_grad():

        n = int(np.prod(hw))

        enet.eval()
        dnet.eval()

        imgs = []

        # for ori_img in ori_imgs:
        #     im = gnet(enet(ori_img.unsqueeze(0)) / 2).cpu().numpy().transpose((0, 2, 3, 1))[0]
        #     imgs.append(im)

        n_batch = int(np.ceil(len(ori_imgs) / batch_size))
        for b_id in range(n_batch):
            ori_imgs_batch = ori_imgs[b_id*batch_size: (b_id+1)*batch_size]
            im = dnet(*enet(ori_imgs_batch))
            imgs.append(im)

        imgs = torch.cat(imgs, 0).permute(0, 2, 3, 1).cpu().numpy()

        img_hw = imgs[0].shape[0:2]
        img_depth = imgs[0].shape[2]

        figure = np.zeros((img_hw[0] * hw[0], img_hw[1] * hw[1], img_depth))

        for i in range(n):
            if len(imgs) == i:
                break
            row = i // hw[1]
            col = i % hw[1]

            figure[row*img_hw[0]:(row+1)*img_hw[0], col*img_hw[1]:(col+1)*img_hw[1]] = imgs[i]

        figure = (figure + 1) / 2 * 255
        figure = np.round(figure).clip(0, 255).astype(np.uint8)
        imageio.imwrite(path, figure)

    torch.backends.cudnn.benchmark = True


def next_data(dataloader):
    for imgs in dataloader:
        yield im_arr_to_tensor(imgs)
    return None


def next_data2(dataloader):
    for imgs in dataloader:
        # s = np.random.uniform(0.3, 1.)
        s = torch.rand(imgs.shape[0], 1, 1, 1) * 0.9 + 0.1
        s = torch.where(s > 0.95, torch.ones_like(s), s)
        imgs: torch.Tensor = imgs.float() * s / 255. * 2 - 1
        imgs = imgs.permute(0, 3, 1, 2)
        yield imgs
    return None
