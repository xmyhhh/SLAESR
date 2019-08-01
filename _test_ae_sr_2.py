import os
import time

from auto_encode_sr_net2 import GenNet
from auto_encode_sr_net2 import Encoder

from model_utils_torch import *
from ae_utils import *
from sr_utils import *


# 图像边长不固定，使用这个反而会慢
# torch.backends.cudnn.benchmark = True


if __name__ == '__main__':

    torch.set_grad_enabled(False)

    example_id = '2'

    model_file = 'model_{}.pt'.format(example_id)

    sr_src_dir = 'sr_src'
    sr_dst_dir = 'sr_dst_{}'.format(example_id)

    os.makedirs(sr_src_dir, exist_ok=True)
    os.makedirs(sr_dst_dir, exist_ok=True)

    sub_im_hw = (512, 512)

    enet = Encoder()
    gnet = GenNet()

    has_cuda = False

    if torch.cuda.is_available():
        has_cuda = True
        enet.cuda()
        gnet.cuda()

    try:
        model = None
        model = torch.load(model_file)
        enet.load_state_dict(model['enet'])
        gnet.load_state_dict(model['gnet'])
        print('Load model success')

    except (FileNotFoundError, RuntimeError):
        print('Not found save model')

    finally:
        del model

    print('enet params')
    print_params_size(enet.parameters())
    print('gnet params')
    print_params_size(gnet.parameters())

    # 不使用glob，一些特殊字符的文件名扫不出来
    # dataset = glob.glob(sr_src_dir+'/*.jpg')
    # dataset += glob.glob(sr_src_dir+'/*.png')
    #
    dataset = []
    for p in os.listdir(sr_src_dir):
        if os.path.splitext(p)[1] in ['.jpg', '.png', '.bmp']:
            dataset.append(os.path.join(sr_src_dir, p))

    ts1 = time.time()

    torch.cuda.empty_cache()

    gnet.eval()
    enet.eval()

    for i, im_path in enumerate(dataset):
        print('{}/{}'.format(i, len(dataset)), im_path)
        dst_im_path = os.path.join(sr_dst_dir, os.path.basename(im_path))

        im = imageio.imread(im_path)

        im, hw_pad = padding_im_with_multiples_of_n(im, 32)

        # 确保图像通道数为3
        if im.ndim == 2:
            im = np.tile(im[:, :, None], [1, 1, 3])
        if im.shape[2] == 4:
            im = im[:, :, :3]

        sub_ims, im_hw_c, ori_im_hw = im_to_batch_with_seamless(im, sub_im_hw)
        # nim = batch_to_im_with_no_seamless(sub_ims, im.shape[:2], im_hw_c, (640, 640))

        # 将大图分块，大图没办法一次处理完
        # 有切缝！！！。。。
        # sub_ims, im_hw_c = im_to_batch(im, (640, 640))

        for i in range(len(sub_ims)):
            im = sub_ims[i]

            bim = im_arr_to_tensor(im[None])
            if has_cuda:
                bim = bim.cuda()

            latent_z = enet(bim)
            # latent_z[1][:, :, :, :latent_z[1].shape[3]//2] = 0.
            # latent_z[0].normal_()
            # latent_z[1].normal_()

            # imageio.imwrite('1.jpg', np.asarray((latent_z[0].cpu().numpy()[0, 0] + 1) / 2 * 255, np.uint8))

            new_imgs = gnet(latent_z)

            im = tensor_to_im_arr(new_imgs)[0]

            sub_ims[i] = im

        # im = batch_to_im(sub_ims, im_hw_c)
        im = batch_to_im_with_seamless(sub_ims, [ori_im_hw[0] * 2, ori_im_hw[1] * 2], im_hw_c, (sub_im_hw[0] * 2, sub_im_hw[1] * 2))

        # 去除pad
        # im = im[:im.shape[0]-h_pad*2, :im.shape[1]-w_pad*2]
        im = unpadding_im_with_multiples_of_n(im, [hw_pad[0]*2, hw_pad[1]*2])

        imageio.imwrite(dst_im_path, im)

    print('Success')