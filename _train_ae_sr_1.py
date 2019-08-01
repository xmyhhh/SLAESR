import os
import time
import json

from auto_encode_sr_net1 import GenNet
from auto_encode_sr_net1 import Encoder

from torch.utils.data import DataLoader
import model_utils_torch
from model_utils_torch import *
# from dataset_reader import SrDatasetReader

from dataset_reader.NoTagFaceDatasetReader import NoTagFaceDatasetReader
from ae_utils import *
from tensorboardX import SummaryWriter


torch.backends.cudnn.benchmark = True


if __name__ == '__main__':

    example_id = '1'

    sample_dir = 'samples_{}'.format(example_id)
    extra_file = 'extra_{}.json'.format(example_id)
    best_extra_file = 'best_' + extra_file
    model_file = 'model_{}.pt'.format(example_id)
    best_model_file = 'best_' + model_file
    log_dir = 'logs_' + example_id

    os.makedirs(sample_dir, exist_ok=True)

    sometime_work_and_sleep = False
    auto_save_best_model = False

    img_dim = 128
    batch_size = 21
    # 梯度累积次数
    grad_cum_count = 1

    iters_per_sample = 100
    iter_count = 100000000
    n_size = 9

    enet = Encoder().cuda()
    gnet = GenNet().cuda()

    ssim_losser = model_utils_torch.image.SSIM(data_range=2.).cuda()

    # 手动变更学习率
    # 0-8000；g：3e-4；d：3e-4
    # 8000-10000：g：1e-5；d：5e-5
    # 10000-；g：1e-6；d：5e-6
    lr = 1e-3
    optimizer = torch.optim.Adam(list(gnet.parameters()) + list(enet.parameters()), betas=(0., 0.99), lr=lr, eps=1e-8)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [1000, 6000, 12000], 1)
    best_loss = np.inf
    start_it = 0
    try:
        model = None
        model = torch.load(model_file)
        enet.load_state_dict(model['enet'])
        gnet.load_state_dict(model['gnet'])
        optimizer.load_state_dict(model['optim'])
        print('Load model success')

        if os.path.isfile(extra_file):
            extra = json.load(open(extra_file, 'r'))
            start_it = extra['cur_it'] + 1
            best_loss = extra['best_loss']

    except (FileNotFoundError, RuntimeError):
        print('Not found save model')

    finally:
        del model

    print('enet params')
    print_params_size(enet.parameters())
    print('gnet params')
    print_params_size(gnet.parameters())

    dataset = NoTagFaceDatasetReader(r'../datasets/getchu_aligned_with_label/GetChu_aligned2', (img_dim, img_dim),
                                     iter_count, random_horizontal_flip=True, mini_dataset=0)
    # dataset = SrDatasetReader.SrDatasetReader(r'../datasets/绮丽', iter_count=iter_count)

    dataloader = DataLoader(dataset, batch_size, False, num_workers=1, timeout=10)

    ts1 = time.time()

    nd = next_data2(dataloader)
    torch.cuda.empty_cache()

    loss_acc = 0.
    l1_loss_acc = 0.
    ssim_loss_acc = 0.

    writer = SummaryWriter(log_dir)

    # 用于RGB shuffle操作。用来处理 数据集颜色不平衡，某些颜色表现会较差的问题
    rgb_shuffle_id = [0, 1, 2]

    for i in range(start_it, iter_count):
        scheduler.step(i)

        loss = l1_loss = ssim_loss = 0
        z_max = z_min = z_std = z_mean = 0
        gm_ssim_loss = 0

        enet.train()
        gnet.train()

        # 先累积梯度
        optimizer.zero_grad()
        for _ in range(grad_cum_count):
            batch_imgs = next(nd)
            # 试试 RGB shuffle
            np.random.shuffle(rgb_shuffle_id)
            batch_imgs = batch_imgs[:, rgb_shuffle_id]

            ori_imgs = batch_imgs.cuda()

            # SR，先下采样
            ds_imgs = F.interpolate(ori_imgs, size=(img_dim//2, img_dim//2), mode='area')

            latent_z = enet(ds_imgs)

            new_imgs = gnet(latent_z)

            l1_loss = F.l1_loss(new_imgs, ori_imgs, reduction='mean')
            ssim_loss = (1 - ssim_losser(new_imgs, ori_imgs)).mean()
            loss = l1_loss + ssim_loss

            loss.backward()
        optimizer.step()

        l1_loss = l1_loss.item()
        ssim_loss = ssim_loss.item()
        loss = loss.item()

        l1_loss_acc += l1_loss
        ssim_loss_acc += ssim_loss
        loss_acc += loss

        if np.isnan(loss):
            print('Found loss Nan!', 'loss %f' % loss)
            raise AttributeError('Found Nan')

        if i > 0 and i % 10 == 0:
            # print loss
            l1_loss = l1_loss_acc / 10
            ssim_loss = ssim_loss_acc / 10
            loss = loss_acc / 10
            ts2 = time.time()
            print('iter: %d, loss: %.4f' % (i, loss), 'l1_loss %.4f' % l1_loss, 'ssim_loss %.4f' % ssim_loss,
                  'time: %.4f' % (ts2 - ts1))
            writer.add_scalar('l1_loss', l1_loss, i)
            writer.add_scalar('ssim_loss', ssim_loss, i)
            writer.add_scalar('loss', loss, i)
            ts1 = ts2
            l1_loss_acc = ssim_loss_acc = loss_acc = 0.

        if i > 0 and i % iters_per_sample == 0:
            enet.eval()
            gnet.eval()

            batch_imgs = []
            for _ in range(3):
                batch_imgs.append(next(nd))

            ori_imgs = torch.cat(batch_imgs, 0)
            ori_imgs = ori_imgs.cuda()
            ds_imgs = F.interpolate(ori_imgs, size=(img_dim//2, img_dim//2), mode='area')

            sample_with_label(enet, gnet, '%s/test_%d.jpg' % (sample_dir, i), ds_imgs, (8, 8), batch_size=batch_size)
            sample_with_label(enet, gnet, '%s/test_%d_SR.jpg' % (sample_dir, i), ori_imgs, (8, 8), batch_size=batch_size//2)

            if auto_save_best_model and loss < best_loss:
                print('save best model')
                best_model = {'enet': enet.state_dict(),
                              'gnet': gnet.state_dict()}
                best_extra = {'cur_it': i,
                              'best_loss': best_loss}
                best_loss = loss
                torch.save(best_model, best_model_file)
                json.dump(best_extra, open(best_extra_file, 'w'))

            model = {'enet': enet.state_dict(),
                     'gnet': gnet.state_dict(),
                     'optim': optimizer.state_dict()}
            extra = {'cur_it': i,
                     'best_loss': best_loss}
            torch.save(model, model_file)
            json.dump(extra, open(extra_file, 'w'))

            del model

        # 定时待机
        while sometime_work_and_sleep and (55 <= time.localtime().tm_min < 60 or 25 <= time.localtime().tm_min < 30):
            time.sleep(20)
