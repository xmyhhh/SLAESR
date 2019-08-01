'''
修改自 AE/auto_encode_net2.py
使用biggan格式的残差自编码器

编码器输入 64x64
解码器输出 128x128
实现超分辨率
'''

from model_utils_torch import *
from BlurPool2D import BlurPool2D


def ConvBnAct(in_ch, out_ch, ker_sz, stride, pad, use_bn, act):
    layers = [nn.Conv2d(in_ch, out_ch, ker_sz, stride, pad, bias=not use_bn)]
    if use_bn: layers.append(nn.BatchNorm2d(out_ch, eps=1e-8, momentum=0.9))
    if act is not None: layers.append(act)
    return nn.Sequential(*layers)


def ConvWnAct(in_ch, out_ch, ker_sz, stride, pad, use_wn, act):
    layers = [nn.Conv2d(in_ch, out_ch, ker_sz, stride, pad, bias=True)]
    if use_wn: layers[0] = nn.utils.weight_norm(layers[0])
    if act is not None: layers.append(act)
    return nn.Sequential(*layers)


def DeConvBnAct(in_ch, out_ch, ker_sz, stride, pad, out_pad, use_bn, act):
    layers = [nn.ConvTranspose2d(in_ch, out_ch, ker_sz, stride, pad, out_pad, bias=not use_bn)]
    if use_bn: layers.append(nn.BatchNorm2d(out_ch, eps=1e-8, momentum=0.9))
    if act is not None: layers.append(act)
    return nn.Sequential(*layers)


edge_dim = 1
color_dim = 64


class GenBlock(nn.Module):
    def __init__(self, in_ch, out_ch, act, mode='up'):
        super().__init__()
        assert mode in ['up', 'keep']
        # self.act = act
        # if self.act is None:
        #     self.act = nn.Identity()

        # blur_pool = BlurPool2D(3, 1)

        if mode == 'up':
            self.skip_conv1 = nn.Sequential(
                ConvBnAct(in_ch, out_ch, 3, 1, 1, False, None),
                Upsample(scale_factor=2., mode='bilinear'),
            )
        elif in_ch != out_ch:
            self.skip_conv1 = ConvBnAct(in_ch, out_ch, 1, 1, 0, False, None)
        else:
            self.skip_conv1 = nn.Identity()

        if mode == 'up':
            self.conv1 = nn.Sequential(
                ConvBnAct(in_ch, out_ch, 3, 1, 1, True, act),
                Upsample(scale_factor=2., mode='bilinear')
            )
        else:
            self.conv1 = ConvBnAct(in_ch, out_ch, 3, 1, 1, True, act)
        self.conv2 = ConvBnAct(out_ch, out_ch, 3, 1, 1, True, act)

    def forward(self, x):
        y1 = self.skip_conv1(x)
        y = self.conv1(x)
        y = self.conv2(y)
        y = y1 + y
        return y


class GenNet(nn.Module):
    def __init__(self):
        super().__init__()
        act = Swish()

        self.blur_pool = BlurPool2D(3, 1)

        self.color_conv1 = ConvBnAct(color_dim, 256, 3, 1, 1, True, act)
        self.color_conv2 = ConvBnAct(256, 256, 3, 1, 1, True, act)

        # 4x4
        # 3x3
        # self.gb1 = GenBlock(256, 256, act, 'up')
        # 8x8
        # 6x6
        self.gb2 = GenBlock(256, 256, act, 'up')
        # 16x16
        # 12x12
        self.gb3 = GenBlock(256, 128, act, 'up')
        # 32x32
        # 24x24
        self.gb4 = GenBlock(128, 64, act, 'up')
        # 64x64
        # 48x48
        self.edge_conv1 = ConvBnAct(edge_dim, 32, 3, 1, 1, True, act)
        self.edge_conv2 = ConvBnAct(32, 64, 3, 1, 1, True, act)
        self.edge_conv3 = nn.Sequential(ConvBnAct(64, 128, 3, 2, 1, True, act),
                                        self.blur_pool)
        self.edge_conv4 = nn.Sequential(ConvBnAct(128, 256, 3, 2, 1, True, act),
                                        self.blur_pool)
        self.edge_conv5 = nn.Sequential(ConvBnAct(256, 256, 3, 2, 1, True, act),
                                        self.blur_pool)

        self.gb5 = GenBlock(64, 32, act, 'up')
        # 128x128
        # 96x96
        # self.conv1 = ConvBnAct(32, 32, 3, 1, 1, True, act)
        self.img_conv = ConvBnAct(32, 3, 1, 1, 0, False, None)
        # self.img_conv = nn.Sequential(
        #     nn.ReflectionPad2d(1),
        #     nn.Conv2d(32, 3, 3, 1, 0, bias=False),
        #     nn.Tanh()
        # )

    def forward(self, x):
        edge_z, color_z = x

        edge_1 = self.edge_conv1(edge_z)
        edge_2 = self.edge_conv2(edge_1)
        edge_3 = self.edge_conv3(edge_2)
        edge_4 = self.edge_conv4(edge_3)
        edge_5 = self.edge_conv5(edge_4)

        color_y = self.color_conv1(color_z)
        color_y = self.color_conv2(color_y)

        y = color_y + edge_5
        # y = self.gb1(y) + edge_5
        y = self.gb2(y) + edge_4
        y = self.gb3(y) + edge_3
        y = self.gb4(y) + edge_2
        y = self.gb5(y)

        # y = self.conv1(y)
        y = self.img_conv(y)
        return y


class EncBlock(nn.Module):
    def __init__(self, in_ch, out_ch, act, mode='down'):
        super().__init__()
        assert mode in ['down', 'keep']
        # self.act = act
        # if self.act is None:
        #     self.act = nn.Identity()
        # norm2d = nn.BatchNorm2d
        # norm2d_kwargs = {'eps': 1e-8, 'momentum': 0.9}

        blur_pool = BlurPool2D(3, 1)

        # 必须要加bn，不然会数值会爆炸
        if mode == 'down':
            self.conv_skip = nn.Sequential(ConvBnAct(in_ch, out_ch, 3, 2, 1, False, None),
                                           blur_pool)
        elif in_ch != out_ch:
            self.conv_skip = ConvBnAct(in_ch, out_ch, 1, 1, 0, False, None)
        else:
            self.conv_skip = nn.Identity()

        if mode == 'down':
            self.conv1 = nn.Sequential(ConvBnAct(in_ch, int(out_ch*1.5), 3, 2, 1, True, act),
                                       blur_pool)
        else:
            self.conv1 = ConvBnAct(in_ch, int(out_ch*1.5), 3, 1, 1, True, act)
        self.conv2 = ConvBnAct(int(out_ch*1.5), out_ch, 3, 1, 1, True, act)

    def forward(self, x):
        y1 = self.conv_skip(x)
        y = self.conv1(x)
        y = self.conv2(y)
        y = y1 + y
        return y


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        act = Swish()

        # 128x128
        # 96x96
        # self.conv1 = ConvBnAct(3, 32, 1, 1, 0, False, None)
        # self.db1 = EncBlock(32, 64, act)
        # 64x64
        # 48x48
        self.conv1 = ConvBnAct(3, 64, 1, 1, 0, False, None)
        self.db1 = EncBlock(64, 64, act, mode='keep')
        self.edge_conv1 = ConvBnAct(64, 32, 3, 1, 1, True, act)
        self.edge_conv2 = ConvBnAct(32, edge_dim, 3, 1, 1, False, nn.Tanh())

        self.db2 = EncBlock(64, 128, act)

        # 32x32
        # 24x24
        self.db3 = EncBlock(128, 256, act)
        # 16x16
        # 12x12
        self.db4 = EncBlock(256, 256, act)
        # 8x8
        # 6x6
        # self.db5 = EncBlock(256, 256, act)
        # 4x4
        # 3x3
        self.color_conv1 = ConvBnAct(256, 256, 3, 1, 1, True, act)
        self.color_conv2 = ConvBnAct(256, color_dim, 3, 1, 1, False, nn.Tanh())

    def forward(self, x):
        y = self.conv1(x)
        y = self.db1(y)
        edge_y = self.edge_conv1(y)
        edge_y = self.edge_conv2(edge_y)

        y = self.db2(y)
        y = self.db3(y)
        y = self.db4(y)
        # y = self.db5(y)

        color_y = self.color_conv1(y)
        color_y = self.color_conv2(color_y)

        return edge_y, color_y


if __name__ == '__main__':
    enet = Encoder()
    gnet = GenNet()
    print('enet')
    print_params_size2(enet)
    print('gnet')
    print_params_size2(gnet)
    a = torch.rand(3, 3, 128, 128)
    edge_z, color_z = enet(a)
    c = gnet([edge_z, color_z])
    print(edge_z.shape, color_z.shape)
    print(c.shape)
