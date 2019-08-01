'''
Modified from
https://github.com/adobe/antialiased-cnns/blob/master/models_lpf/__init__.py
'''

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.jit


class BlurPool2D(torch.jit.ScriptModule):
    __constants__ = ['pad_mode', 'pad', 'stride']
    def __init__(self, ker_sz=3, stride=2, pad_mode='reflect'):
        super().__init__()
        assert pad_mode in ['constant', 'reflect', 'replicate', 'circular']
        self.pad_mode = pad_mode
        self.pad = [int(1. * (ker_sz - 1) / 2), int(np.ceil(1. * (ker_sz - 1) / 2)), int(1. * (ker_sz - 1) / 2), int(np.ceil(1. * (ker_sz - 1) / 2))]
        self.stride = stride

        if ker_sz==3:
            k = [1., 2., 1.]
        elif ker_sz==5:
            k = [1., 4., 6., 4., 1.]
        else:
            raise AssertionError('Only support ker_sz == 3 or 5 now')

        kernel = torch.tensor(k)[None, None, None]
        kernel = kernel / torch.sum(kernel)
        self.register_buffer('kernel', kernel)

    @torch.jit.script_method
    def forward(self, x):
        y = F.pad(x, self.pad, mode=self.pad_mode)
        k = self.kernel.expand(x.shape[1], -1, -1, -1)
        y = F.conv2d(y, k, stride=[1, self.stride], groups=x.shape[1])
        y = F.conv2d(y, k.transpose(2, 3), stride=[self.stride, 1], groups=x.shape[1])
        return y


if __name__ == '__main__':
    a=torch.rand(4,3,10,10)
    m = BlurPool2D()
    print(list(m.parameters()))
    b = m(a)
    print(b.shape)
