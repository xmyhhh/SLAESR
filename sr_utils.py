import numpy as np
import cv2


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


def batch_to_im(sub_ims, im_hw_c):
    '''
    与 im_to_batch 相反，可以把分块图像序列拼接成一张图像
    :param sub_ims:
    :param im_hw_c:
    :return:
    '''
    h_c, w_c = im_hw_c
    im_table = []
    for h_i in range(h_c):
        x_table = []
        for w_i in range(w_c):
            x_table.append(sub_ims.pop(0))
        im_table.append(np.concatenate(x_table, 1))
    im = np.concatenate(im_table, 0)
    return im


def im_to_batch_with_seamless(im, sub_im_hw=(640, 640)):
    '''
    功能与 im_to_batch 相似，不过这里的分块后的图像有1/4部分重叠，用于生成无缝图
    无缝方式，x轴上，将每张图分割成4块，第一张图的第3，4区域与第二张图的1，2区域相同，舍弃第一张图的区域4和第二张图的区域1，然后将他们拼接起来
    这样就避免了边缘质量差的问题
    :param im: 输入图像
    :param sub_im_hw: 分块图像大小
    :return:
    '''
    assert im.shape[0] % 4 == 0 and im.shape[1] % 4 == 0
    assert sub_im_hw[0] % 4 == 0 and sub_im_hw[1] % 4 == 0
    ori_hw = im.shape[:2]
    h_step = sub_im_hw[0] // 2
    w_step = sub_im_hw[1] // 2
    h_c = int(np.ceil(ori_hw[0] / h_step))
    w_c = int(np.ceil(ori_hw[1] / w_step))
    sub_ims = []
    for h_i in range(h_c):
        for w_i in range(w_c):
            nim = im[h_step * h_i: h_step * h_i + sub_im_hw[0], w_step * w_i: w_step * w_i + sub_im_hw[1]]
            sub_ims.append(nim)
    # 因为图像并不规则，拼接时需要原图大小
    return sub_ims, [h_c, w_c], list(ori_hw)


def batch_to_im_with_seamless(sub_ims, ori_im_hw, im_hw_c, sub_im_hw=(640, 640)):
    '''
    与 im_to_batch_with_seamless 功能相反
    :param sub_ims:
    :param ori_im_hw:
    :param im_hw_c:
    :param sub_im_hw:
    :return:
    '''
    # 用于确保图像大小为32倍数，理论上最小可以设置为4
    assert ori_im_hw[0] % 4 == 0 and ori_im_hw[1] % 4 == 0
    assert sub_im_hw[0] % 4 == 0 and sub_im_hw[1] % 4 == 0
    h_c, w_c = im_hw_c
    # 1/2 步长，用于定位贴图开始
    h_step = sub_im_hw[0] // 2
    w_step = sub_im_hw[1] // 2
    # 1/4 步长，用于跳过不填充区
    h_step_half = sub_im_hw[0] // 4
    w_step_half = sub_im_hw[1] // 4
    im = np.zeros([*ori_im_hw, sub_ims[0].shape[-1]], sub_ims[0].dtype)
    for h_i in range(h_c):
        y_first = True if h_i == 0 else False
        for w_i in range(w_c):
            x_first = True if w_i == 0 else False

            y_pos = h_i * h_step
            x_pos = w_i * w_step

            # 位于第一行或第一列，相应边不进行裁切
            y_pos_inc = 0 if y_first else h_step_half
            x_pos_inc = 0 if x_first else w_step_half

            _im = sub_ims.pop(0)
            im[y_pos+y_pos_inc: y_pos+sub_im_hw[0], x_pos+x_pos_inc: x_pos+sub_im_hw[1]] = _im[y_pos_inc:, x_pos_inc:]
    return im


def padding_im_with_multiples_of_n(im, n=32, borderType=cv2.BORDER_CONSTANT, value=None):
    h_pad = im.shape[0] % n
    w_pad = im.shape[1] % n
    if h_pad > 0:
        h_pad = n - h_pad
    if w_pad > 0:
        w_pad = n - w_pad
    im = cv2.copyMakeBorder(im, 0, h_pad, 0, w_pad, borderType, value=value)
    return im, [h_pad, w_pad]


def unpadding_im_with_multiples_of_n(im, hw_pad):
    im = im[:im.shape[0] - hw_pad[0], :im.shape[1] - hw_pad[1]]
    return im
