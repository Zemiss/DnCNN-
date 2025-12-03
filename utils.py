import math
import torch
import torch.nn as nn
import numpy as np
# 导入skimage库中的峰值信噪比(PSNR)计算函数
from skimage.metrics import peak_signal_noise_ratio as compare_psnr

# 权重初始化函数：使用Kaiming初始化方法
def weights_init_kaiming(m):
    """
    Kaiming初始化（也称He初始化）适用于ReLU激活函数
    可以避免梯度消失和梯度爆炸问题
    """
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        # ===========================================================================================
        # 【旧版PyTorch语法修改5：更新Kaiming初始化函数名】
        # 旧版：nn.init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
        # 新版：nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
        # 原因：PyTorch统一了命名规范，in-place操作的函数名以下划线'_'结尾
        #       kaiming_normal已被弃用，应使用kaiming_normal_
        # ===========================================================================================
        nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm') != -1:
        # BatchNorm层的权重和偏置初始化
        m.weight.data.normal_(mean=0, std=math.sqrt(2./9./64.)).clamp_(-0.025,0.025)
        nn.init.constant_(m.bias.data, 0.0)

# PSNR（峰值信噪比）计算函数：衡量图像质量的指标，值越大质量越好
def batch_PSNR(img, imclean, data_range):
    """
    计算一个批次图像的平均PSNR
    参数:
        img: 预测图像
        imclean: 干净的原始图像
        data_range: 数据范围（通常为1.0）
    返回:
        批次平均PSNR值
    """
    Img = img.data.cpu().numpy().astype(np.float32)
    Iclean = imclean.data.cpu().numpy().astype(np.float32)
    PSNR = 0
    for i in range(Img.shape[0]):
        PSNR += compare_psnr(Iclean[i,:,:,:], Img[i,:,:,:], data_range=data_range)
    return (PSNR/Img.shape[0])

# 数据增强函数：通过旋转、翻转等操作增加数据多样性，提升模型鲁棒性
def data_augmentation(image, mode):
    """
    对图像进行8种不同的变换
    参数:
        image: 输入图像，形状为(C, H, W)
        mode: 增强模式（0-7）
    返回:
        增强后的图像
    """
    out = np.transpose(image, (1,2,0))  # 转换为(H, W, C)
    if mode == 0:
        # 原始图像，不做变换
        out = out
    elif mode == 1:
        # 上下翻转
        out = np.flipud(out)
    elif mode == 2:
        # 逆时针旋转90度
        out = np.rot90(out)
    elif mode == 3:
        # 旋转90度后上下翻转
        out = np.rot90(out)
        out = np.flipud(out)
    elif mode == 4:
        # 旋转180度
        out = np.rot90(out, k=2)
    elif mode == 5:
        # 旋转180度后翻转
        out = np.rot90(out, k=2)
        out = np.flipud(out)
    elif mode == 6:
        # 旋转270度
        out = np.rot90(out, k=3)
    elif mode == 7:
        # 旋转270度后翻转
        out = np.rot90(out, k=3)
        out = np.flipud(out)
    return np.transpose(out, (2,0,1))  # 转换回(C, H, W)
