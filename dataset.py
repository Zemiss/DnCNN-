import os
import os.path
import numpy as np
import random
import h5py
import torch
import cv2
import glob
import torch.utils.data as udata
from utils import data_augmentation

def normalize(data):
    """将图像数据归一化到[0,1]范围"""
    return data/255.

# 图像分块函数：将原始图片切分成多个小图像块（patch）
def Im2Patch(img, win, stride=1):
    """
    参数:
        img: 输入图像，形状为(C, H, W)
        win: 图像块的大小
        stride: 滑动窗口的步长
    返回:
        图像块数组，形状为(C, win, win, TotalPatNum)
    """
    k = 0
    endc = img.shape[0]  # 通道数
    endw = img.shape[1]  # 图像宽度
    endh = img.shape[2]  # 图像高度
    patch = img[:, 0:endw-win+0+1:stride, 0:endh-win+0+1:stride]
    TotalPatNum = patch.shape[1] * patch.shape[2]  # 总图像块数量
    Y = np.zeros([endc, win*win,TotalPatNum], np.float32)
    for i in range(win):
        for j in range(win):
            patch = img[:,i:endw-win+i+1:stride,j:endh-win+j+1:stride]
            Y[:,k,:] = np.array(patch[:]).reshape(endc, TotalPatNum)
            k = k + 1
    return Y.reshape([endc, win, win, TotalPatNum])

# 数据预处理函数：生成训练和验证数据集（保存为h5文件）
# 训练数据：多尺度、多增强的小图像块
# 验证数据：完整图像（不分块）
def prepare_data(data_path, patch_size, stride, aug_times=1):
    # 处理训练数据
    print('process training data')
    scales = [1, 0.9, 0.8, 0.7]  # 多尺度训练，增强模型鲁棒性
    files = glob.glob(os.path.join(data_path, 'train', '*.png'))
    files.sort()
    h5f = h5py.File('train.h5', 'w')  # 创建h5文件用于存储训练数据
    train_num = 0
    for i in range(len(files)):
        img = cv2.imread(files[i])
        h, w, c = img.shape
        for k in range(len(scales)):
            # 按不同比例缩放图像
            Img = cv2.resize(img, (int(h*scales[k]), int(w*scales[k])), interpolation=cv2.INTER_CUBIC)
            Img = np.expand_dims(Img[:,:,0].copy(), 0)  # 只取第一个通道（灰度图）
            Img = np.float32(normalize(Img))  # 归一化到[0,1]
            patches = Im2Patch(Img, win=patch_size, stride=stride)  # 切分成小块
            print("file: %s scale %.1f # samples: %d" % (files[i], scales[k], patches.shape[3]*aug_times))
            for n in range(patches.shape[3]):
                data = patches[:,:,:,n].copy()
                h5f.create_dataset(str(train_num), data=data)  # 存储原始图像块
                train_num += 1
                # 数据增强：旋转、翻转等
                for m in range(aug_times-1):
                    data_aug = data_augmentation(data, np.random.randint(1,8))
                    h5f.create_dataset(str(train_num)+"_aug_%d" % (m+1), data=data_aug)
                    train_num += 1
    h5f.close()
    # 处理验证数据
    print('\nprocess validation data')
    files.clear()
    files = glob.glob(os.path.join(data_path, 'Set12', '*.png'))
    files.sort()
    h5f = h5py.File('val.h5', 'w')  # 创建h5文件用于存储验证数据
    val_num = 0
    for i in range(len(files)):
        print("file: %s" % files[i])
        img = cv2.imread(files[i])
        img = np.expand_dims(img[:,:,0], 0)  # 只取第一个通道（灰度图）
        img = np.float32(normalize(img))  # 归一化到[0,1]
        h5f.create_dataset(str(val_num), data=img)  # 存储完整图像（不切分）
        val_num += 1
    h5f.close()
    print('training set, # samples %d\n' % train_num)
    print('val set, # samples %d\n' % val_num)

# 自定义数据集类，用于加载h5文件中的数据
class Dataset(udata.Dataset):
    def __init__(self, train=True):
        """
        参数:
            train: True表示加载训练集，False表示加载验证集
        """
        super(Dataset, self).__init__()
        self.train = train
        if self.train:
            h5f = h5py.File('train.h5', 'r')
        else:
            h5f = h5py.File('val.h5', 'r')
        self.keys = list(h5f.keys())  # 获取所有数据集的键
        random.shuffle(self.keys)  # 随机打乱数据
        h5f.close()
    
    def __len__(self):
        """返回数据集大小"""
        return len(self.keys)
    
    def __getitem__(self, index):
        """根据索引获取数据"""
        if self.train:
            h5f = h5py.File('train.h5', 'r')
        else:
            h5f = h5py.File('val.h5', 'r')
        key = self.keys[index]
        data = np.array(h5f[key])
        h5f.close()
        return torch.Tensor(data)
