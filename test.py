import cv2
import os
import argparse
import glob
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from models import DnCNN
from utils import *

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# 命令行参数解析
parser = argparse.ArgumentParser(description="DnCNN_Test")
parser.add_argument("--num_of_layers", type=int, default=20, help="网络总层数")
parser.add_argument("--logdir", type=str, default="logs", help='日志文件路径')
parser.add_argument("--test_data", type=str, default='Set12', help='在Set12或Set68上测试')
parser.add_argument("--test_noiseL", type=float, default=25, help='测试集使用的噪声水平')
opt = parser.parse_args()

def normalize(data):
    """将图像数据归一化到[0,1]范围"""
    return data/255.

# 测试主函数：使用预训练的DnCNN模型对图像进行去噪处理，并评估其性能
def main():
    # 构建模型
    print('Loading model ...\n')
    net = DnCNN(channels=1, num_of_layers=opt.num_of_layers)
    device_ids = [0]
    model = nn.DataParallel(net, device_ids=device_ids).cuda()
    # ===========================================================================================
    # 【旧版PyTorch语法修改4：使用weights_only=True加载模型】
    # 旧版：model.load_state_dict(torch.load(os.path.join(opt.logdir, 'net.pth')))
    # 新版：model.load_state_dict(torch.load(os.path.join(opt.logdir, 'net.pth'), weights_only=True))
    # 原因：为了安全性，新版PyTorch建议显式指定weights_only=True
    #       这样可以防止加载恶意构造的pickle文件导致的任意代码执行
    # ===========================================================================================
    model.load_state_dict(torch.load(os.path.join(opt.logdir, 'net.pth'), weights_only=True))
    model.eval()
    # 加载数据信息
    print('Loading data info ...\n')
    files_source = glob.glob(os.path.join('data', opt.test_data, '*.png'))
    files_source.sort()
    # 处理数据
    psnr_test = 0
    for f in files_source:
        # 读取图像
        Img = cv2.imread(f)
        Img = normalize(np.float32(Img[:,:,0]))  # 只取第一个通道（灰度图）
        Img = np.expand_dims(Img, 0)  # 添加通道维度
        Img = np.expand_dims(Img, 1)  # 添加batch维度
        ISource = torch.Tensor(Img)
        # 生成噪声
        noise = torch.FloatTensor(ISource.size()).normal_(mean=0, std=opt.test_noiseL/255.)
        # 生成含噪图像
        INoisy = ISource + noise
        ISource, INoisy = Variable(ISource.cuda()), Variable(INoisy.cuda())
        with torch.no_grad():  # 不计算梯度，节省内存
            Out = torch.clamp(INoisy-model(INoisy), 0., 1.)  # 去噪并限制在[0,1]范围
        psnr = batch_PSNR(Out, ISource, 1.)
        psnr_test += psnr
        print("%s PSNR %f" % (f, psnr))
    psnr_test /= len(files_source)
    print("\nPSNR on test data %f" % psnr_test)

if __name__ == "__main__":
    main()
