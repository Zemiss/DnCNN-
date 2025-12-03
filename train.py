import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as utils
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from models import DnCNN
from dataset import prepare_data, Dataset
from utils import *

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# 命令行参数解析
parser = argparse.ArgumentParser(description="DnCNN")
parser.add_argument("--preprocess", type=bool, default=False, help='是否运行数据预处理')
parser.add_argument("--batchSize", type=int, default=128, help="训练批次大小")
parser.add_argument("--num_of_layers", type=int, default=20, help="网络总层数")
parser.add_argument("--epochs", type=int, default=100, help="训练轮数")
parser.add_argument("--milestone", type=int, default=30, help="学习率衰减的里程碑，应小于总轮数")
parser.add_argument("--lr", type=float, default=1e-3, help="初始学习率")
parser.add_argument("--outf", type=str, default="logs", help='日志文件保存路径')
parser.add_argument("--mode", type=str, default="S", help='已知噪声水平(S)或盲训练(B)')
parser.add_argument("--noiseL", type=float, default=25, help='噪声水平；当mode=B时忽略')
parser.add_argument("--val_noiseL", type=float, default=25, help='验证集使用的噪声水平')
opt = parser.parse_args()

# 模型训练主函数
def main():
    # 加载数据集
    print('Loading dataset ...\n')
    dataset_train = Dataset(train=True)
    dataset_val = Dataset(train=False)
    loader_train = DataLoader(dataset=dataset_train, num_workers=4, batch_size=opt.batchSize, shuffle=True)
    print("# of training samples: %d\n" % int(len(dataset_train)))
    # 构建模型
    net = DnCNN(channels=1, num_of_layers=opt.num_of_layers)
    net.apply(weights_init_kaiming)
    # ===========================================================================================
    # 【旧版PyTorch语法修改1：更新MSELoss参数】
    # 旧版：criterion = nn.MSELoss(size_average=False)
    # 新版：使用reduction='sum'替代size_average=False
    # 原因：size_average参数在新版本中已被弃用，使用reduction参数更加清晰
    # ===========================================================================================
    criterion = nn.MSELoss(reduction='sum')
    # Move to GPU
    device_ids = [0]
    model = nn.DataParallel(net, device_ids=device_ids).cuda()
    criterion.cuda()
    # 优化器设置（使用AdamW，带权重衰减）
    optimizer = optim.AdamW(model.parameters(), lr=opt.lr, weight_decay=1e-4)
    # 学习率调度器（余弦退火）
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.epochs, eta_min=1e-6)
    # 训练过程
    writer = SummaryWriter(opt.outf)
    step = 0
    noiseL_B=[0,55]  # 盲训练的噪声范围，当opt.mode=='S'时忽略
    for epoch in range(opt.epochs):
        current_lr = optimizer.param_groups[0]['lr']
        print('learning rate %f' % current_lr)
        # 开始训练
        for i, data in enumerate(loader_train, 0):
            # 训练步骤
            model.train()
            model.zero_grad()
            optimizer.zero_grad()
            img_train = data
            if opt.mode == 'S':
                noise = torch.FloatTensor(img_train.size()).normal_(mean=0, std=opt.noiseL/255.)
            if opt.mode == 'B':
                noise = torch.zeros(img_train.size())
                stdN = np.random.uniform(noiseL_B[0], noiseL_B[1], size=noise.size()[0])
                for n in range(noise.size()[0]):
                    sizeN = noise[0,:,:,:].size()
                    noise[n,:,:,:] = torch.FloatTensor(sizeN).normal_(mean=0, std=stdN[n]/255.)
            imgn_train = img_train + noise
            img_train, imgn_train = Variable(img_train.cuda()), Variable(imgn_train.cuda())
            noise = Variable(noise.cuda())
            out_train = model(imgn_train)
            loss = criterion(out_train, noise) / (imgn_train.size()[0]*2)
            loss.backward()
            optimizer.step()
            # 计算训练结果
            model.eval()
            out_train = torch.clamp(imgn_train-model(imgn_train), 0., 1.)
            psnr_train = batch_PSNR(out_train, img_train, 1.)
            print("[epoch %d][%d/%d] loss: %.4f PSNR_train: %.4f" %
                (epoch+1, i+1, len(loader_train), loss.item(), psnr_train))
            if step % 10 == 0:
                # 记录标量值到TensorBoard
                writer.add_scalar('loss', loss.item(), step)
                writer.add_scalar('PSNR on training data', psnr_train, step)
            step += 1
        
        scheduler.step()  # 更新学习率
        ## 每个epoch结束后的验证
        model.eval()
        # ===========================================================================================
        # 【旧版PyTorch语法修改2：使用torch.no_grad()替代volatile=True】
        # 旧版：Variable(tensor, volatile=True) 用于在推理时禁用梯度计算
        # 新版：使用torch.no_grad()上下文管理器包裹推理代码，更加简洁和高效
        # 原因：volatile参数在PyTorch 0.4.0版本后已被弃用，torch.no_grad()是推荐的做法
        # 作用：在验证阶段不计算梯度，节省内存并加速推理
        # ===========================================================================================
        with torch.no_grad():
            # 验证过程
            psnr_val = 0
            for k in range(len(dataset_val)):
                img_val = torch.unsqueeze(dataset_val[k], 0)
                noise = torch.FloatTensor(img_val.size()).normal_(mean=0, std=opt.val_noiseL/255.)
                imgn_val = img_val + noise
                # ===========================================================================================
                # 【旧版PyTorch语法修改3：移除Variable和volatile参数】
                # 旧版：img_val, imgn_val = Variable(img_val.cuda(), volatile=True), Variable(imgn_val.cuda(), volatile=True)
                # 新版：img_val, imgn_val = img_val.cuda(), imgn_val.cuda()
                # 原因：PyTorch 0.4.0后，Tensor和Variable合并，不再需要手动创建Variable
                #       volatile参数的功能由torch.no_grad()替代
                # ===========================================================================================
                img_val, imgn_val = img_val.cuda(), imgn_val.cuda() 
                out_val = torch.clamp(imgn_val-model(imgn_val), 0., 1.)
                psnr_val += batch_PSNR(out_val, img_val, 1.)
            psnr_val /= len(dataset_val)
        print("\n[epoch %d] PSNR_val: %.4f" % (epoch+1, psnr_val))
        writer.add_scalar('PSNR on validation data', psnr_val, epoch)
        # 记录图像到TensorBoard
        out_train = torch.clamp(imgn_train-model(imgn_train), 0., 1.)
        Img = utils.make_grid(img_train.data, nrow=8, normalize=True, scale_each=True)
        Imgn = utils.make_grid(imgn_train.data, nrow=8, normalize=True, scale_each=True)
        Irecon = utils.make_grid(out_train.data, nrow=8, normalize=True, scale_each=True)
        writer.add_image('clean image', Img, epoch)
        writer.add_image('noisy image', Imgn, epoch)
        writer.add_image('reconstructed image', Irecon, epoch)
        # 保存模型
        torch.save(model.state_dict(), os.path.join(opt.outf, 'net.pth'))

if __name__ == "__main__":
    if opt.preprocess:
        if opt.mode == 'S':
            prepare_data(data_path='data', patch_size=40, stride=10, aug_times=1)
        if opt.mode == 'B':
            prepare_data(data_path='data', patch_size=50, stride=10, aug_times=2)
    main()
