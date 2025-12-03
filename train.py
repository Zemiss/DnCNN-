import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as utils
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter # 推荐使用PyTorch内置的tensorboard
from models import DnCNN, Discriminator
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
parser.add_argument("--milestone", type=int, default=50, help="学习率衰减的里程碑，应小于总轮数")
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

    # ===========================================================================================
    # 【GAN修改1：构建生成器和判别器模型】
    # 生成器G：就是我们原来的DnCNN模型，负责生成去噪图像
    # 判别器D：新增的Discriminator模型，负责判别图像真伪
    # ===========================================================================================
    # 构建生成器 (Generator)
    net_G = DnCNN(channels=1, num_of_layers=opt.num_of_layers)
    net_G.apply(weights_init_kaiming)
    # 构建判别器 (Discriminator)
    net_D = Discriminator(channels=1, features=net_G.features)
    net_D.apply(weights_init_kaiming)

    # ===========================================================================================
    # 【旧版PyTorch语法修改1：更新MSELoss参数】
    # 旧版：criterion = nn.MSELoss(size_average=False)
    # 新版：使用reduction='sum'替代size_average=False
    # 原因：size_average参数在新版本中已被弃用，使用reduction参数更加清晰
    # ===========================================================================================
    # 【GAN修改2：定义损失函数】
    # 像素损失(Pixel Loss)：衡量生成图像和真实图像的像素差异，使用L1损失效果通常比MSE好
    # 对抗损失(Adversarial Loss)：衡量生成器欺骗判别器的能力，使用二元交叉熵损失
    # ===========================================================================================
    criterion_pixel = nn.L1Loss() # 使用L1损失代替MSE
    criterion_adv = nn.BCELoss()

    # 将模型和损失函数移动到GPU
    device_ids = [0]
    model_G = nn.DataParallel(net_G, device_ids=device_ids).cuda()
    model_D = nn.DataParallel(net_D, device_ids=device_ids).cuda()
    criterion_pixel.cuda()
    criterion_adv.cuda()

    # ===========================================================================================
    # 【GAN修改3：为生成器和判别器分别设置优化器】
    # ===========================================================================================
    optimizer_G = optim.AdamW(model_G.parameters(), lr=opt.lr, weight_decay=1e-4)
    optimizer_D = optim.AdamW(model_D.parameters(), lr=opt.lr * 0.1, weight_decay=1e-4) # 判别器学习率可以设置得低一些
    # 学习率调度器
    scheduler_G = optim.lr_scheduler.CosineAnnealingLR(optimizer_G, T_max=opt.epochs, eta_min=1e-6)
    scheduler_D = optim.lr_scheduler.CosineAnnealingLR(optimizer_D, T_max=opt.epochs, eta_min=1e-6)

    # 训练过程
    writer = SummaryWriter(opt.outf)
    step = 0
    noiseL_B=[0,55]  # 盲训练的噪声范围，当opt.mode=='S'时忽略
    for epoch in range(opt.epochs):
        # 获取当前生成器的学习率并打印
        current_lr_G = optimizer_G.param_groups[0]['lr']
        print('learning rate of G: %f' % current_lr_G)
        # 开始训练
        for i, data in enumerate(loader_train, 0):
            # 将模型设置为训练模式
            model_G.train()
            model_D.train()
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
            
            # ===========================================================================================
            # 【GAN修正1：动态生成标签以匹配批次大小】
            # 最后一个批次的大小可能小于batchSize，固定大小的label会引发尺寸不匹配错误。
            # 因此，在每个批次内部根据输入图像的实际大小动态创建标签。
            # ===========================================================================================
            current_batch_size = img_train.size(0)
            real_label = torch.ones(current_batch_size, 1).cuda()
            fake_label = torch.zeros(current_batch_size, 1).cuda()

            # ===========================================================================================
            # 【GAN修改4：对抗训练步骤】
            # 步骤1：训练判别器D
            # ===========================================================================================
            optimizer_D.zero_grad()

            # 用真实图像训练判别器
            output_real = model_D(img_train)
            loss_D_real = criterion_adv(output_real, real_label)

            # 用生成器生成的假图像训练判别器
            residual_pred = model_G(imgn_train)
            denoised_img = torch.clamp(imgn_train - residual_pred, 0., 1.)
            output_fake = model_D(denoised_img.detach()) # detach()防止梯度传回生成器
            loss_D_fake = criterion_adv(output_fake, fake_label)

            # 判别器总损失并更新
            loss_D = (loss_D_real + loss_D_fake) * 0.5
            loss_D.backward()
            optimizer_D.step()

            # ===========================================================================================
            # 步骤2：训练生成器G
            # ===========================================================================================
            optimizer_G.zero_grad()

            # 对抗损失：目标是让判别器将生成图像误判为“真”
            output_fake_for_G = model_D(denoised_img)
            loss_G_adv = criterion_adv(output_fake_for_G, real_label)

            # 像素损失：让生成图像在像素上接近真实图像
            loss_G_pixel = criterion_pixel(denoised_img, img_train)

            # 生成器总损失（加权求和）并更新
            loss_G = loss_G_pixel + 0.001 * loss_G_adv # 对抗损失的权重可以调整
            loss_G.backward()
            optimizer_G.step()

            # ===========================================================================================
            # 【GAN修改5：更新日志和评估】
            # ===========================================================================================
            psnr_train = batch_PSNR(denoised_img, img_train, 1.)
            print("[epoch %d][%d/%d] loss_D: %.4f, loss_G: %.4f (pixel: %.4f, adv: %.4f), PSNR_train: %.4f" %
                (epoch+1, i+1, len(loader_train), loss_D.item(), loss_G.item(), loss_G_pixel.item(), loss_G_adv.item(), psnr_train))

            if step % 10 == 0:
                # 记录标量值到TensorBoard
                writer.add_scalar('loss_D', loss_D.item(), step)
                writer.add_scalar('loss_G', loss_G.item(), step)
                writer.add_scalar('loss/pixel', loss_G_pixel.item(), step)
                writer.add_scalar('loss/adversarial', loss_G_adv.item(), step) # 【GAN修正4】增加对抗损失的记录
                writer.add_scalar('PSNR on training data', psnr_train, step)
            step += 1
        
        scheduler_G.step()  # 更新学习率
        scheduler_D.step()
        ## 每个epoch结束后的验证
        # ===========================================================================================
        # 【GAN修正2：正确设置模型为评估模式】
        # 旧代码中的`model.eval()`是错误的，因为`model`变量已不存在。
        # 需要分别对生成器和判别器设置评估模式。
        # ===========================================================================================
        model_G.eval()
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
                out_val = torch.clamp(imgn_val-model_G(imgn_val), 0., 1.)
                psnr_val += batch_PSNR(out_val, img_val, 1.)
            psnr_val /= len(dataset_val)
        print("\n[epoch %d] PSNR_val: %.4f" % (epoch+1, psnr_val))
        writer.add_scalar('PSNR on validation data', psnr_val, epoch)
        # 记录图像到TensorBoard
        out_train = torch.clamp(imgn_train-model_G(imgn_train), 0., 1.)
        Img = utils.make_grid(img_train.data, nrow=8, normalize=True, scale_each=True)
        Imgn = utils.make_grid(imgn_train.data, nrow=8, normalize=True, scale_each=True)
        Irecon = utils.make_grid(out_train.data, nrow=8, normalize=True, scale_each=True)
        writer.add_image('clean image', Img, epoch)
        writer.add_image('noisy image', Imgn, epoch)
        writer.add_image('reconstructed image', Irecon, epoch)
        # 保存生成器和判别器模型
        torch.save(model_G.state_dict(), os.path.join(opt.outf, 'net_G.pth'))
        torch.save(model_D.state_dict(), os.path.join(opt.outf, 'net_D.pth'))

if __name__ == "__main__":
    if opt.preprocess:
        if opt.mode == 'S':
            prepare_data(data_path='data', patch_size=40, stride=10, aug_times=1)
        if opt.mode == 'B':
            prepare_data(data_path='data', patch_size=50, stride=10, aug_times=2)
    main()
