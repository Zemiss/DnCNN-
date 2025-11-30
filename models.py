import torch
import torch.nn as nn

# Squeeze-and-Excitation (SE) 注意力模块
class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        # Squeeze 操作: 全局平均池化
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # Excitation 操作: 两个全连接层
        self.fc = nn.Sequential(
            # 可以尝试减小 reduction 比例（如 8 或 4）来增强注意力（增加参数）
            nn.Linear(channel, channel // reduction, bias=False), 
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        # 将注意力权重乘以输入特征
        return x * y.expand_as(x)

class DnCNN_SE(nn.Module):
    def __init__(self, channels, num_of_layers=17, reduction=16):
        super(DnCNN_SE, self).__init__()
        kernel_size = 3
        padding = 1
        features = 64
        layers = []
        
        # 第一层: Conv + ReLU (不加BN，为了不改变输入数据的均值)
        layers.append(nn.Conv2d(in_channels=channels, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False))
        layers.append(nn.ReLU(inplace=True))

        # 中间层: Conv + BN + SE + ReLU
        # 调整了SELayer的位置，将其放在BN之后、ReLU之前
        for _ in range(num_of_layers - 2):
            # Conv
            layers.append(nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False))
            # BN
            layers.append(nn.BatchNorm2d(features))
            # 新增: SELayer (在 BN 之后)
            layers.append(SELayer(features, reduction=reduction)) 
            # ReLU
            layers.append(nn.ReLU(inplace=True)) 
            
        # 最后一层: Conv (输出预测的噪声)
        layers.append(nn.Conv2d(in_channels=features, out_channels=channels, kernel_size=kernel_size, padding=padding, bias=False))
        
        self.dncnn = nn.Sequential(*layers)

    def forward(self, x):
        # DnCNN 学习残差（噪声）
        # self.dncnn(x) 得到的是预测的噪声图 (residual_map)
        residual_map = self.dncnn(x) 
        
        # 核心：残差学习。去噪图像 = 带噪图像 - 预测的噪声
        return x - residual_map