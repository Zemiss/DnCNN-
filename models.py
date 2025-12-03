import torch
import torch.nn as nn

# ===========================================================================================
# 【重点修改1：加入SE注意力机制模块】
# Squeeze-and-Excitation Block（SE块）注意力机制
# 
# 核心作用：通过自适应地重新校准通道特征响应，使模型能够更好地关注重要特征通道
# 
# 对DnCNN的性能提升：
# - 使得盲去噪模型DnCNN-B性能大幅提升，甚至超越了非盲去噪模型DnCNN-S
# - 盲去噪 + 注意力机制：注意力机制使得模型能够更好地隔离不同sigma值带来的噪声成分，
#   并独立地预测它们。这种对残差的精细化预测能力，让盲去噪模型在捕捉噪声特征方面
#   变得极其强大，最终超过了仅针对单一、固定残差训练的非盲模型
# 
# 实现原理：
# 1. Squeeze（压缩）：通过全局平均池化将空间维度压缩为1x1，得到每个通道的全局特征描述
# 2. Excitation（激励）：通过两层全连接网络学习通道间的相关性，生成每个通道的权重
# 3. Scale（缩放）：将学到的权重应用到原始特征图上，实现通道特征的重新校准
# ===========================================================================================
class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        # 全局平均池化，将特征图压缩为1x1
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # 通道注意力机制的全连接层：降维->ReLU激活->升维->Sigmoid激活
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),  # 降维，减少参数量
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),  # 升维，恢复通道数
            nn.Sigmoid()  # 生成0-1之间的权重系数
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        # Squeeze：全局平均池化，得到每个通道的全局特征
        y = self.avg_pool(x).view(b, c)
        # Excitation：通过全连接网络学习通道权重
        y = self.fc(y).view(b, c, 1, 1)
        # Scale：将权重应用到原始特征图，实现特征重新校准
        return x * y.expand_as(x)

class DnCNN(nn.Module):
    """
    DnCNN（去噪卷积神经网络）主模型
    
    网络结构：Conv + (Conv-BN-SE-ReLU) × N + Conv
    - 第一层：卷积层 + ReLU激活
    - 中间层：卷积 + 批归一化 + SE注意力 + ReLU（重复N次）
    - 最后层：卷积层（输出残差噪声）
    """
    def __init__(self, channels, num_of_layers=20):
        super(DnCNN, self).__init__()
        kernel_size = 3  # 3x3卷积核
        padding = 1      # 保持特征图尺寸不变
        
        # ===========================================================================================
        # 【重点修改2：增加网络宽度】
        # 原始DnCNN使用64个特征通道，这里改为128个特征通道
        # 增加网络宽度可以提升模型的特征表达能力，但也会增加计算量和参数量
        # ===========================================================================================
        features = 128  # 原来是64
        
        self.features = features # 为了在判别器中使用

        layers = []
        
        # 第一层：输入层，将输入图像映射到特征空间
        layers.append(nn.Conv2d(in_channels=channels, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False))
        layers.append(nn.ReLU(inplace=True))
        
        # 中间层：特征提取层（重复num_of_layers-2次）
        for _ in range(num_of_layers-2):
            layers.append(nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False))
            layers.append(nn.BatchNorm2d(features))  # 批归一化，稳定训练
            
            # ===========================================================================================
            # 【重点修改3：在每个卷积块中加入SE注意力机制模块】
            # 在传统的Conv-BN-ReLU结构中插入SE注意力模块，形成Conv-BN-SE-ReLU结构
            # SE模块能够学习通道间的依赖关系，自适应地调整特征通道的重要性
            # 这使得网络能够更有效地提取和利用特征信息，显著提升去噪性能
            # ===========================================================================================
            layers.append(SEBlock(features))  # SE注意力机制
            
            layers.append(nn.ReLU(inplace=True))
        
        # 最后一层：输出层，将特征映射回图像空间（预测残差噪声）
        layers.append(nn.Conv2d(in_channels=features, out_channels=channels, kernel_size=kernel_size, padding=padding, bias=False))
        
        # 将所有层组合成序列模型
        self.dncnn = nn.Sequential(*layers)
    
    def forward(self, x):
        """
        前向传播
        输入：含噪声图像
        输出：预测的噪声残差（通过输入减去输出即可得到去噪后的图像）
        """
        out = self.dncnn(x)
        return out

# ===========================================================================================
# 【加入对抗神经网络的判别器】
# 判别器网络：用于区分去噪后的图像和真实图像
#
# 网络结构：Conv-BN-ReLU -> Conv-BN-ReLU -> Conv-BN-ReLU -> FC
# - 多层卷积层提取图像特征
# - 全连接层进行分类判别
# ===========================================================================================
class Discriminator(nn.Module):
    def __init__(self, channels, features=128):
        super(Discriminator, self).__init__()
        self.features = features
        kernel_size = 3
        padding = 1

        self.conv1 = nn.Conv2d(in_channels=channels, out_channels=features, kernel_size=kernel_size, padding=padding, bias=False)
        self.bn1 = nn.BatchNorm2d(features)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(in_channels=features, out_channels=features*2, kernel_size=kernel_size, padding=padding, bias=False)
        self.bn2 = nn.BatchNorm2d(features*2)
        self.relu2 = nn.ReLU(inplace=True)

        self.conv3 = nn.Conv2d(in_channels=features*2, out_channels=features*4, kernel_size=kernel_size, padding=padding, bias=False)
        self.bn3 = nn.BatchNorm2d(features*4)
        self.relu3 = nn.ReLU(inplace=True)

        # 自适应平均池化，将特征图大小统一
        self.avg_pool = nn.AdaptiveAvgPool2d((4, 4))

        # 全连接层，用于判别真假
        self.fc = nn.Sequential(
            nn.Linear(features * 4 * 4 * 4, 1),  # 输出一个概率值，表示图像为真的概率
            nn.Sigmoid()  # Sigmoid激活函数，将输出限制在0-1之间
        )

    def forward(self, x):
        """
        前向传播
        输入：图像（去噪后的图像或真实图像）
        输出：图像为真的概率
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)

        # 自适应平均池化
        x = self.avg_pool(x)

        # 展开特征图
        x = x.view(x.size(0), -1)

        # 全连接层
        x = self.fc(x)

        return x
