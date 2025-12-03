# DnCNN-PyTorch NeiKos496小组作业

这是 TIP2017 论文 Beyond a Gaussian Denoiser: Residual Learning of Deep CNN for Image Denoising 的 PyTorch 实现(http://ieeexplore.ieee.org/document/7839189/)。

作者的 MATLAB 实现请见此处(https://github.com/cszn/DnCNN).

原代码是使用 PyTorch < 0.4 编写的，本次作业对低版本语法进行了优化使得适应 PyTorch 2.5.1

如何运行

1. 依赖项 
PyTorch(2.5.1)
torchvision
OpenCV for Python
HDF5 for Python
tensorboardX (PyTorch 的 TensorBoard 可视化工具)

2.训练 DnCNN-S/B (已知噪声水平的 DnCNN)

conda activate myenv
cd /home/xie/zzzmypython/DnCNNpytorch
cd C:\Users\12445\Desktop\DnCNN-

第一次加上预训练
--preprocess True 

python train.py --preprocess False --num_of_layers 17 --mode S --noiseL 50 --val_noiseL 50 --outf logs/DnCNN-S-50 --epochs 5 --milestone 3

后续训练：
nohup python train.py --preprocess True --num_of_layers 17 --mode S --noiseL 15 --val_noiseL 15 --outf logs/DnCNN-S-15 > logs/DnCNN-S-15.log 2>&1 &

nohup python train.py --preprocess False --num_of_layers 17 --mode S --noiseL 25 --val_noiseL 25 --outf logs/DnCNN-S-25  > logs/DnCNN-S-25.log 2>&1 &

nohup python train.py --preprocess False --num_of_layers 17 --mode S --noiseL 50 --val_noiseL 50 --outf logs/DnCNN-S-50 > logs/DnCNN-S-50.log 2>&1 &

nohup python train.py --preprocess False --num_of_layers 20 --mode B --val_noiseL 25 --outf logs/DnCNN-B  > logs/DnCNN-B.log 2>&1 &

测试set68
python test.py --num_of_layers 17 --logdir logs/DnCNN-S-15 --test_data "Set68" --test_noiseL 15
python test.py --num_of_layers 17 --logdir logs/DnCNN-S-25 --test_data "Set68" --test_noiseL 25
python test.py --num_of_layers 17 --logdir logs/DnCNN-S-50 --test_data "Set68" --test_noiseL 50

python test.py --num_of_layers 20 --logdir logs/DnCNN-B --test_data "Set68" --test_noiseL 15
python test.py --num_of_layers 20 --logdir logs/DnCNN-B --test_data "Set68" --test_noiseL 25
python test.py --num_of_layers 20 --logdir logs/DnCNN-B --test_data "Set68" --test_noiseL 50

### BSD68 平均 RSNR（最后两个模型为复现原代码）
|:-----------:|:-------:|:-------:|:---------------:|:---------------:|:---------------:|:---------------:|
| Noise Level | DnCNN-S | DnCNN-B | DnCNN-S-PyTorch | DnCNN-B-PyTorch | DnCNN-S-pytorch | DnCNN-B-pytorch |
|:-----------:|:-------:|:-------:|:---------------:|:---------------:|:---------------:|:---------------:|
|     15      |  31.73  |  31.61  |      31.71      |      31.60      |      31.70      |      30.90      |
|     25      |  29.23  |  29.16  |      29.21      |      29.15      |      29.17      |      28.34      |
|     50      |  26.23  |  26.23  |      26.22      |      26.20      |      26.16      |      25.68      |
|:-----------:|:-------:|:-------:|:---------------:|:---------------:|:---------------:|:---------------:|

# 1为加入注意力机制 （训练批次五轮，第三轮开始衰减）
|:-----------:|:--------:|:--------:|
| Noise Level | DnCNN-S1 | DnCNN-B1 | 
|:-----------:|:--------:|:--------:|
|     15      |  31.62   |  31.68   | 
|     25      |  29.08   |  29.19   |
|     50      |  25.87   |  26.20   |
|:-----------:|:--------:|:--------:|

测试set12
python test.py --num_of_layers 17 --logdir logs/DnCNN-S-15 --test_data "Set12" --test_noiseL 15
python test.py --num_of_layers 17 --logdir logs/DnCNN-S-25 --test_data "Set12" --test_noiseL 25
python test.py --num_of_layers 17 --logdir logs/DnCNN-S-50 --test_data "Set12" --test_noiseL 50

python test.py --num_of_layers 20 --logdir logs/DnCNN-B --test_data "Set12" --test_noiseL 15
python test.py --num_of_layers 20 --logdir logs/DnCNN-B --test_data "Set12" --test_noiseL 25
python test.py --num_of_layers 20 --logdir logs/DnCNN-B --test_data "Set12" --test_noiseL 50

### Set12 平均 PSNR
|:-----------:|:-------:|:-------:|:---------------:|:---------------:|:---------------:|:---------------:|
| Noise Level | DnCNN-S | DnCNN-B | DnCNN-S-PyTorch | DnCNN-B-PyTorch | DnCNN-S-pytorch | DnCNN-B-pytorch |
|:-----------:|:-------:|:-------:|:---------------:|:---------------:|:---------------:|:---------------:|
|     15      | 32.859  | 32.680  |     32.837      |     32.725      |     32.811      |     31.810      |
|     25      | 30.436  | 30.362  |     30.404      |     30.344      |     30.349      |     29.219      |
|     50      | 27.178  | 27.206  |     27.165      |     27.138      |     27.057      |     26.435      |
|:-----------:|:-------:|:-------:|:---------------:|:---------------:|:---------------:|:---------------:|

# 1为加入注意力机制 
|:-----------:|:---------:|:---------:|
| Noise Level | DnCNN-S1  | DnCNN-B1  | 
|:-----------:|:---------:|:---------:|
|     15      |  32.643   |  32.757   | 
|     25      |  30.200   |  30.430   |
|     50      |  26.661   |  27.184   |
|:-----------:|:---------:|:---------:|

tensorboard --logdir=C:\Users\12445\Desktop\DnCNNpytorch\logs\DnCNN-S-15
tensorboard --logdir=C:\Users\12445\Desktop\DnCNNpytorch\logs\DnCNN-S-25
tensorboard --logdir=C:\Users\12445\Desktop\DnCNNpytorch\logs\DnCNN-S-50
tensorboard --logdir=C:\Users\12445\Desktop\DnCNNpytorch\logs\DnCNN-B