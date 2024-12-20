#AlexNet 
import torch 
from torch import nn 
from d2l import torch as d2l
 
#容量控制和预处理
net = nn.Sequential(
    #nn.Sequential 构建顺序网络结构
    #nn.Conv2d（输入通道数，输出通道数，卷积核大小，步长，填充）
    #输出可以理解为提取的特征数，填充是在周围填充的空格数
    #nn.MaxPool2d（池化核大小，步长）
    #nn.ReLU() ReLU非线性函数
    #nn.Flatten 展平
    #nn.Linear（输入特征数，输出特征数），全连接
    #nn.Dropout（损失概率） 防止过拟合
    #torch.randn(N,C,H,W):输入量，通道数，高，宽
    
    #使用11x11的更大窗口来捕捉对象，同时步幅为4，以减少输出高度和宽度，输出通道数目远大于LeNet
    nn.Conv2d(1, 96, kernel_size=11, stride=4, padding=1), nn.ReLU(),
    ###此处1正在imagenet为3
    nn.MaxPool2d(kernel_size=3, stride=2),
    #减少卷积窗口，使用填充为2来使输入与输出高和宽一致，且增大输出通道数
    ###其实不知为何不用3，1
    nn.Conv2d(96, 256, kernel_size=5, padding=2), nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2),
    #使用三个连续的卷积层和较小的卷积窗口，除了最后的卷积层，输出通道数量进一步增强，在前两个卷积层之后，汇聚层不用减少输入的高度和宽度
    nn.Conv2d(256, 384, kernel_size=3, padding=1), nn.ReLU(),
    nn.Conv2d(384, 384, kernel_size=3, padding=1), nn.ReLU(),
    nn.Conv2d(384, 256, kernel_size=3, padding=1), nn.ReLU(),
    ###此处可改为：384,384，但后面要调整
    ###数据集过小，还是减少卷积
    nn.MaxPool2d(kernel_size=3, stride=2),
    nn.Flatten(), #将数据拉伸成一维
    #全连接层的输出数量是LeNet中的好几倍，用dropout层来减轻过拟合
    nn.Linear(6400, 4096), nn.ReLU(),
    nn.Dropout(p=0.5),
    nn.Linear(4096, 4096), nn.ReLU(),
    nn.Dropout(p=0.5),
    #输出层，使用的Fashion-MNIST，所以类别数为10
    nn.Linear(4096, 10)
    ###此处10正在imagenet为100
)
x = torch.randn(1, 1, 224, 224)
for layer in net:
    x = layer(x)
    print(layer.__class__.__name__,'output shape:\t', x.shape)
 
#读取数据集
'''使用的Fashion-MNIST数据，但Fashion-MNIST图像分辨率(28x28)低于ImageNet图像，所以将其增加到224x224'''
batch_size = 128
###batch_size 样本数量（一批次学习的数量）
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=224)
###train_iter：训练集，test_iter：测试集，resize：将28*变为224*

#训练AlexNet
lr, num_epochs = 0.01, 10 #比LeNet使用更小学习率是因为AlexNet网络更深更广，图像分辨率更高，训练更昂贵
###lr：学习率，num_epochs：训练轮数（进行多少次总体学习）

d2l.train_ch6(net, train_iter, test_iter, num_epochs, lr, d2l.try_gpu())
d2l.plt.show()