from functools import reduce
import torch
from torch import nn, Tensor
from comparison.DRTNet.upstream.models.CGlayers import CG_Conv2d
from torch.nn import functional as F
# from semseg.models.layers import DropPath
import torch.nn.init as init
from comparison.DRTNet.upstream.models.EMA import ema
# -------------------- CustomDWConv_pooling
class CustomDWConv_pooling(nn.Module):
    def __init__(self, dim, kernel):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel, 1, padding='same', groups=dim)

        # Apply Kaiming initialization with fan-in to the dwconv layer
        init.kaiming_normal_(self.dwconv.weight, mode='fan_in', nonlinearity='relu')

    def forward(self, x: Tensor, H, W) -> Tensor:
        B, _, C = x.shape
        # print(x.shape)
        x = x.transpose(1, 2).view(B, C, H, W)
        # print(x.shape)
        x = self.dwconv(x)
        return x

# -------------------- CustomPWConv
class CustomPWConv(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.pwconv = nn.Conv2d(dim, dim, 1)
        self.bn = nn.BatchNorm2d(dim)
        # Initialize pwconv layer with Kaiming initialization
        init.kaiming_normal_(self.pwconv.weight, mode='fan_in', nonlinearity='relu')

    def forward(self, x: Tensor) -> Tensor:  # , H, W)
        B,  C,H,W = x.shape  # ([1, 1024, 62])
        # print('x',x.shape)
        # x = x.reshape(B, C, H, W)  # view  .transpose(1, 2)
        # print('reshape',x.shape)
        x = self.bn(self.pwconv(x))
        # print(x.shape)
        return x #x.flatten(2).transpose(1, 2)

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes , 1, bias=False) # 11.04 nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes, in_planes , 1, bias=False) # 11.04 nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # print(self.fc1(self.avg_pool(x)).shape)  # torch.Size([1, 3, 64, 64])  torch.Size([1, 3, 1, 1])
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # print('x', x.shape)  # c=3,6
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)

        x = torch.cat([avg_out, max_out], dim=1)
        # print('x',x.shape)  # c=2
        x = self.conv1(x)
        return self.sigmoid(x)

# [pool1,pool2,pool3,pool4]分别进行池化之后加和
# 使用1*1卷积进行特征融合
class MultiScaleFusion_Pooling(nn.Module):
    def __init__(self, in_channels, pool_sizes,len):
        super(MultiScaleFusion_Pooling, self).__init__()
        self.len = len
        self.pool_sizes = pool_sizes
        # self.groups = groups

        self.pool_layers = nn.ModuleList()
        self.conv_layers = nn.ModuleList()
        for size in pool_sizes:
            self.pool_layers.append(nn.AdaptiveAvgPool2d(output_size=(size, size)))
            self.conv_layers.append(CustomDWConv_pooling(dim=in_channels, kernel=1))

        self.PWconv_pool_in = CustomPWConv(in_channels)
        self.GConv = CG_Conv2d(in_channels, in_channels, 3, padding=1)
        self.PWconv_pool_out = CustomPWConv(in_channels)
        self.context = Context(in_channels,len=in_channels)

        self.conv2d = nn.Conv2d(in_channels, in_channels, 1, 1)
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=(1, 5), stride=1 ,padding=1),
            nn.Conv2d(in_channels, in_channels, kernel_size=(5, 1), stride=1,padding=1)
        )

        self.conv3d = nn.Sequential(
            nn.Conv3d(in_channels, in_channels, kernel_size=(1,3,3), stride=1, padding=(0,1,1)),
            nn.Conv3d(in_channels, in_channels, kernel_size=(3, 1, 1), stride=1, padding=(1, 0, 0))
          )
        self.ca = ChannelAttention(in_channels)
        self.ca1 = ChannelAttention(in_channels)
        self.sa = SpatialAttention()

    def forward(self, x: Tensor) -> Tensor:  # H, W

        # 3D
        # data_3d = x.unsqueeze(2).repeat(1, 1, self.len, 1, 1)
        # data_3d = self.conv3d(data_3d).mean(dim=2)  # torch.Size([1, c, len, 32, 32]) mean沿着len维度求均值
        # print(x.shape)
        x_res = x

        data_3d = self.conv3(x)
        x = self.conv2d(x)
        # print(x.shape)
        x = data_3d + x
        # print(x.shape)
        B,C, H,W = x.shape  # 11.1

        x_cbam = x

        x = self.PWconv_pool_in(x)  # , H, W'
        # print('pool x', x.shape)
        # x = self.GConv(x)
        # print('pool x',x.shape)
        # Parallel pooling
        x_in = x.reshape(B, C, H, W)  # .transpose(1, 2)  view-->reshape
        x_add = 0
        for i in range(len(self.pool_layers)):
            pooled_feature = self.conv_layers[i](self.pool_layers[i](x_in).flatten(2).transpose(1, 2),
                                                 self.pool_sizes[i], self.pool_sizes[i])
            x2 = F.interpolate(pooled_feature, size=(H, W), mode='bilinear')
            x_add += x2

        # 输出
        # print('x_add',x_add.shape) #()1,211,128,128
        x_out = self.PWconv_pool_out(x_add)  # , H, W
        x_out = x_res + x_out
        # print('x_out',x_out.shape)
        x_cbam = x_cbam * self.ca(x_cbam)
        x_cbam = x_cbam * self.sa(x_cbam)
        x_weight = self.context(x_out)  # 11.14
        x_out = x_out + x_in +  x_cbam
        x_out = x_weight * x_out

        return x_out


import torch
import torch.nn as nn
import torch.nn.functional as F


class Context(nn.Module):
    def __init__(self, in_channels,len,groups=36,M=2, r=16, L=32):  # Botswana 36
        super(Context, self).__init__()
        self.len = len
        # self.conv3d = nn.Conv3d(in_channels,1,(1,1,1),1,)
        # 1x1x1 卷积
        self.conv1x1 = nn.Conv2d(in_channels=in_channels, out_channels=1, kernel_size=1)

        # 初始化卷积权重
        # self.conv1x1.weight.data.fill_(0)
        # self.conv1x1.bias.data.fill_(0)

        # 平均池化
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
##############################
        self.groups = groups
        self.in_channels = in_channels
        self.M = M
        d = max(in_channels // r, L)
        self.global_pool = nn.AdaptiveAvgPool2d(output_size=1)
        self.fc1 = nn.Sequential(nn.Conv2d(in_channels, d, 1, bias=False),
                                 nn.BatchNorm2d(d),
                                 nn.ReLU(inplace=True))  # 降维
        self.fc2 = nn.Conv2d(d, in_channels * M, 1, 1, bias=False)
        self.conv = nn.ModuleList()
        for i in range(M):
            self.conv.append(nn.Sequential(
                nn.Conv2d(in_channels, in_channels, 3, 1, padding=1 + i, dilation=1 + i,  bias=False),
                # nn.Conv2d(in_channels, in_channels, 3, 1, padding=1 + i, dilation=1 + i, groups=self.groups,bias=False),

                nn.BatchNorm2d(in_channels),
                nn.ReLU(inplace=True)))
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        b, c, h, w = x.shape
        # 第一个分支: reshape 操作
        x1 = x.view(c, -1)  # Shape: [c, b*h*w]
        # 第二个分支: 1x1卷积 + reshape + softmax
        x2 = self.conv1x1(x)  # Shape: [1, b, h, w]
        x2 = x2.view(b * h * w, 1, 1, 1)  # Reshape to [b*h*w, 1, 1, 1]
        x2 = F.softmax(x2, dim=0)  # Softmax over b*h*w
        x2 = x2.view(b * h * w, 1)  # [b*h*w, 1]
        # 将两个分支相乘
        x_combined = torch.matmul(x1, x2).unsqueeze(-1).unsqueeze(-1)  # Shape: [c, 1, 1, 1]
        # 平均池化并经过sigmoid
        x_combined = self.avg_pool(x_combined)  # Shape: [c, 1, 1, 1]
        x_combined = torch.sigmoid(x_combined).permute(1,0,2,3)  # Sigmoid操作
        return x_combined
    ######################################################### 11.14改

        # print('s',s.shape) # [1, 380, 1, 1]
        # 不同分支输入
        # output = []
        # batch_size = x.size(0)
        # for i, conv in enumerate(self.conv):  # 将每个分支的输出添加到列表中
        #     output.append(self.conv[i](x))
        # U = sum(output)
        # s = self.global_pool(U)  # 全局平均池化  torch.Size([1, 32, 1, 1])
        # z = self.fc1(s)  # 降维，映射特征
        # a_b = self.fc2(z)  # 升维，
        # a_b = a_b.reshape(batch_size, self.M, self.in_channels, -1)  # 调整形状，变成两个全连接层的值
        # a_b = self.softmax(a_b)
        # # print("a_b.shape before reshape:", a_b.shape)  # a_b.shape before reshape: torch.Size([1, 2, 432, 1024])
        # x_out = self.conv1x1(x)
        # # 选择部分
        # a_b = list(a_b.chunk(self.M, dim=1))
        # a_b = list(map(lambda x_: x_.reshape(batch_size, self.in_channels, 1, 1), a_b))
        # V = list(map(lambda x_, y: x_ * y, x_out, a_b))
        # V = reduce(lambda x_, y: x_ + y, V)
        # print('V', V.shape)
        return V




# # 示例：输入尺寸为 [1, 308, 32, 32]
# input_tensor = torch.randn(1, 308, 32, 32)
#
# # 初始化模型
# model = Context(c=308,len=308)
#
# # 前向传播
# output = model(input_tensor)
#
# # 输出尺寸
# print(f'输出权重w的尺寸: {output.shape}')

######################################## 假设输入通道数为 62，池化大小为 [1, 2, 4, 8]
# in_channels = 62
# pool_sizes = [1, 2, 4, 8]
#
# # 创建 MultiScaleFusion_Pooling 实例
# multi_scale_fusion = MultiScaleFusion_Pooling(in_channels, pool_sizes, len = in_channels)
#
# # 模拟输入数据
# # 这里假设 batch size 为 1，特征维度为 62，序列长度（H 和 W）可以设置为 32
# batch_size = 1
# H, W = 32, 32
# input_tensor = torch.randn(batch_size, in_channels, H ,W)  # 形状为 (B, C, H*W)
# # print('input_tensor.shape', input_tensor.shape)
#
# # 将输入传递给模型
# output = multi_scale_fusion(input_tensor) # , H, W
#
# # 打印输出形状
# print("Output shape:", output.shape)
