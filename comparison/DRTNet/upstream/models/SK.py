import  torch.nn as nn
import torch
from functools import reduce


class SKConv(nn.Module):
    def __init__(self, in_channels, out_channels, groups,stride=1, M=2, r=2, L=32):
        """
        :param in_channels:  输入通道维度
        :param out_channels: 输出通道维度   原论文中 输入输出通道维度相同
        :param stride:  步长，默认为1
        :param M:  分支数
        :param r: 特征Z的长度，计算其维度d 时所需的比率（论文中 特征S->Z 是降维，故需要规定 降维的下界）
        :param L:  论文中规定特征Z的下界，默认为32
        采用分组卷积： groups = 32,所以输入channel的数值必须是group的整数倍
        """
        super(SKConv, self).__init__()
        d = max(in_channels // r, L)
        self.M = M
        self.out_channels = out_channels
        self.conv = nn.ModuleList()
        print('out_channels',out_channels) # 204
        self.groups = groups

        for i in range(M):
            self.conv.append(nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, stride, padding=1 + i, dilation=1 + i, groups=self.groups, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)))
        self.global_pool = nn.AdaptiveAvgPool2d(output_size=1)
        self.fc1 = nn.Sequential(nn.Conv2d(out_channels, d, 1, bias=False),
                                 nn.GroupNorm(1, d),
                                 nn.ReLU(inplace=True))  # 降维
        self.fc2 = nn.Conv2d(d, out_channels * M, 1, 1, bias=False)

        self.softmax = nn.Softmax(dim=1)
        self.temperature = 0.1
    def forward(self, input,input2):
        batch_size = input.size(0)
        output = []
        # 不同分支输入
        for i, conv in enumerate(self.conv): # 将每个分支的输出添加到列表中
            output.append(self.conv[i](input))
            output.append(self.conv[i](input2))
        # U = reduce(lambda x, y: x + y, output)

        # 融合部分
        U = sum(output)
        s = self.global_pool(U)  # 全局平均池化
        # print('s',s.shape) # [1, 380, 1, 1]
        z = self.fc1(s) # 降维，映射特征
        a_b = self.fc2(z)  # 升维，
        a_b = a_b.reshape(batch_size, self.M, self.out_channels, -1)  # 调整形状，变成两个全连接层的值

        # 添加温度
        a_b = a_b / self.temperature  # 应用温度缩放
        a_b = self.softmax(a_b)

        # 选择部分
        a_b = list(a_b.chunk(self.M, dim=1))
        a_b = list(map(lambda x: x.reshape(batch_size, self.out_channels, 1, 1),
                       a_b))
        V = list(map(lambda x, y: x * y, output,
                     a_b))
        V = reduce(lambda x, y: x + y,
                   V)
        return V
