# 矩形注意力窗口，竖直形状和水平形状各一半 参考论文WcASSR


import torch.nn as nn
from comparison.DRTNet.upstream.models.IntmdSequential import IntermediateSequential
import numpy as np

class SelfAttention(nn.Module):
    """
    加窗口注意力。3.20修改
    修改后的SelfAttention模块中，local_attetion方法将窗口掩码应用于注意力矩阵，以在窗口内强制执行局部注意力。
    window_mask方法生成一个二进制掩码，该掩码在窗口区域中为零，在其他位置为一。窗口大小由window_size参数指定.
    """
    def __init__(
        self, dim, heads=8, qkv_bias=False, qk_scale=None, dropout_rate=0.0, window_size=16
    ):
        super().__init__()
        self.num_heads = heads
        head_dim = dim // heads
        self.scale = qk_scale or head_dim ** -0.5
        self.window_size = window_size

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(dropout_rate)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout_rate)
        self.out = {}

    def forward(self, input):
        a = x = input['x']
        b = y = input['y']

        B, N, C = x.shape
        qkv1 = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        qkv2 = (
            self.qkv(y)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv1[0], qkv1[1], qkv1[2]
        q1, k1, v1 = qkv2[0], qkv2[1], qkv2[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        # attn = self.local_attention(attn)  # Apply local attention within the window
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        attn1 = (q1 @ k1.transpose(-2, -1)) * self.scale
        # attn1 = self.local_attention(attn1)  # Apply local attention within the window
        attn1 = attn1.softmax(dim=-1)
        attn1 = self.attn_drop(attn1)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        x = x + a

        y = (attn1 @ v1).transpose(1, 2).reshape(B, N, C)
        y = self.proj(y)
        y = self.proj_drop(y)
        y = y + b

        self.out['x'], self.out['y'] = x, y
        return self.out
    def local_attention(self, attn):
        B, H, N, _ = attn.shape
        window_size = self.window_size
        window = self.window_mask((N, N), window_size).to(attn.device)
        attn = attn.masked_fill(window == 0, float("-inf"))
        return attn
    def window_mask(self, shape, window_size):
        mask = torch.ones(shape)
        mask[:window_size, :window_size] = 0
        mask[-window_size:, -window_size:] = 0
        return mask

class SelfAttention_reatangle(nn.Module):
    """
    矩形窗口注意力。3.21修改。添加两个新的参数 vertical_windows 和 horizontal_windows，这些参数表示竖直和横向矩形窗口的数量
    """
    def __init__(
        self, dim, heads=8, qkv_bias=False, qk_scale=None, dropout_rate=0.0,
        window_size=7, vertical_windows=4, horizontal_windows=4
    ):
        super().__init__()
        self.num_heads = heads
        head_dim = dim // heads  # 128/8=16
        self.scale = qk_scale or head_dim ** -0.5
        self.window_size = window_size
        self.vertical_windows = vertical_windows or heads // 2
        self.horizontal_windows = horizontal_windows or heads // 2

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(dropout_rate)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout_rate)
        self.out = {}

    def forward(self, input):
        """
        修改SelfAttention类的forward方法，根据竖直和横向窗口的数量生成对应的二进制掩码
        """
        a = x = input['x']
        b = y = input['y']

        B, N, C = x.shape
        qkv1 = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        qkv2 = (
            self.qkv(y)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv1[0], qkv1[1], qkv1[2]
        q1, k1, v1 = qkv2[0], qkv2[1], qkv2[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = self.local_attention(attn)  # Apply local attention within the window
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        attn1 = (q1 @ k1.transpose(-2, -1)) * self.scale
        attn1 = self.local_attention(attn1)  # Apply local attention within the window
        attn1 = attn1.softmax(dim=-1)
        attn1 = self.attn_drop(attn1)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        x = x + a

        y = (attn1 @ v1).transpose(1, 2).reshape(B, N, C)
        y = self.proj(y)
        y = self.proj_drop(y)
        y = y + b

        self.out['x'], self.out['y'] = x, y
        return self.out


    def local_attention(self, attn):  # 3.22 遍历每个窗口
        B, H, N, _ = attn.shape
        window_size = self.window_size
        windows = self.window_mask_rect((N, N))#.to(attn.device)  # 更新window_mask_rect的调用
        for window in windows:
            window = window.to(attn.device)
            attn = attn.masked_fill(window == 0, -1e9)
            softmax_attn = attn.softmax(dim = -1)
            attn = softmax_attn
        return attn


    def window_mask_rect(self, shape):  #############3.22修改
        """
        将窗口内的掩码值设置为0。窗口的大小是4×16，移动的步长是  。
        窗口移动方式是从左到右，从上到下，如果窗口移动到的位置超出了掩码的范围，那么只有在掩码范围内的部分会被设置为0。
        整个掩码矩阵上移动的窗口，窗口内的掩码值为0，窗口外的掩码值为1。
        这个窗口从左上角开始，每次向右移动一个h_step，移动到右边界后再向下移动一个v_step，直到覆盖整个掩码矩阵
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 获取设备信息
        # mask = torch.ones(shape)
        v_step = 8 #16  # 垂直步长
        h_step = 16#S 32  # 水平步长
        v_window = 8  # 垂直窗口大小
        h_window = 16  # 水平窗口大小
        masks = []

        # for i in range(0, shape[0], v_step):
        #     for j in range(0, shape[1], h_step):
        #         mask = torch.ones(shape).to(device)
        #         mask[i:i + v_window, j:j + h_window] = 0
        #         masks.append(mask)
        mask_h = torch.ones(shape).to(device)  # 上半部分水平掩码
        mask_h[:v_window,:]=0
        masks.append(mask_h)
        ############
        mask_h_shift = torch.ones(shape).to(device)
        mask_h_shift[v_step:v_step+v_window, :] = 0
        masks.append(mask_h_shift)

        mask_v = torch.ones(shape).to(device)
        mask_v[:,:h_window] = 0
        masks.append(mask_v)
        #####################
        mask_v_shift = torch.ones(shape).to(device)
        mask_v_shift[:, h_step:h_step+h_window] = 0
        masks.append(mask_v_shift)

        return masks



# class SelfAttention(nn.Module):  # 加窗 3.21
#     """
#     加窗口注意力。3.20修改
#     修改后的SelfAttention模块中，local_attetion方法将窗口掩码应用于注意力矩阵，以在窗口内强制执行局部注意力。
#     window_mask方法生成一个二进制掩码，该掩码在窗口区域中为零，在其他位置为一。窗口大小由window_size参数指定.
#     """
#     def __init__(
#         self, dim, heads=8, qkv_bias=False, qk_scale=None, dropout_rate=0.0, window_size=7
#     ):
#         super().__init__()
#         self.num_heads = heads
#         head_dim = dim // heads
#         self.scale = qk_scale or head_dim ** -0.5
#         self.window_size = window_size
#
#         self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
#         self.attn_drop = nn.Dropout(dropout_rate)
#         self.proj = nn.Linear(dim, dim)
#         self.proj_drop = nn.Dropout(dropout_rate)
#         self.out = {}
#
#
#
#     def forward(self, input):
#         a = x = input['x']
#         b = y = input['y']
#
#         B, N, C = x.shape
#         qkv1 = (
#             self.qkv(x)
#             .reshape(B, N, 3, self.num_heads, C // self.num_heads)
#             .permute(2, 0, 3, 1, 4)
#         )
#         qkv2 = (
#             self.qkv(y)
#             .reshape(B, N, 3, self.num_heads, C // self.num_heads)
#             .permute(2, 0, 3, 1, 4)
#         )
#         q, k, v = qkv1[0], qkv1[1], qkv1[2]
#         q1, k1, v1 = qkv2[0], qkv2[1], qkv2[2]
#
#         attn = (q @ k.transpose(-2, -1)) * self.scale
#         attn = self.local_attention(attn)  # Apply local attention within the window
#         attn = attn.softmax(dim=-1)
#         attn = self.attn_drop(attn)
#
#         attn1 = (q1 @ k1.transpose(-2, -1)) * self.scale
#         attn1 = self.local_attention(attn1)  # Apply local attention within the window
#         attn1 = attn1.softmax(dim=-1)
#         attn1 = self.attn_drop(attn1)
#
#         x = (attn @ v).transpose(1, 2).reshape(B, N, C)
#         x = self.proj(x)
#         x = self.proj_drop(x)
#         x = x + a
#
#         y = (attn1 @ v1).transpose(1, 2).reshape(B, N, C)
#         y = self.proj(y)
#         y = self.proj_drop(y)
#         y = y + b
#
#         self.out['x'], self.out['y'] = x, y
#         return self.out
#
#     def local_attention(self, attn):
#         B, H, N, _ = attn.shape
#         window_size = self.window_size
#         window = self.window_mask((N, N), window_size).to(attn.device)
#         attn = attn.masked_fill(window == 0, float("-inf"))
#         return attn
#
#     def window_mask(self, shape, window_size):
#         mask = torch.ones(shape)
#         mask[:window_size, :window_size] = 0
#         mask[-window_size:, -window_size:] = 0
#         return mask


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x)


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
        self.input = {}

    def forward(self, x):
        self.input['x'] = self.norm(x['x'])
        self.input['y'] = self.norm(x['y'])
        return self.fn(self.input)


class PreNormDrop(nn.Module):
    def __init__(self, dim, dropout_rate, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.fn = fn  # SelfAttention((qkv): Linear(in_features=128, out_features=384, bias=False)
        self.input = {}
        self.out = {}

    def forward(self, x):
        self.input['x'] = self.norm(x['x']) # 1,64,128
        self.input['y'] = self.norm(x['y'])
        self.out['x'], self.out['y'] = self.dropout(self.fn(self.input)['x']), self.dropout(self.fn(self.input)['y'])  # 1,1024,64
        return self.out


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout_rate):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            # nn.GELU(),改了
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(p=dropout_rate),
        )
        self.out = {}

    def forward(self, input):
        self.out['x'], self.out['y'] = self.net(input['x']), self.net(input['y'])
        return self.out


class LearnedPositionalEncoding(nn.Module):
    def __init__(self, max_position_embeddings, embedding_dim, seq_length):
        super(LearnedPositionalEncoding, self).__init__()

        self.position_embeddings = nn.Parameter(torch.zeros(1, seq_length, embedding_dim)) #8x #  embedding_dim 64

    def forward(self, x, position_ids=None):

        position_embeddings = self.position_embeddings
        # print('x:',x.shape) # ([1, 64, 128])
        # print('position_embeddings:',position_embeddings.shape) # ([1, 64, 128])
        return x + position_embeddings


import torch
import math


class SinusoidalPositionalEncoding(torch.nn.Module):
    def __init__(self, d_model, max_len=5000):
        """
        正弦和余弦位置编码初始化
        :param d_model: 模型的维度
        :param max_len: 序列的最大长度
        """
        super(SinusoidalPositionalEncoding, self).__init__()

        # 初始化一个形状为 (max_len, d_model) 的位置编码矩阵
        pe = torch.zeros(max_len, d_model)
        # 生成一个形状为 (max_len, 1) 的位置索引矩阵
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        # 根据公式计算div_term
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        # 对位置矩阵的偶数位置填充sin值
        pe[:, 0::2] = torch.sin(position * div_term)
        # 对位置矩阵的奇数位置填充cos值
        pe[:, 1::2] = torch.cos(position * div_term)

        # 在第0维增加一维，然后转置，使其形状为 (1, max_len, d_model)
        pe = pe.unsqueeze(0).transpose(0, 1)
        # 将pe注册为buffer，表示它不是模型的参数，而是模型的一部分，不会在训练中更新
        self.register_buffer('pe', pe)

    def forward(self, x):
        # 将位置编码加到输入张量上
        return x + self.pe[:x.size(0), :]

##############################################
class TransformerModel_rectangle(nn.Module):
    def __init__(
            self,
            map_size,
            M_channel,
            dim,
            depth,
            heads,
            mlp_dim,
            dropout_rate=0.1,
            attn_dropout_rate=0.1,
    ):
        super().__init__()
        layers = []
        for _ in range(depth):
            layers.extend(
                [
                    Residual(
                        PreNormDrop(
                            dim,
                            dropout_rate,
                            # SelfAttention(dim, heads=heads, dropout_rate=attn_dropout_rate, window_size=16),
                            # window_size=7加窗口，3.20
                            SelfAttention_reatangle(dim, heads=heads, dropout_rate=attn_dropout_rate),  # 修改对应注意力模块
                        )
                    ),
                    Residual(
                        PreNorm(dim, FeedForward(dim, mlp_dim, dropout_rate))
                    ),
                ]
            )
            # dim = dim / 2
        self.net = IntermediateSequential(*layers)
        self.input = {}
        self.output = {}

        self.map_size = map_size
        self.linear_encoding = nn.Linear(M_channel, dim)
        self.linear_encoding_de = nn.Linear(dim, M_channel)
        self.position_encoding = LearnedPositionalEncoding(M_channel, dim, map_size*map_size)
        #self.transformersingle = TransformerModelsingle()

    def forward(self, x, y):
        # self.input['x'] = x
        # self.input['y'] = y
        self.input['x'] = x
        self.input['y'] = y

        x_ = x.permute(0, 2, 3, 1).contiguous()  # contiguous() 深拷贝，强制拷贝一份tensor
        y_ = y.permute(0, 2, 3, 1).contiguous()# 1,8,8,206
        x = x_.view(x_.size(0), x_.size(2)*x_.size(1), -1)
        y = y_.view(y_.size(0), y_.size(2) * y_.size(1), -1)
        self.input['x'] = self.position_encoding(self.linear_encoding(x))  # 分别应用线性映射和位置编码，并保存在input字典中
        self.input['y'] = self.position_encoding(self.linear_encoding(y))
        results = self.net(self.input)    # 残差结构、注意力计算
        x, y = results['x'], results['y']
        x = self.linear_encoding_de(x).permute(0, 2, 1).contiguous()  # 对x应用反向的线性映射和维度变换操作
        self.output['x'] = x.view(x.size(0), x.size(1), self.map_size, self.map_size) # 将处理后的x重新变换成特定形状，并保存在output字典中
        y = self.linear_encoding_de(y).permute(0, 2, 1).contiguous()  # 同上
        self.output['y'] = y.view(y.size(0), y.size(1), self.map_size, self.map_size)
        self.output['z'] = self.output['x'] + self.output['y']  # 将x和y相加得到z，并保存在output字典中
        #self.output['z'] = self.transformersingle(self.output['z'])
        return self.output  # 结果以字典的形式返回，其中包括了经过处理的 x、y 以及它们的组合结果 z






