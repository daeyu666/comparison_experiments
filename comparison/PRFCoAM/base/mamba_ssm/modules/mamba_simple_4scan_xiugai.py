# Copyright (c) 2023, Tri Dao, Albert Gu.

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
device = torch.device('cuda:0')
from einops import rearrange, repeat

try:
    from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
except ImportError:
    causal_conv1d_fn, causal_conv1d_update = None
try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, mamba_inner_fn, bimamba_inner_fn, \
        mamba_inner_fn_no_out_proj
except ImportError:
    selective_scan_fn, mamba_inner_fn, bimamba_inner_fn, mamba_inner_fn_no_out_proj = None, None, None, None

try:
    from mamba_ssm.ops.triton.selective_state_update import selective_state_update
except ImportError:
    selective_state_update = None

try:
    from mamba_ssm.ops.triton.layernorm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None


class ChannelAttentionModule(nn.Module):
    def __init__(self, channel, reduction=8):
        super(ChannelAttentionModule, self).__init__()
        mid_channel = channel // reduction
        # 使用自适应池化缩减map的大小，保持通道不变
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # (1) 表示输出的高度和宽度都被设置为 1。
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.shared_MLP = nn.Sequential(
            nn.Linear(in_features=channel, out_features=mid_channel),
            nn.ReLU(),
            nn.Linear(in_features=mid_channel, out_features=channel)
        )
        self.sigmoid = nn.Sigmoid()
        # self.act=SiLU()

    def forward(self, x):
        b, _, h, w = x.shape
        # print('x.shape:', x.shape)
        avgout = self.shared_MLP(self.avg_pool(x).view(x.size(0), -1)).unsqueeze(2)
        maxout = self.shared_MLP(self.max_pool(x).view(x.size(0), -1)).unsqueeze(2)
        out = self.sigmoid(avgout + maxout)
        # print('out.shape:', out.shape)
        xout = out.reshape(b, 1, -1)
        # xout = out.reshape(b, -1, 1, 1).repeat(1, 1, h, w)
        # print("xout.shape:", xout.shape)
        return xout

# 空间注意力模块
class SpatialAttentionModule(nn.Module):
    def __init__(self):
        super(SpatialAttentionModule, self).__init__()
        self.conv2d = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3)
        # self.act=SiLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # map尺寸不变，缩减通道
        _, c, _, _ = x.shape
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avgout, maxout], dim=1)
        out = self.sigmoid(self.conv2d(out))
        # out = out
        # print(xout.shape)
        return out


class Mamba(nn.Module):
    def __init__(
            self,
            d_model,
            d_state=16,
            d_conv=4,
            expand=1,
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            conv_bias=True,
            bias=False,
            use_fast_path=True,  # Fused kernel options
            layer_idx=None,
            device=None,
            dtype=None,
            bimamba_type="v2",
            if_devide_out=False,
            init_layer_scale=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.Cin = 102
        self.Cout = 4
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = d_model
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.use_fast_path = use_fast_path
        self.layer_idx = layer_idx
        self.bimamba_type = bimamba_type
        self.if_devide_out = if_devide_out

        self.init_layer_scale = init_layer_scale
        if init_layer_scale is not None:
            self.gamma = nn.Parameter(init_layer_scale * torch.ones((d_model)), requires_grad=True)

        self.conv2d_1 = nn.Conv2d(
            in_channels=d_model,
            out_channels=d_model,
            bias=conv_bias,
            kernel_size=3,
            padding=1,
        )

        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
            **factory_kwargs,
        )

        self.activation = "silu"
        self.act = nn.SiLU()
        # print("self.diner.shape:", self.d_inner)
        # print("self.dt_rank + self.d_state * 2:", self.dt_rank, self.d_state * 2)
        self.x_proj = nn.Linear(
            self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
        )
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = self.dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(torch.rand(self.d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        self.dt_proj.bias._no_reinit = True

        # S4D real initialization
        A = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=self.d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True

        # D "skip" parameter
        self.D = nn.Parameter(torch.ones(self.d_inner, device=device))  # Keep in fp32
        self.D._no_weight_decay = True
        # bidirectional
        A_b = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=self.d_inner,
        ).contiguous()
        A_b_log = torch.log(A_b)  # Keep A_b_log in fp32
        self.A_b_log = nn.Parameter(A_b_log)
        self.A_b_log._no_weight_decay = True

        A_c = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=self.d_inner,
        ).contiguous()
        A_c_log = torch.log(A_c)  # Keep A_b_log in fp32
        self.A_c_log = nn.Parameter(A_c_log)
        self.A_c_log._no_weight_decay = True


        A_d = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=self.d_inner,
        ).contiguous()
        A_d_log = torch.log(A_d)  # Keep A_b_log in fp32
        self.A_d_log = nn.Parameter(A_d_log)
        self.A_d_log._no_weight_decay = True

        self.conv1d_b = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
            **factory_kwargs,
        )
        self.conv1d_c = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
            **factory_kwargs,
        )
        self.x_proj_b = nn.Linear(
            self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
        )
        self.x_proj_c = nn.Linear(
            self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
        )
        self.dt_proj_b = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)
        self.dt_proj_c = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)

        self.D_b = nn.Parameter(torch.ones(self.d_inner, device=device))  # Keep in fp32
        self.D_b._no_weight_decay = True

        self.D_c = nn.Parameter(torch.ones(self.d_inner, device=device))  # Keep in fp32
        self.D_c._no_weight_decay = True

        self.D_d = nn.Parameter(torch.ones(self.d_inner, device=device))  # Keep in fp32
        self.D_d._no_weight_decay = True

        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)

        self.sa = SpatialAttentionModule().to(device)
        self.ca = ChannelAttentionModule(self.Cin).to(device)
    def forward(self, hidden_states, inference_params=None, extra_emb1=None, extra_emb2=None):
        """
        hidden_states: (B, L, D)
        Returns: same shape as hidden_states
        """
        b, c, h, w = hidden_states.shape
        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)
        # In the backward pass we write dx and dz next to each other to avoid torch.cat

        if self.bimamba_type == "v2":

            # x = self.conv2d_1(hidden_states)
            xa = self.ca(hidden_states)
            # print(x.shape, xa.shape)
            # xz = torch.cat([x, xa], 1)
            patch_size = 2
            N = round(h / patch_size)

            x = hidden_states.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
            x = x.contiguous().view(b, c, N*N, patch_size, patch_size).permute(0, 2, 3, 4, 1).reshape(b*N*N, patch_size*patch_size, c)
            # x = hidden_states.reshape(b, c, N, patch_size, N, patch_size).permute(0, 2, 4, 3, 5, 1).reshape(b*N*N, patch_size * patch_size, c).contiguous()

            xa_f = xa.repeat(1, h * w, 1).reshape(b*N*N, patch_size * patch_size, c)
            xz = torch.cat([x, xa_f], 1)
            # print('xz.shape:', xz.shape)
            A_b = -torch.exp(self.A_b_log.float())

            out = mamba_inner_fn_no_out_proj(
                xz,
                self.conv1d.weight,
                self.conv1d.bias,
                self.x_proj.weight,
                self.dt_proj.weight,
                A,
                None,  # input-dependent B
                None,  # input-dependent C
                self.D.float(),
                delta_bias=self.dt_proj.bias.float(),
                delta_softplus=True,
            )
            out_b = mamba_inner_fn_no_out_proj(
                xz.flip([-1]),
                self.conv1d_b.weight,
                self.conv1d_b.bias,
                self.x_proj_b.weight,
                self.dt_proj_b.weight,
                A_b,
                None,
                None,
                self.D_b.float(),
                delta_bias=self.dt_proj_b.bias.float(),
                delta_softplus=True,
            )
            # F.linear(rearrange(out_z, "b d l -> b l d"), out_proj_weight, out_proj_bias)
            # aaaa = rearrange(out + out_b.flip([-1]), "b d l -> b l d")
            # bbbb = self.out_proj.weight
            if not self.if_devide_out:
                out = F.linear(rearrange(out + out_b.flip([-1]), "b d l -> b l d"), self.out_proj.weight,
                               self.out_proj.bias)

            else:
                out = F.linear(rearrange(out + out_b.flip([-1]), "b d l -> b l d") / 2, self.out_proj.weight,
                               self.out_proj.bias)
            out = out.reshape(b, N, N, c, patch_size, patch_size).permute(0, 3, 1, 4, 2, 5).reshape(b, c, h, w)


        elif self.bimamba_type == "v3":

            # x = self.conv2d_1(hidden_states)
            xa = self.sa(hidden_states)
            xz = torch.cat([hidden_states, xa.repeat(1, c, 1, 1)], 1)
            # print('1', xz.shape)
            xz = xz.flatten(2, 3)
            # print('2', xz.shape)
            A_b = -torch.exp(self.A_b_log.float())
            A_c = -torch.exp(self.A_c_log.float())
            A_d = -torch.exp(self.A_d_log.float())
            out = mamba_inner_fn_no_out_proj(
                xz,
                self.conv1d.weight,
                self.conv1d.bias,
                self.x_proj.weight,
                self.dt_proj.weight,
                A,
                None,  # input-dependent B
                None,  # input-dependent C
                self.D.float(),
                delta_bias=self.dt_proj.bias.float(),
                delta_softplus=True,
            )
            # print('xz.shape:', xz.shape)
            out_b = mamba_inner_fn_no_out_proj(
                xz.flip([-1]),
                self.conv1d_b.weight,
                self.conv1d_b.bias,
                self.x_proj_b.weight,
                self.dt_proj_b.weight,
                A_b,
                None,
                None,
                self.D_b.float(),
                delta_bias=self.dt_proj_b.bias.float(),
                delta_softplus=True,
            )
            # print('out_b.shape:', out_b.shape)
            b, c, l = xz.shape
            h = round(math.sqrt(l))
            xzb = xz.view(b, c, h, h).transpose(2, 3).flatten(2, 3)
            out_c = mamba_inner_fn_no_out_proj(
                xzb,
                self.conv1d_b.weight,
                self.conv1d_b.bias,
                self.x_proj_b.weight,
                self.dt_proj_b.weight,
                A_c,
                None,
                None,
                self.D_c.float(),
                delta_bias=self.dt_proj_b.bias.float(),
                delta_softplus=True,
            )
            # print('out_c.shape:', out_c.shape)
            out_d = mamba_inner_fn_no_out_proj(
                xzb.flip([-1]),
                self.conv1d_b.weight,
                self.conv1d_b.bias,
                self.x_proj_b.weight,
                self.dt_proj_b.weight,
                A_d,
                None,
                None,
                self.D_d.float(),
                delta_bias=self.dt_proj_b.bias.float(),
                delta_softplus=True,
            )

            # F.linear(rearrange(out_z, "b d l -> b l d"), out_proj_weight, out_proj_bias)
            if not self.if_devide_out:
                # print('out_c.shape:', out_c.shape)
                out_c = out_c.view(b, round(c/2), h, h).transpose(2, 3).flatten(2, 3)
                out_d = out_d.view(b, round(c/2), h, h).transpose(2, 3).flatten(2, 3).flip([-1])
                out = F.linear(rearrange(out + out_b.flip([-1]) + out_c + out_d, "b d l -> b l d"), self.out_proj.weight,
                               self.out_proj.bias)
                out = out.permute(0, 2, 1).reshape(b, self.Cout, h, w)
            else:
                our_c = out_c.view(b, round(c/2), h, h).transpose(2, 3).flatten(2, 3)
                out_d = out_d.view(b, round(c/2), h, h).transpose(2, 3).flatten(2, 3).flip([-1])
                out = F.linear(rearrange(out + out_b.flip([-1]) + our_c + out_d, "b d l -> b l d") / 2, self.out_proj.weight,
                               self.out_proj.bias)
                out = out.permute(0, 2, 1).reshape(b, self.Cout, h, w)
        if self.init_layer_scale is not None:
            out = out * self.gamma
        return out, xa

# x = torch.randn(2,31,40,40)
# B, C, H, W = x.shape
# xs = x.new_empty((B, 4, C, H * W))
# # 添加横向和竖向的扫描
# xs[:, 0] = x.flatten(2, 3)
# xs[:, 1] = x.transpose(dim0=2, dim1=3).flatten(2, 3)
# xs[:, 2:4] = torch.flip(xs[:, 0:2], dims=[-1])
# print(xs.shape)
if __name__ == "__main__":
    # 模型测试
    # device = torch.device('cuda:0')
    # x = torch.randn(2, 31, 80, 80).to(device)
    # net = Mamba(31, bimamba_type='v3').to(device)
    # b, xa = net(x)
    # print('b.shape:', b.shape, xa.shape)

    class ChannelAttention(nn.Module):
        """
        CBAM混合注意力机制的通道注意力
        """

        def __init__(self, in_channels, ratio=16):
            super(ChannelAttention, self).__init__()
            self.avg_pool = nn.AdaptiveAvgPool2d(1)
            self.max_pool = nn.AdaptiveMaxPool2d(1)

            self.fc = nn.Sequential(
                # 全连接层
                # nn.Linear(in_planes, in_planes // ratio, bias=False),
                # nn.ReLU(),
                # nn.Linear(in_planes // ratio, in_planes, bias=False)

                # 利用1x1卷积代替全连接，避免输入必须尺度固定的问题，并减小计算量
                nn.Conv2d(in_channels, in_channels // ratio, 1, bias=False),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels // ratio, in_channels, 1, bias=False)
            )

            self.sigmoid = nn.Sigmoid()


        def forward(self, x):
            avg_out = self.fc(self.avg_pool(x))
            max_out = self.fc(self.max_pool(x))
            out = avg_out + max_out
            out = self.sigmoid(out)
            print(out.shape)
            return out * x

    a = torch.randn(2, 31, 40, 40)
    net = ChannelAttention(31)
    b = net(a)
    print(b.shape)


#转置测试

from PIL import Image
# import torch
import numpy as np
import matplotlib.pyplot as plt
# 读取PNG文件
# image_path = '/media/xd132/USER/ZTZ/wy_fuse/base/test/hh_2-10_json_11.png'
# image = Image.open(image_path)
# image = torch.from_numpy(np.array(image))
# data = image
#
# # 显示拼接后的图像
# plt.figure(figsize=(6, 6))
# plt.imshow(data.numpy())
# plt.axis('off')
# plt.title('Restored Image')
# plt.show()
#
# image = image.permute(2, 0, 1).reshape(1, 3, 512, 512)
#
# patch_size = 256
# num_patches = 2
#
# patches = image.unfold(2, patch_size, patch_size)
# patches = patches.unfold(3, patch_size, patch_size)
#
# patches = patches.contiguous().view(3, 4, patch_size, patch_size)
#
# patches = patches.contiguous().permute(1, 2, 3, 0).numpy()
#
# # patches_np = patches.squeeze().permute(0, 2, 1, 3, 4).reshape(num_patches * patch_size, num_patches * patch_size, 3).numpy()
# # patches_np = patches_np.transpose(2, 0, 1)
#
# # 显示裁剪后的四个子图像
# fig, axes = plt.subplots(1, 4, figsize=(12, 3))
# for i in range(4):
#     axes[i].imshow(patches[i, :, :, :].reshape(256,256,3))
#     axes[i].axis('off')
#     axes[i].set_title(f'Patch {i+1}')
#
# plt.tight_layout()
# plt.show()
#
# # 将四个子图像拼接回原始大小
# image_restored = patches.reshape(2, 2, 256, 256, 3).transpose(0, 2, 1, 3, 4).reshape(512,512,3)
# image_restored_np = image_restored
# # 将拼接后的图像转换为 numpy 数组，便于显示
# # image_restored_np = image_restored.squeeze().permute(1, 2, 0).numpy()
#
# # 显示拼接后的图像
# plt.figure(figsize=(6, 6))
# plt.imshow(image_restored_np)
# plt.axis('off')
# plt.title('Restored Image')
# plt.show()

# device = torch.device('cuda:0')
# x = torch.randn(2, 31, 80, 80).to(device)
# b, c, h, w = x.shape
# patch_size = 20
# N = round(h / patch_size)
# y = x.reshape(b, c, N, patch_size, N, patch_size).permute(0, 2, 4, 3, 5, 1).reshape(b * N * N, patch_size * patch_size, c).contiguous()
# z = y.reshape(b, N, N, c, patch_size, patch_size).permute(0, 3, 1, 4, 2, 5).reshape(b, c, h, w).contiguous()
#
# y_reshaped = y.reshape(b, N, N, patch_size * patch_size, c)
#
# # 然后进行置换，以匹配原始的 x 维度顺序，但跳过 patch_size 的分割（因为它在之前被合并了）
# y_permuted = y_reshaped.permute(0, 2, 4, 1, 3)  # 注意这里的索引变化
#
# # 最后，将 patch_size 分割回原始维度
# x_reconstructed = y_permuted.reshape(b, c, N * patch_size, N * patch_size)
#
# c = z
#
