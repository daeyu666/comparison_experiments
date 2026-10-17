# MS_MSA 光谱维度MSA
# 2024-05-07
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from comparison.DRTNet.upstream.models.EMA import ema


class MSAB(nn.Module):
    def __init__(self, dim, dim_head, heads, dataset):
        super().__init__()
        # dataset = dataset
        self.blocks = nn.ModuleList([])
        self.blocks.append(nn.ModuleList([
            MS_MSA(dim=dim, dim_head=dim_head, heads=heads, dataset=dataset),
            PreNorm(dim_head, FeedForward(dim=dim_head))  # dim
        ]))

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)  # [1, 8, 8, 618]
        # print('x', x.shape)
        for (attn, ff) in self.blocks:
            x = attn(x)  # + x  # [1, 8, 8, 1024]
            # print('x',x.shape)
            x = ff(x)  # + x
        out = x.permute(0, 3, 1, 2)  # [1, 618, 8, 8]
        # print('out',out.shape)
        return out

# class MSAB_TSTfirst(nn.Module):
#     def __init__(self, dim, dim_head, heads):
#         super().__init__()
#         # dataset = dataset
#         self.blocks = nn.ModuleList([])
#         self.blocks.append(nn.ModuleList([
#             MS_MSA_TSTfirst(dim=dim, dim_head=dim_head, heads=heads),
#             PreNorm(dim, FeedForward(dim=dim))
#         ]))
#
#     def forward(self, x):
#         # print('z',x['z'].shape)
#         x = x.permute(0, 2, 3, 1)  # [1, 32, 32, 382]
#         # print('x', x.shape)
#         for (attn, ff) in self.blocks:
#             x = attn(x)  # + x  # [1, 8, 8, 1024]
#             # print('x',x.shape)
#             x = ff(x)  # + x
#         out = x.permute(0, 3, 1, 2)  # [1, 618, 8, 8]
#         # print('out',out.shape)
#         return out
class MS_MSA(nn.Module):
    def __init__(
            self,
            dim,
            dim_head,
            heads,
            dataset = None
    ):
        super().__init__()
        self.num_heads = heads
        self.dim_head = dim_head
        self.to_q = nn.Linear(dim, dim_head * heads, bias=False)
        self.to_k = nn.Linear(dim, dim_head * heads, bias=False)
        self.to_v = nn.Linear(dim, dim_head * heads, bias=False)
        self.rescale = nn.Parameter(torch.ones(heads, 1, 1))
        self.proj = nn.Linear(dim_head * heads, dim_head, bias=True)
        self.pos_emb = nn.Sequential(
            nn.Conv2d(dim_head, dim_head, 3, 1, 1, bias=False, groups=dim_head),
            GELU(),
            nn.Conv2d(dim_head, dim_head, 3, 1, 1, bias=False, groups=dim_head),
        )
        # print('dim_head',dim_head) # dim_head 1024
        self.dim = dim
        # if dataset == 'PaviaU':
        #     self.ema = ema(618)
        #     self.conv = nn.Conv2d(618, dim_head, 3, 1, 1, bias=False)
        # elif dataset == 'Botswana':
        #     self.ema = ema(870)
        #     self.conv = nn.Conv2d(870, dim_head, 3, 1, 1, bias=False)
        # elif dataset == 'Urban':
        self.ema = ema(dim)
        self.conv = nn.Conv2d(dim, dim_head, 3, 1, 1, bias=False)

    def forward(self, x_in):
        """
        x_in: [b,h,w,c]
        return out: [b,h,w,c]
        """
        b, h, w, c = x_in.shape
        x = x_in.reshape(b, h * w, c)
        q_inp = self.to_q(x)
        k_inp = self.to_k(x)
        v_inp = self.to_v(x)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.num_heads),
                      (q_inp, k_inp, v_inp))
        v = v
        # q: b,heads,hw,c
        q = q.transpose(-2, -1)
        k = k.transpose(-2, -1)
        v = v.transpose(-2, -1)
        q = F.normalize(q, dim=-1, p=2)
        k = F.normalize(k, dim=-1, p=2)
        attn = (k @ q.transpose(-2, -1))  # A = K^T*Q
        attn = attn * self.rescale
        attn = attn.softmax(dim=-1)
        x = attn @ v  # b,heads,d,hw
        x = x.permute(0, 3, 1, 2)  # Transpose
        x = x.reshape(b, h * w, self.num_heads * self.dim_head)
        # print('x',x.shape) # [1, 64, 1024]
        # x= x.view(b, 8,8, -1) # view操作
        # print('x', x.shape)
        out_c = self.proj(x).view(b,  h, w, -1)  # view(b, h, w, c)  # [1, 8, 8, 618]  需要改成[1, 8, 8, 1024]？
        # print('outc',out_c.shape) # [1, 8, 8, 1024]
        # print('v_inp',v_inp.shape) # [1, 64, 1024]

        # ######## 0618EMA ##############
        # out_c = out_c.squeeze(0) # .view(1, 8, 8, 1024)
        # print('inx', x_in.shape) # [1, 8, 8, 618]  inx torch.Size([1, 32, 32, 256])
        x_in = x_in.permute(0, 3, 1, 2)
        x_in = self.ema(x_in)
        x_in = self.conv(x_in)
        x_in = x_in.permute(0, 2, 3, 1)  # outx torch.Size([1, 32, 32, 1024])
        # print('v_inp', v_inp.shape)
        # print('inx', x_in.shape)
        #############################
        out_p = self.pos_emb(v_inp.view(b, h, w, -1).permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        # print('out_p', out_p.shape)
        out = out_c + out_p + x_in
        # print('out', out.shape) # [1, 8, 8, 1024]
        return out

class MS_MSA_TSTfirst(nn.Module):
    def __init__(
            self,
            dim,
            dim_head,
            heads,

    ):
        super().__init__()
        self.num_heads = heads
        self.dim_head = dim_head
        self.to_q = nn.Linear(dim, dim_head * heads, bias=False)
        self.to_k = nn.Linear(dim, dim_head * heads, bias=False)
        self.to_v = nn.Linear(dim, dim_head * heads, bias=False)
        self.rescale = nn.Parameter(torch.ones(heads, 1, 1))
        self.proj = nn.Linear(dim_head * heads, 1024, bias=True)
        self.pos_emb = nn.Sequential(
            nn.Conv2d(dim_head, dim_head, 3, 1, 1, bias=False, groups=dim_head),
            GELU(),
            nn.Conv2d(dim_head, dim_head, 3, 1, 1, bias=False, groups=dim_head),
        )
        # print('dim_head',dim_head) # dim_head 1024
        self.dim = dim
        # if dataset == 'PaviaU':
        #     self.ema = ema(618)
        #     self.conv = nn.Conv2d(618, dim_head, 3, 1, 1, bias=False)
        # elif dataset == 'Botswana':
        #     self.ema = ema(870)
        #     self.conv = nn.Conv2d(870, dim_head, 3, 1, 1, bias=False)
        # elif dataset == 'Urban':
        self.ema = ema(dim)
        self.conv = nn.Conv2d(dim, dim_head, 3, 1, 1, bias=False)

    def forward(self, x_in):
        """
        x_in: [b,h,w,c]
        return out: [b,h,w,c]
        """
        b, h, w, c = x_in.shape
        x = x_in.reshape(b, h * w, c)
        q_inp = self.to_q(x)
        k_inp = self.to_k(x)
        v_inp = self.to_v(x)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.num_heads),
                      (q_inp, k_inp, v_inp))
        v = v
        # q: b,heads,hw,c
        q = q.transpose(-2, -1)
        k = k.transpose(-2, -1)
        v = v.transpose(-2, -1)
        q = F.normalize(q, dim=-1, p=2)
        k = F.normalize(k, dim=-1, p=2)
        attn = (k @ q.transpose(-2, -1))  # A = K^T*Q
        attn = attn * self.rescale
        attn = attn.softmax(dim=-1)
        x = attn @ v  # b,heads,d,hw
        x = x.permute(0, 3, 1, 2)  # Transpose
        x = x.reshape(b, h * w, self.num_heads * self.dim_head)
        # print('x',x.shape) # [1, 64, 1024]
        # x= x.view(b, 8,8, -1) # view操作
        # print('x', x.shape)
        out_c = self.proj(x).view(b, 8, 8, -1)  # view(b, h, w, c)  # [1, 8, 8, 618]  需要改成[1, 8, 8, 1024]？
        # print('outc',out_c.shape) # [1, 8, 8, 1024]
        # print('v_inp',v_inp.shape) # [1, 64, 1024]

        # ######## 0618EMA ##############
        # out_c = out_c.squeeze(0) # .view(1, 8, 8, 1024)
        # print('inx', x_in.shape) # [1, 8, 8, 618]
        x_in = x_in.permute(0, 3, 1, 2)
        x_in = self.ema(x_in)
        x_in = self.conv(x_in)
        x_in = x_in.permute(0, 2, 3, 1)
        # print('outx', x_in.shape)
        #############################
        out_p = self.pos_emb(v_inp.view(b, 8, 8, -1).permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        out = out_c + out_p + x_in
        # print('out', out.shape) # [1, 8, 8, 1024]
        return out

class FeedForward(nn.Module):
    def __init__(self, dim, mult=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim, dim * mult, 1, 1, bias=False),
            GELU(),
            nn.Conv2d(dim * mult, dim * mult, 3, 1, 1, bias=False, groups=dim * mult),
            GELU(),
            nn.Conv2d(dim * mult, dim, 1, 1, bias=False),
        )

    def forward(self, x):
        """
        x: [b,h,w,c]
        return out: [b,h,w,c]
        """
        # print('FFx',x.permute(0, 3, 1, 2).shape)
        out = self.net(x.permute(0, 3, 1, 2))
        return out.permute(0, 2, 3, 1)


class GELU(nn.Module):
    def forward(self, x):
        return F.gelu(x)


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, *args, **kwargs):
        x = self.norm(x)
        return self.fn(x, *args, **kwargs)
