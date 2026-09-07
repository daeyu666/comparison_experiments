import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from timm.models.layers import DropPath, to_2tuple
device = torch.device('cuda:1')
# from test_network import SingleMambaBlock
import torch_dct as dct
import numbers
from mamba_simple_xr3.modules.mamba_simple_4scan_xiugai import Mamba
from einops import rearrange
from torch_dct import dct_2d, idct_2d


def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')

def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        if len(x.shape)==4:
            h, w = x.shape[-2:]
            return to_4d(self.body(to_3d(x)), h, w)
        else:
            return self.body(x)
class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma+1e-5) * self.weight + self.bias


class SingleMambaBlock(nn.Module):
    def __init__(self, dim):
        super(SingleMambaBlock, self).__init__()
        self.encoder = Mamba(dim)
        self.norm = LayerNorm(dim, 'with_bias')
        # self.PatchEmbe=PatchEmbed(patch_size=4, stride=4,in_chans=dim, embed_dim=dim*16)
    def forward(self,ipt):
        x, residual = ipt
        residual = x+residual
        x = self.norm(residual)
        return (self.encoder(x),residual)


class PatchEmbed(nn.Module):
    r""" Image to Patch Embedding

    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)  # B Ph*Pw C
        if self.norm is not None:
            x = self.norm(x)
        return x

    def flops(self):
        flops = 0
        H, W = self.img_size
        if self.norm is not None:
            flops += H * W * self.embed_dim
        return flops


class PatchUnEmbed(nn.Module):

    def __init__(self, img_size=224, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()

    def forward(self, x, x_size):

        B, HW, C = x.shape
        x = x.transpose(1, 2).view(B, C, x_size, x_size)  # B Ph*Pw C

        return x


class BasicConv2d(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1, padding=1, use_relu=True):
        super(BasicConv2d, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, out_channel,
                               kernel_size=kernel_size, stride=stride,
                               padding=padding, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channel, out_channel,
                               kernel_size=kernel_size, stride=stride,
                               padding=padding, bias=False)
        self.use_relu = use_relu

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        return x


class Registration(nn.Module):
    def __init__(self, in_channel, kernel_size=3, stride=1, padding=1):
        super(Registration, self).__init__()
        self.convh = nn.Conv2d(in_channel, 1, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.conv1 = nn.Conv2d(in_channel, in_channel, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channel, in_channel, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.conv3 = nn.Sequential(nn.Conv2d(kernel_size*kernel_size, 4, 3, 1, 1),
                                   nn.ReLU(),
                                   nn.Conv2d(4, 2, 3, 1, 1))
        self.kernel_size = kernel_size
        self.sigmoid = nn.Sigmoid()

        self.RG = SpatialTransformation()

    def forward(self, LRHS, HRMS, LRHS_):
        threshold = 0.1
        LRHS_h = dct_2d(LRHS, norm='ortho')
        LRHS_h = LRHS * (torch.abs(LRHS_h) < threshold)
        LRHS_h = idct_2d(LRHS_h, norm='ortho')
        LRHS_h = 0.01 * self.sigmoid(self.convh(LRHS_h))
        #
        LRHS = LRHS + LRHS_h*LRHS
        LRHS = self.relu(self.conv1(LRHS))
        HRMS = self.relu(self.conv2(HRMS))
        B, C, H, _ = LRHS.shape
        padding = (self.kernel_size - 1) // 2  # 为了确保窗口可以覆盖边界像素
        # 对HRMS进行填充以处理边界情况
        padded_HRMS = F.pad(HRMS, (padding, padding, padding, padding), mode='constant', value=0)
        # 使用unfold来展开局部窗口
        unfolded_HRMS = padded_HRMS.unfold(2, self.kernel_size, 1).unfold(3, self.kernel_size, 1)
        unfolded_HRMS = unfolded_HRMS.reshape(B, C, H * H, self.kernel_size * self.kernel_size).permute(0, 2, 3, 1)
        LRHS = LRHS.reshape(B, C, H * H, 1).permute(0, 2, 1, 3)
        out = torch.matmul(unfolded_HRMS, LRHS).reshape(B, H, H, self.kernel_size * self.kernel_size).permute(0, 3, 1, 2)
        out = self.conv3(out).clamp(min=-2, max=2)

        out = out.permute(0, 2, 3, 1)
        LRHS_ = LRHS_.permute(0, 2, 3, 1)
        c = self.RG(LRHS_, out).permute(0, 3, 1, 2)
        # print("out.shape:::::", out.shape)
        return c, out


class FUSE_Net(nn.Module):
    def __init__(self, Cin, Cout, hw):
        super(FUSE_Net, self).__init__()

        self.convp_h = nn.Conv2d(Cout, Cin, 3, 1, 1)
        # self.spatialmamba = SingleMambaBlock(Cin)
        # self.spectramamba = SingleMambaBlock(hw)
        self.spatialmamba = Mamba(Cout, bimamba_type='v3')
        self.spectramamba = Mamba(hw, bimamba_type='v2')
        # self.patchemb = PatchEmbed()
        # self.unpatchemb = PatchUnEmbed()
        # self.conv_fuse = nn.Conv2d(Cin + Cout, Cin, stride=1, padding=1, kernel_size=3)

        self.conv_fuse = nn.Sequential(nn.Conv2d(Cin + Cout, 2 * (Cout + Cin), stride=1, padding=1, kernel_size=3),
                                        nn.ReLU(inplace=True),
                                        nn.Conv2d(2 * (Cin + Cout), 4 * Cin, stride=1, padding=1, kernel_size=3),
                                        nn.ReLU(inplace=True),
                                        nn.Conv2d(4 * Cin, 4 * Cin, stride=1, padding=1, kernel_size=3),
                                        nn.ReLU(inplace=True),
                                        nn.Conv2d(4 * Cin, 2 * Cin, stride=1, padding=1, kernel_size=3),
                                        nn.ReLU(inplace=True),
                                        nn.Conv2d(2 * Cin, Cin, stride=1, padding=1, kernel_size=3
                                       ))
    def forward(self, LRHS, HRMS):
        # HRMS = self.convp_h(HRMS)
        b, c, _, _ = LRHS.shape
        LRHS, ca = self.spectramamba(LRHS)
        LRHS = torch.nn.functional.interpolate(LRHS, size=None, scale_factor=2, mode='nearest', align_corners=None)
        HRMS, sa = self.spatialmamba(HRMS)
        LRHS = LRHS * ca.reshape(b, c, 1, 1) + LRHS
        HRMS = HRMS * sa + HRMS
        FUSE = torch.cat([LRHS, HRMS], 1)
        FUSE = self.conv_fuse(FUSE)
        return FUSE


class R(nn.Module):
    def __init__(self, cin, cout):
        super(R, self).__init__()

        self.conv1 = nn.Conv2d(cin, cout * 4, 3, 1, 1)
        self.conv2 = nn.Conv2d(cout * 4, cout, 3, 1, 1)
        self.relu = nn.ReLU()

    def forward(self, LRHS):

        LRHS = self.conv2(self.relu(self.conv1(LRHS)))

        return LRHS


class B(nn.Module):
    def __init__(self, sin):
        super(B, self).__init__()

        self.conv1 = nn.Conv2d(sin, sin, 3, 1, 1)
        self.conv2 = nn.Conv2d(sin, sin, 3, 2, 1)
        self.relu = nn.ReLU()

    def forward(self, HRMS):

        HRMS = self.conv1(self.relu(self.conv2(HRMS)))

        return HRMS

"""
    重采样模块
"""
class SpatialTransformation(nn.Module):
    def __init__(self, use_gpu=True):
        self.use_gpu = use_gpu
        super(SpatialTransformation, self).__init__()

    def meshgrid(self, height, width):
        x_t = torch.matmul(torch.ones([height, 1]), torch.transpose(torch.unsqueeze(torch.linspace(0.0, width -1.0, width), 1), 1, 0))
        y_t = torch.matmul(torch.unsqueeze(torch.linspace(0.0, height - 1.0, height), 1), torch.ones([1, width]))

        x_t = x_t.expand([height, width])
        ''
        y_t = y_t.expand([height, width])
        if self.use_gpu==True:
            x_t = x_t.to(device)
            y_t = y_t.to(device)

        return x_t, y_t

    def repeat(self, x, n_repeats):
        rep = torch.transpose(torch.unsqueeze(torch.ones(n_repeats), 1), 1, 0)
        rep = rep.long()
        x = torch.matmul(torch.reshape(x, (-1, 1)), rep)
        if self.use_gpu:
            x = x.to(device)
        return torch.squeeze(torch.reshape(x, (-1, 1)))


    def interpolate(self, im, x, y):

        im = F.pad(im, (0,0,1,1,1,1,0,0))

        batch_size, height, width, channels = im.shape

        batch_size, out_height, out_width = x.shape

        x = x.reshape(1, -1)
        y = y.reshape(1, -1)

        x = x + 1
        y = y + 1

        max_x = width - 1
        max_y = height - 1

        x0 = torch.floor(x).long()
        x1 = x0 + 1
        y0 = torch.floor(y).long()
        y1 = y0 + 1

        x0 = torch.clamp(x0, 0, max_x)
        x1 = torch.clamp(x1, 0, max_x)
        y0 = torch.clamp(y0, 0, max_y)
        y1 = torch.clamp(y1, 0, max_y)

        dim2 = width
        dim1 = width*height
        base = self.repeat(torch.arange(0, batch_size)*dim1, out_height*out_width)

        base_y0 = base + y0*dim2
        base_y1 = base + y1*dim2

        idx_a = base_y0 + x0
        idx_b = base_y1 + x0
        idx_c = base_y0 + x1
        idx_d = base_y1 + x1

        # use indices to lookup pixels in the flat image and restore
        # channels dim
        im_flat = torch.reshape(im, [-1, channels])
        im_flat = im_flat.float()
        dim, _ = idx_a.transpose(1, 0).shape
        Ia = torch.gather(im_flat, 0, idx_a.transpose(1, 0).expand(dim, channels))
        Ib = torch.gather(im_flat, 0, idx_b.transpose(1, 0).expand(dim, channels))
        Ic = torch.gather(im_flat, 0, idx_c.transpose(1, 0).expand(dim, channels))
        Id = torch.gather(im_flat, 0, idx_d.transpose(1, 0).expand(dim, channels))

        # and finally calculate interpolated values
        x1_f = x1.float()
        y1_f = y1.float()

        dx = x1_f - x
        dy = y1_f - y

        wa = (dx * dy).transpose(1,0)
        wb = (dx * (1-dy)).transpose(1, 0)
        wc = ((1-dx) * dy).transpose(1, 0)
        wd = ((1-dx) * (1-dy)).transpose(1, 0)

        output = torch.sum(torch.squeeze(torch.stack([wa*Ia, wb*Ib, wc*Ic, wd*Id], dim=1)), 1)
        output = torch.reshape(output, [-1, out_height, out_width, channels])
        return output

    def forward(self, moving_image, deformation_matrix):
        dx = deformation_matrix[:, :, :, 0]
        dy = deformation_matrix[:, :, :, 1]

        batch_size, height, width = dx.shape

        x_mesh, y_mesh = self.meshgrid(height, width)

        x_mesh = x_mesh.expand([batch_size, height, width])
        y_mesh = y_mesh.expand([batch_size, height, width])
        x_new = dx + x_mesh
        y_new = dy + y_mesh

        return self.interpolate(moving_image, x_new, y_new)


class Net(nn.Module):
    def __init__(self, cin, cout):
        super(Net, self).__init__()

        self.R = R(cin, cout)
        self.BP = B(cout)
        self.BH = B(cin)
        self.RG = Registration(cout)
        self.FUSE_2 = FUSE_Net(cin, cout, 2*2)
        self.FUSE_4 = FUSE_Net(cin, cout, 2*2)
        self.fuse_last = nn.Sequential(nn.Conv2d(cin, 2 * cin, stride=1, padding=1, kernel_size=3),
                                        nn.ReLU(inplace=True),
                                        nn.Conv2d(2 * cin, 2 * cin, stride=1, padding=1, kernel_size=3),
                                        nn.ReLU(inplace=True),
                                        nn.Conv2d(2 * cin, 2 * cin, stride=1, padding=1, kernel_size=3),
                                        nn.ReLU(inplace=True),
                                        nn.Conv2d(2 * cin, cin, stride=1, padding=1, kernel_size=3
                                       ))
    def forward(self, LRHS, HRMS):

        LRHS_LRMS = self.R(LRHS)
        HRMS_LRMS_2 = self.BP(HRMS)
        HRMS_LRMS_1 = self.BP(HRMS_LRMS_2)

        # LRHS_LRMS_RG_1, RG_1 = self.RG(LRHS_LRMS, HRMS_LRMS_1, LRHS)
        # print("LRHS_LRMS_RG_1:", LRHS_LRMS_RG_1.shape)

        LRHS_FUSE_2 = self.FUSE_2(LRHS_LRMS, HRMS_LRMS_2)
        # LRHS_LRMS_RG_2, RG_2 = self.RG(self.R(LRHS_FUSE_2), HRMS_LRMS_2, LRHS_FUSE_2)

        HRHS = self.FUSE_4(LRHS_FUSE_2, HRMS)
        HRHS = self.fuse_last(HRHS)

        HRHS_HRMS = self.R(HRHS)
        HRHS_LRHS_2 = self.BP(HRHS_HRMS)

        return HRHS, HRHS_HRMS, HRHS_LRHS_2, LRHS_FUSE_2


if __name__ == "__main__":

    a = torch.randn(2, 102, 40, 40).to(device)
    b = torch.randn(2, 4, 160, 160).to(device)
    net = Net(102, 4).to(device)
    HRHS, HRHS_HRMS, HRHS_LRHS_2, LRHS_LRMS_RG_2, RG_1, RG_2 = net(a, b)
    print(HRHS.shape)

