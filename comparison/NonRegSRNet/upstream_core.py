"""Active released NonRegSRNet blocks (author: Ke Zheng).
Source: saber-zero/NonRegSRNet, commit 84e236db23695120ce0eb2a2f25b78d7fde70c21,
DeformatedHyperImageFusion/model/network.py (blob 6f23cef4a6371b6d99d58087b2df4e58097a6d73).
Only active blocks are included. CUDA allocations follow input device; differentiable
activation clamps are out-of-place; zero SRF projection has a uniform fallback.
"""
import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F
import numpy as np


def init_weights(net, init_type="normal", gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, "weight") and (classname.find("Conv") != -1 or classname.find("Linear") != -1):
            if init_type == "normal":
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == "xavier":
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == "kaiming":
                init.kaiming_normal_(m.weight.data, a=0, mode="fan_in")
            elif init_type == "orthogonal":
                init.orthogonal_(m.weight.data, gain=gain)
            elif init_type == "mean_space":
                batchsize, channel, height, weight = list(m.weight.data.size())
                m.weight.data.fill_(1 / (height * weight))
            elif init_type == "mean_channel":
                batchsize, channel, height, weight = list(m.weight.data.size())
                m.weight.data.fill_(1 / (channel))
            else:
                raise NotImplementedError("initialization method [%s] is not implemented" % init_type)
            if hasattr(m, "bias") and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find("BatchNorm2d") != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    net.apply(init_func)

class SumToOneLoss(nn.Module):
    def __init__(self):
        super(SumToOneLoss, self).__init__()
        self.register_buffer("one", torch.tensor(1, dtype=torch.float))

        self.loss = nn.L1Loss(reduction="sum")

    def get_target_tensor(self, input):
        target_tensor = self.one
        return target_tensor.expand_as(input)

    def __call__(self, input):
        input = torch.sum(input, 1)
        target_tensor = self.get_target_tensor(input)
        # print(input[0,:,:])
        loss = self.loss(input, target_tensor)
        # loss = torch.sum(torch.abs(target_tensor - input))
        return loss

def cross_correlation_loss(I, J, n):
    #     I = I.permute(0, 3, 1, 2)
    #     J = J.permute(0, 3, 1, 2)
    batch_size, channels, xdim, ydim = I.shape
    I2 = torch.mul(I, I)
    J2 = torch.mul(J, J)
    IJ = torch.mul(I, J)
    sum_filter = I.new_ones((1, channels, n, n))
    I_sum = torch.conv2d(I, sum_filter, padding=1, stride=(1, 1))
    J_sum = torch.conv2d(J, sum_filter, padding=1, stride=(1, 1))
    I2_sum = torch.conv2d(I2, sum_filter, padding=1, stride=(1, 1))
    J2_sum = torch.conv2d(J2, sum_filter, padding=1, stride=(1, 1))
    IJ_sum = torch.conv2d(IJ, sum_filter, padding=1, stride=(1, 1))
    win_size = n ** 2
    u_I = I_sum / win_size
    u_J = J_sum / win_size
    cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_size
    I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size
    J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size
    cc = cross * cross / (I_var * J_var + np.finfo(float).eps)

    return torch.mean(cc)

def smooothing_loss(y_pred):
    #     dy = torch.abs(y_pred[:, 1:, :, :] - y_pred[:, :-1, :, :])
    #     dx = torch.abs(y_pred[:, :, 1:, :] - y_pred[:, :, :-1, :])
    dy = torch.abs(y_pred[:, :, 1:, :] - y_pred[:, :, :-1, :])
    dx = torch.abs(y_pred[:, :, :, 1:] - y_pred[:, :, :, :-1])

    dx = torch.mul(dx, dx)
    dy = torch.mul(dy, dy)
    d = torch.mean(dx) + torch.mean(dy)
    return d / 2.0

class UNet(nn.Module):
    def contracting_block(self, in_channels, out_channels, kernel_size=3):
        """
        This function creates one contracting block
        """
        block = torch.nn.Sequential(
            torch.nn.Conv2d(kernel_size=kernel_size, in_channels=in_channels, out_channels=out_channels, padding=1,),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(),
            torch.nn.Conv2d(kernel_size=kernel_size, in_channels=out_channels, out_channels=out_channels, padding=1,),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(),
        )
        return block

    def expansive_block(self, in_channels, mid_channel, out_channels, kernel_size=3):
        """
        This function creates one expansive block
        """
        block = torch.nn.Sequential(
            torch.nn.Conv2d(kernel_size=kernel_size, in_channels=in_channels, out_channels=mid_channel, padding=1,),
            torch.nn.BatchNorm2d(mid_channel),
            torch.nn.ReLU(),
            torch.nn.Conv2d(kernel_size=kernel_size, in_channels=mid_channel, out_channels=mid_channel, padding=1,),
            torch.nn.BatchNorm2d(mid_channel),
            torch.nn.ReLU(),
            # torch.nn.ConvTranspose2d(
            #     in_channels=mid_channel,
            #     out_channels=out_channels,
            #     kernel_size=3,
            #     stride=2,
            #     padding=1,
            #     output_padding=1,
            # ),
            torch.nn.Conv2d(kernel_size=kernel_size, in_channels=mid_channel, out_channels=out_channels, padding=1,),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(),
        )
        return block

    def final_block(self, in_channels, mid_channel, out_channels, kernel_size=3):
        """
        This returns final block
        """
        block = torch.nn.Sequential(
            torch.nn.Conv2d(kernel_size=kernel_size, in_channels=in_channels, out_channels=mid_channel, padding=1,),
            torch.nn.BatchNorm2d(mid_channel),
            torch.nn.ReLU(),
            torch.nn.Conv2d(kernel_size=kernel_size, in_channels=mid_channel, out_channels=out_channels, padding=1,),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(),
        )
        return block

    def __init__(self, in_channel, out_channel):
        super(UNet, self).__init__()
        # Encode
        self.conv_encode1 = self.contracting_block(in_channels=in_channel, out_channels=32)
        self.conv_maxpool1 = torch.nn.MaxPool2d(kernel_size=1)
        self.conv_encode2 = self.contracting_block(32, 64)
        self.conv_maxpool2 = torch.nn.MaxPool2d(kernel_size=2)
        self.conv_encode3 = self.contracting_block(64, 128)
        # self.conv_maxpool3 = torch.nn.MaxPool2d(kernel_size=2)
        # Bottleneck
        mid_channel = 128
        self.bottleneck = torch.nn.Sequential(
            torch.nn.Conv2d(kernel_size=3, in_channels=mid_channel, out_channels=mid_channel * 2, padding=1,),
            torch.nn.BatchNorm2d(mid_channel * 2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(kernel_size=3, in_channels=mid_channel * 2, out_channels=mid_channel, padding=1,),
            torch.nn.BatchNorm2d(mid_channel),
            torch.nn.ReLU(),
            # torch.nn.ConvTranspose2d(in_channels=mid_channel, out_channels=mid_channel, kernel_size=3, stride=2, padding=1, output_padding=1),
            # torch.nn.BatchNorm2d(mid_channel),
            # torch.nn.ReLU(),
        )
        # Decode
        self.conv_decode3 = self.expansive_block(256, 128, 64)
        self.conv_decode2 = self.expansive_block(128, 64, 32)
        self.final_layer = self.final_block(64, 32, out_channel)

    def crop_and_concat(self, upsampled, bypass, resize=True):
        """
        This layer resize the layer from contraction block and concat it with expansive block vector
        """
        if resize:
            # c = (bypass.size()[2] - upsampled.size()[2]) // 2
            # bypass = F.pad(bypass, (-c, -c, -c, -c))
            bypass = nn.functional.interpolate(bypass, upsampled.size()[2:], mode="bilinear", align_corners=False)
        return torch.cat((upsampled, bypass), 1)

    def forward(self, x):
        

        # Encode
        encode_block1 = self.conv_encode1(x)
        # encode_pool1 = self.conv_maxpool1(encode_block1)
        encode_block2 = self.conv_encode2(encode_block1)
        # encode_pool2 = self.conv_maxpool2(encode_block2)
        encode_block3 = self.conv_encode3(encode_block2)
        # encode_pool3 = self.conv_maxpool3(encode_block3)
        # Bottleneck
        bottleneck1 = self.bottleneck(encode_block3)
        # Decode
        decode_block3 = self.crop_and_concat(bottleneck1, encode_block3)
        cat_layer2 = self.conv_decode3(decode_block3)
        decode_block2 = self.crop_and_concat(cat_layer2, encode_block2)
        cat_layer1 = self.conv_decode2(decode_block2)
        decode_block1 = self.crop_and_concat(cat_layer1, encode_block1)
        # decode_block1 = nn.functional.interpolate(decode_block1, x.size()[2:], mode="bilinear", align_corners=False)
        final_layer = self.final_layer(decode_block1)

        # import ipdb
        # ipdb.set_trace()
        return final_layer

class SpatialTransformation(nn.Module):
    def __init__(self, use_gpu=False):
        self.use_gpu = use_gpu
        super(SpatialTransformation, self).__init__()

    def meshgrid(self, height, width):
        x_t = torch.matmul(
            torch.ones([height, 1]), torch.transpose(torch.unsqueeze(torch.linspace(0.0, width - 1.0, width), 1), 1, 0),
        )
        y_t = torch.matmul(torch.unsqueeze(torch.linspace(0.0, height - 1.0, height), 1), torch.ones([1, width]),)

        x_t = x_t.expand([height, width])
        y_t = y_t.expand([height, width])
        x_t = x_t.to(self._device)
        y_t = y_t.to(self._device)

        return x_t, y_t

    def repeat(self, x, n_repeats):
        rep = torch.transpose(torch.unsqueeze(torch.ones(n_repeats), 1), 1, 0)
        rep = rep.long()
        x = torch.matmul(torch.reshape(x, (-1, 1)), rep)
        x = x.to(self._device)
        return torch.squeeze(torch.reshape(x, (-1, 1)))

    def interpolate(self, im, x, y):

        im = F.pad(im, (0, 0, 1, 1, 1, 1, 0, 0))

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
        dim1 = width * height
        base = self.repeat(torch.arange(0, batch_size) * dim1, out_height * out_width)

        base_y0 = base + y0 * dim2
        base_y1 = base + y1 * dim2

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

        wa = (dx * dy).transpose(1, 0)
        wb = (dx * (1 - dy)).transpose(1, 0)
        wc = ((1 - dx) * dy).transpose(1, 0)
        wd = ((1 - dx) * (1 - dy)).transpose(1, 0)

        output = torch.sum(torch.squeeze(torch.stack([wa * Ia, wb * Ib, wc * Ic, wd * Id], dim=1)), 1)
        output = torch.reshape(output, [-1, out_height, out_width, channels])
        return output

    def forward(self, moving_image, deformation_matrix):
        self._device = moving_image.device
        dx = deformation_matrix[:, :, :, 0]
        dy = deformation_matrix[:, :, :, 1]

        batch_size, height, width = dx.shape

        x_mesh, y_mesh = self.meshgrid(height, width)

        x_mesh = x_mesh.expand([batch_size, height, width])
        y_mesh = y_mesh.expand([batch_size, height, width])

        x_new = dx + x_mesh
        y_new = dy + y_mesh

        return self.interpolate(moving_image, x_new, y_new)

class Msi2A(nn.Module):
    def __init__(self, input_c, output_c, ngf=64, useClamp=True):
        super(Msi2A, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(input_c, ngf * 2, 1, 1, 0),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ngf * 2, ngf * 4, 1, 1, 0),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ngf * 4, ngf * 8, 1, 1, 0),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ngf * 8, output_c, 1, 1, 0),
        )

        self.softmax = nn.Softmax(dim=1)
        self.useClamp = useClamp

    def forward(self, x):
        if self.useClamp:
            return self.net(x).clamp(0, 1)
        else:
            return self.softmax(self.net(x))

class A2Img(nn.Module):
    def __init__(self, input_c, output_c):
        super(A2Img, self).__init__()
        self.net = nn.Sequential(nn.Conv2d(input_c, output_c, 1, 1, 0, bias=False),)

    def forward(self, x):
        return self.net(x).clamp(0, 1)

class HSI2A(nn.Module):
    def __init__(self, input_c, output_c, ngf=64, useClamp=True):
        super(HSI2A, self).__init__()
        self.net = nn.Sequential(
            nn.Conv2d(input_c, ngf * 2, 1, 1, 0),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ngf * 2, ngf * 4, 1, 1, 0),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ngf * 4, ngf * 8, 1, 1, 0),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ngf * 8, output_c, 1, 1, 0),
            # nn.Tanh()
        )
        self.softmax = nn.Softmax(dim=1)
        self.useClamp = useClamp

    def forward(self, x):
        if self.useClamp:
            return self.net(x).clamp(0, 1)
        else:
            return self.softmax(self.net(x))

class PSF(nn.Module):
    def __init__(self, scale):
        super(PSF, self).__init__()
        self.net = nn.Conv2d(1, 1, scale, scale, 0, bias=False)
        self.scale = scale
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        batch, channel, height, weight = list(x.size())
        return torch.cat([self.net(x[:, i, :, :].view(batch, 1, height, weight)) for i in range(channel)], 1,)

class ConvolutionHsi2Msi(nn.Module):
    def __init__(self, hsi_channels, msi_channels, sp_range):
        super(ConvolutionHsi2Msi, self).__init__()

        self.sp_range = sp_range.astype(int)
        self.length_of_each_band = self.sp_range[:, 1] - self.sp_range[:, 0] + 1
        self.length_of_each_band = self.length_of_each_band.tolist()
        # import ipdb
        # ipdb.set_trace()
        self.conv2d_list = nn.ModuleList([nn.Conv2d(x, 1, 1, 1, 0, bias=False) for x in self.length_of_each_band])
        # self.scale_factor_net = nn.Conv2d(1,1,1,1,0,bias=False)

    def forward(self, input):
        # batch,channel,height,weight = list(input.size())
        # scaled_intput = torch.cat([self.scale_factor_net(input[:,i,:,:].view(batch,1,height,weight)) for i in range(channel)], 1)
        scaled_intput = input
        cat_list = []
        for i, layer in enumerate(self.conv2d_list):
            input_slice = scaled_intput[:, self.sp_range[i, 0] : self.sp_range[i, 1] + 1, :, :]
            # input_slice = scaled_intput[:, :, :, :]
            out = layer(input_slice) / layer.weight.detach().sum(dim=1).view(1).clamp_min(1e-12)
            cat_list.append(out)
        return torch.cat(cat_list, 1).clamp(0, 1)

class ZeroOneClipper(object):
    def __call__(self, module):
        # filter the variables to get the ones you want
        if hasattr(module, "weight"):
            w = module.weight.data
            w.clamp_(0, 1)

class SumToOneClipper(object):
    def __call__(self, module):
        if hasattr(module, "weight"):
            if module.in_channels != 1:
                w = module.weight.data
                w.clamp_(0, 10)
                denom = w.sum(dim=1, keepdim=True)
                w.copy_(torch.where(denom > 0, w / denom.clamp_min(1e-12), torch.full_like(w, 1.0 / w.shape[1])))
            elif module.in_channels == 1:
                w = module.weight.data
                w.clamp_(0, 5)
