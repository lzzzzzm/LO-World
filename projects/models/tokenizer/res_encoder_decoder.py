# Modified from OccWorld
import torch
import torch.nn.functional as F
import torch.nn as nn

from mmdet3d.registry import MODELS
from mmengine.model import BaseModule


def nonlinearity(x):
    # swish
    return x * torch.sigmoid(x)


def Normalize(in_channels):
    if in_channels <= 32:
        num_groups = in_channels // 4
    else:
        num_groups = 32
    return nn.GroupNorm(num_groups=num_groups, num_channels=in_channels, eps=1e-6, affine=True)


class Upsample(nn.Module):
    def __init__(self, in_channels, with_conv, stride=2):
        super().__init__()
        self.with_conv = with_conv
        self.stride = stride
        if with_conv:
            self.conv = nn.Conv2d(in_channels, in_channels, 3, 1, 1)

    def forward(self, x, shape):
        # Use stride as the scale factor.
        x = torch.nn.functional.interpolate(x, scale_factor=self.stride, mode='nearest')

        # Align the upsampled feature map with the target spatial shape.
        diffY = shape[0] - x.size()[2]
        diffX = shape[1] - x.size()[3]
        x = F.pad(x, [diffX // 2, diffX - diffX // 2,
                      diffY // 2, diffY - diffY // 2])

        if self.with_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    def __init__(self, in_channels, with_conv, stride=2):
        super().__init__()
        self.with_conv = with_conv
        self.stride = stride
        if with_conv:
            # Use the specified stride.
            self.conv = nn.Conv2d(in_channels, in_channels, 3, stride=stride, padding=1)

    def forward(self, x):
        if self.with_conv:
            x = self.conv(x)
        else:
            # Use the same stride for pooling.
            x = torch.nn.functional.avg_pool2d(x, kernel_size=self.stride, stride=self.stride)
        return x


class AsymmetricDownsample(nn.Module):
    """Downsample with a configurable stride."""

    def __init__(self, in_channels, with_conv, stride=(1, 2)):
        super().__init__()
        self.with_conv = with_conv
        self.stride = stride
        if with_conv:
            # Set the stride from the constructor argument, e.g. (1, 2).
            self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3,
                                  stride=stride, padding=1)

    def forward(self, x):
        if self.with_conv:
            x = self.conv(x)
        else:
            x = torch.nn.functional.avg_pool2d(x, kernel_size=self.stride, stride=self.stride)
        return x


class AsymmetricUpsample(nn.Module):
    """Upsample with configurable scale factors."""

    def __init__(self, in_channels, with_conv, stride=(1, 2)):
        super().__init__()
        self.with_conv = with_conv
        self.stride = stride
        if with_conv:
            self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x, shape=None):
        # Use the stride tuple as the scale factor.
        x = torch.nn.functional.interpolate(x, scale_factor=self.stride, mode='nearest')
        if self.with_conv:
            x = self.conv(x)
        return x


class ResnetBlock(nn.Module):
    def __init__(self, in_channels, out_channels=None, conv_shortcut=False):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = Normalize(in_channels)
        self.conv1 = torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)

        self.norm2 = Normalize(out_channels)
        self.conv2 = torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
            else:
                self.nin_shortcut = torch.nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x, temp):
        h = x
        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.conv1(h)

        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)

        return x + h


class AttnBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels)
        self.q = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.k = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.v = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.proj_out = torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b, c, h, w = q.shape
        q = q.reshape(b, c, h * w)
        q = q.permute(0, 2, 1)  # b,hw,c
        k = k.reshape(b, c, h * w)  # b,c,hw
        w_ = torch.bmm(q, k)  # b,hw,hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
        w_ = w_ * (int(c) ** (-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # attend to values
        v = v.reshape(b, c, h * w)
        w_ = w_.permute(0, 2, 1)  # b,hw,hw (first hw of k, second of q)
        h_ = torch.bmm(v, w_)  # b, c,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
        h_ = h_.reshape(b, c, h, w)

        h_ = self.proj_out(h_)

        return x + h_


@MODELS.register_module()
class Encoder2D(BaseModule):
    def __init__(self, *, ch, num_res_blocks, in_channels, out_channels, ch_mult=(1, 2, 4, 8),
                 attn_level=(2,), downsample_level=(0, 1), downsample_type=(0, 0, 0, 0), stride=(2, 2, 2, 2),
                 resamp_with_conv=True,
                 **ignore_kwargs):
        super().__init__()
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.in_channels = in_channels
        self.downsample_level = downsample_level

        self.conv_in = torch.nn.Conv2d(in_channels, self.ch, kernel_size=3, stride=1, padding=1)

        in_ch_mult = (1,) + tuple(ch_mult)
        self.down = nn.ModuleList()
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch * in_ch_mult[i_level]
            block_out = ch * ch_mult[i_level]
            for i_block in range(self.num_res_blocks[i_level]):
                block.append(ResnetBlock(in_channels=block_in, out_channels=block_out))
                block_in = block_out
                if i_level in attn_level:
                    attn.append(AttnBlock(block_in))
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level in downsample_level:
                if downsample_type[i_level] == 0:
                    down.downsample = Downsample(block_in, resamp_with_conv, stride=stride[i_level])
                elif downsample_type[i_level] == 1:
                    down.downsample = AsymmetricDownsample(block_in, resamp_with_conv, stride=(1, stride[i_level]))
                elif downsample_type[i_level] == 2:
                    down.downsample = AsymmetricDownsample(block_in, resamp_with_conv, stride=(stride[i_level], 1))

            self.down.append(down)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in, out_channels=block_in)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(in_channels=block_in, out_channels=block_in)

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(block_in, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        # x: bs, C, H, W
        shapes = []
        temb = None

        h = self.conv_in(x)
        for i_level in range(self.num_resolutions):

            for i_block in range(self.num_res_blocks[i_level]):
                h = self.down[i_level].block[i_block](h, temb)

                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)

            if i_level in self.downsample_level:
                shapes.append(h.shape[-2:])
                h = self.down[i_level].downsample(h)

        #
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        # end
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h, shapes


@MODELS.register_module()
class Decoder2D(BaseModule):
    def __init__(self, *, ch, in_channels, out_channels, ch_mult=(1, 2, 4, 8), num_res_blocks,
                 attn_level=(0,), upsample_level=(1, 2), upsample_type=(0, 0, 0, 0), stride=(2, 2, 2, 2),
                 resamp_with_conv=True,
                 dual_out_channels=None, **ignorekwargs):
        super().__init__()
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.in_channels = in_channels
        self.attn_level = attn_level
        self.upsample_level = upsample_level
        self.dual_out_channels = dual_out_channels

        # compute in_ch_mult, block_in and curr_res at lowest res
        in_ch_mult = (1,) + tuple(ch_mult)
        block_in = ch * ch_mult[self.num_resolutions - 1]

        # z to block_in
        self.conv_in = torch.nn.Conv2d(in_channels, block_in, kernel_size=3, stride=1, padding=1)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in, out_channels=block_in, )
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(in_channels=block_in, out_channels=block_in, )

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch * ch_mult[i_level]

            for i_block in range(self.num_res_blocks[i_level]):  # change this to align with encoder
                block.append(ResnetBlock(in_channels=block_in, out_channels=block_out, ))
                block_in = block_out
                if i_level in attn_level:
                    attn.append(AttnBlock(block_in))

            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level in upsample_level:
                if upsample_type[i_level] == 0:
                    up.upsample = Upsample(block_in, resamp_with_conv, stride=stride[i_level])
                elif upsample_type[i_level] == 1:
                    up.upsample = AsymmetricUpsample(block_in, resamp_with_conv, stride=(1, stride[i_level]))
                elif upsample_type[i_level] == 2:
                    up.upsample = AsymmetricUpsample(block_in, resamp_with_conv, stride=(stride[i_level], 1))

            self.up.insert(0, up)  # prepend to get consistent order

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(block_in, out_channels, kernel_size=3, stride=1, padding=1)

        if self.dual_out_channels is not None:
            self.norm_out_2 = Normalize(block_in)
            self.conv_out_2 = torch.nn.Conv2d(block_in, dual_out_channels, kernel_size=3, stride=1, padding=1)

    def forward_features(self, z, shapes):
        # z: bs*F, C, H, W
        self.last_z_shape = z.shape

        # timestep embedding
        temb = None

        # z to block_in
        h = self.conv_in(z)

        # middle
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        # upsampling
        for i_level in reversed(range(self.num_resolutions)):

            for i_block in range(self.num_res_blocks[i_level]):  # change this to align encoder
                h = self.up[i_level].block[i_block](h, temb)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)

            if i_level in self.upsample_level:
                h = self.up[i_level].upsample(h, shapes.pop())

        return h

    def forward_heads(self, h):
        """Apply the task-specific output heads to a shared decoder feature."""

        out_h = self.norm_out(h)
        out_h = nonlinearity(out_h)
        out_h = self.conv_out(out_h)

        if self.dual_out_channels:
            out_h_2 = self.norm_out_2(h)
            out_h_2 = nonlinearity(out_h_2)
            out_h_2 = self.conv_out_2(out_h_2)
            return out_h, out_h_2

        return out_h

    def forward(self, z, shapes):
        h = self.forward_features(z, shapes)
        return self.forward_heads(h)
