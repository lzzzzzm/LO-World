import torch
import torch.nn as nn

from einops import rearrange

from mmdet.models.backbones.resnet import BasicBlock as ResBasicBlock

class MotionNet(nn.Module):
    def __init__(self, embed_dims):
        super(MotionNet, self).__init__()
        self.embed_dims = embed_dims
        self.trans_embedding = nn.Sequential(
            nn.Linear(12, embed_dims),
            nn.ReLU(True),
            nn.Linear(embed_dims, embed_dims)
        )

    def forward(self, curr_to_futu_info):
        trans_embedding = self.trans_embedding(curr_to_futu_info)
        # trans_embedding: [bs, f, c]
        return trans_embedding

class LayoutNet(nn.Module):
    def __init__(self, in_channel, embed_dims):
        super(LayoutNet, self).__init__()
        self.embed_dims = embed_dims
        # Two Resnet Downsample layer
        self.layout_embedding = nn.Sequential(
            ResBasicBlock(
                inplanes=in_channel,
                planes=embed_dims//2,
                stride=2,
                downsample=nn.Conv2d(in_channel, embed_dims//2, kernel_size=1, stride=2)
            ),
            ResBasicBlock(
                inplanes=embed_dims//2,
                planes=embed_dims,
                stride=2,
                downsample=nn.Conv2d(embed_dims//2, embed_dims, kernel_size=1, stride=2)
            ),
        )

    def forward(self, bev_layout):
        reshape = False
        if bev_layout.dim() == 5:
            # bev_layout: [bs, f, c, h, w]
            bs, f, c, h, w = bev_layout.shape
            bev_layout = bev_layout.view(bs * f, c, h, w)
            reshape = True

        # bev_layout: [bs, c, h, w]
        layout_embedding = self.layout_embedding(bev_layout)  # [bs*f, c, h/4, w/4]
 
        # change to [b, s, c]
        layout_embedding = rearrange(layout_embedding, 'b c h w -> b (h w) c')

        if reshape:
            layout_embedding = layout_embedding.view(bs, f, -1, self.embed_dims)

        return layout_embedding
