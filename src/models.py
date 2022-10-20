"""
Copyright (C) 2020 NVIDIA Corporation.  All rights reserved.
Licensed under the NVIDIA Source Code License. See LICENSE at https://github.com/nv-tlabs/lift-splat-shoot.
Authors: Jonah Philion and Sanja Fidler
"""

import torch
from torch import nn
from efficientnet_pytorch import EfficientNet
from torchvision.models.resnet import resnet18
import numpy as np
from scipy.linalg import pinv, inv
from .tools import gen_dx_bx, cumsum_trick, QuickCumsum


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super().__init__()

        self.up = nn.Upsample(scale_factor=scale_factor, mode='bilinear',
                              align_corners=True)

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x1 = torch.cat([x2, x1], dim=1)
        return self.conv(x1)


class CamEncode(nn.Module):
    def __init__(self, D, C, downsample):
        super(CamEncode, self).__init__()
        self.D = D
        self.C = C

        self.trunk = EfficientNet.from_pretrained("efficientnet-b0")

        self.up1 = Up(320 + 112, 512)
        self.depthnet = nn.Conv2d(512, self.D + self.C, kernel_size=1, padding=0)

    def get_depth_dist(self, x, eps=1e-20):
        return x.softmax(dim=1)

    def get_depth_feat(self, x):
        x = self.get_eff_depth(x)
        # Depth
        x = self.depthnet(x)

        depth = self.get_depth_dist(x[:, :self.D])
        new_x = depth.unsqueeze(1) * x[:, self.D:(self.D + self.C)].unsqueeze(2)

        return depth, new_x

    def get_eff_depth(self, x):
        # adapted from https://github.com/lukemelas/EfficientNet-PyTorch/blob/master/efficientnet_pytorch/model.py#L231
        endpoints = dict()
        # Stem
        x = self.trunk._swish(self.trunk._bn0(self.trunk._conv_stem(x)))
        prev_x = x # shape: 4, 32, 300, 600
        # Blocks
        for idx, block in enumerate(self.trunk._blocks):
            drop_connect_rate = self.trunk._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self.trunk._blocks)  # scale drop connect_rate
            x = block(x, drop_connect_rate=drop_connect_rate)
            if prev_x.size(2) > x.size(2):
                endpoints['reduction_{}'.format(len(endpoints) + 1)] = prev_x
            prev_x = x

        # Head
        endpoints['reduction_{}'.format(len(endpoints) + 1)] = x
        x = self.up1(endpoints['reduction_5'], endpoints['reduction_4'])
        return x

    def forward(self, x):
        depth, x = self.get_depth_feat(x)

        return x


class BevEncode(nn.Module):
    def __init__(self, inC, outC):
        super(BevEncode, self).__init__()

        trunk = resnet18(pretrained=False, zero_init_residual=True)
        self.conv1 = nn.Conv2d(inC, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = trunk.bn1
        self.relu = trunk.relu

        self.layer1 = trunk.layer1
        self.layer2 = trunk.layer2
        self.layer3 = trunk.layer3

        # self.up1 = Up(64 + 256, 256, scale_factor=4)
        self.up1 = Up(64 + 256, 256, scale_factor=(4, 150/38))
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear',
                        align_corners=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, outC, kernel_size=1, padding=0),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x1 = self.layer1(x)
        x = self.layer2(x1)
        x = self.layer3(x)

        x = self.up1(x, x1)
        x = self.up2(x)

        return x


class LiftSplatShoot(nn.Module):
    def __init__(self, grid_conf, data_aug_conf, outC):
        super(LiftSplatShoot, self).__init__()
        self.grid_conf = grid_conf
        self.data_aug_conf = data_aug_conf

        dx, bx, nx = gen_dx_bx(self.grid_conf['xbound'],
                               self.grid_conf['ybound'],
                               self.grid_conf['zbound'],
                               )
        self.dx = nn.Parameter(dx, requires_grad=False)
        self.bx = nn.Parameter(bx, requires_grad=False)
        self.nx = nn.Parameter(nx, requires_grad=False)

        self.downsample = 16  # The downsample rate we use should be ?
        self.camC = 64
        self.frustum = self.create_frustum()
        self.D, _, _, _ = self.frustum.shape
        self.camencode = CamEncode(self.D, self.camC, self.downsample)
        self.bevencode = BevEncode(inC=self.camC, outC=outC)

        # toggle using QuickCumsum vs. autograd
        self.use_quickcumsum = True

        # inverse matrix
        matrix = np.asarray([[2019.613635, 1745.881668, -111.4337968, -419.9388818],
                             [26.01936737, 870.7969811, -2038.300785, -120.9971104],
                             [0.02443084799, 0.997614078, -0.06457000164, -0.006415358346]])
        m = np.vstack([matrix, np.asarray([0, 0, 0, 1])])
        self.inv = torch.Tensor(inv(m))

    def create_frustum(self):
        # make grid in image plane
        ogfH, ogfW = self.data_aug_conf['final_dim']  # 128, 352
        fH, fW = ogfH // self.downsample, ogfW // self.downsample  # 8, 22
        ds = torch.arange(*self.grid_conf['dbound'], dtype=torch.float).view(-1, 1, 1).expand(-1, fH, fW)
        D, _, _ = ds.shape
        xs = torch.linspace(0, ogfW - 1, fW, dtype=torch.float).view(1, 1, fW).expand(D, fH,
                                                                                      fW)  # from 0 to 351, 22 bins
        ys = torch.linspace(0, ogfH - 1, fH, dtype=torch.float).view(1, fH, 1).expand(D, fH,
                                                                                      fW)  # from 0 to 128, 8 bins
        # D x H x W x 3
        frustum = torch.stack((xs, ys, ds), -1)
        return nn.Parameter(frustum, requires_grad=False)

    def get_geometry(self, x):
        """Determine the (x,y,z) locations (in the ego frame)
        of the points in the point cloud.
        Returns B x N x D x H/downsample x W/downsample x 3
        """
        B, N, _, h_, w_ = x.shape
        # rots: 3x3 ,trans: 1x3, intrins: 3x3, post_rots:3x3, post_trans: 1x1
        # undo post-transformation
        # B x N x D x H x W x 3
        # self.frustum is 41x8x22x3
        D, H, W, _ = self.frustum.shape
        # points = self.frustum - post_trans.view(B, N, 1, 1, 1, 3)  # points' shape will be 1x1x41x8x22x3
        points = self.frustum.repeat(B, N, 1, 1, 1, 1)
        # points = torch.inverse(post_rots).view(B, N, 1, 1, 1, 3, 3).matmul(
        #     points.unsqueeze(-1))  # insert dimension at position '-1'
        # cam_to_ego
        points = torch.cat((points[:, :, :, :, :, :1] * points[:, :, :, :, :, 2:3] * 1700 / h_,
                            points[:, :, :, :, :, 1:2] * points[:, :, :, :, :, 2:3] * 3517 / w_,
                            points[:, :, :, :, :, 2:3],
                            torch.ones_like(points[:, :, :, :, :, 2:3])
                            ), 5)
        combine = self.inv.to(x.device)
        combine = combine.repeat(B, N, 1, 1, 1, 1).transpose(-1, -2)
        # points = combine.repeat(B, N, 1, 1, 1, 1, 1).matmul(points).squeeze(-1)
        points = points.matmul(combine)
        # points += trans.view(B, N, 1, 1, 1, 3)  # points: batch, ncams, 41, 8, 22, 3, under XYZ coordinates
        return points[..., :3]

    def get_cam_feats(self, x):
        """Return B x N x D x H/downsample x W/downsample x C
        """
        B, N, C, imH, imW = x.shape

        x = x.view(B * N, C, imH, imW)
        x = self.camencode(x) # shape: 512/16 1024/16
        # print(x.shape) # 4 64, 71, 32, 64
        x = x.view(B, N, self.camC, self.D, imH // self.downsample, imW // self.downsample)
        x = x.permute(0, 1, 3, 4, 5, 2)  # camera features: batch, ncams, 41, 8, 22, 64

        return x

    def voxel_pooling(self, geom_feats, x):
        B, N, D, H, W, C = x.shape  # D is distance, from 4m to 45m
        Nprime = B * N * D * H * W

        # flatten x
        x = x.reshape(Nprime, C)

        # flatten indices
        # geom_feats is positions under XYZ coordinate
        # print(torch.min(geom_feats[..., 0]), torch.max(geom_feats[..., 0])) # from 5.67 to 45.89
        # print(torch.min(geom_feats[..., 1]), torch.max(geom_feats[..., 1]))  # from -26.76 to 28.65
        # print(torch.min(geom_feats[..., 2]), torch.max(geom_feats[..., 2]))  # from -11.99 to 9.00
        geom_feats = ((geom_feats - (self.bx - self.dx / 2.)) / self.dx).long()
        # print(geom_feats.shape)
        # print(torch.min(geom_feats[..., 0]), torch.max(geom_feats[..., 0])) # from 111 to 192
        # print(torch.min(geom_feats[..., 1]), torch.max(geom_feats[..., 1])) # from 46 to 159
        # print(torch.min(geom_feats[..., 2]), torch.max(geom_feats[..., 2])) # from 0 to 0
        # print(self.bx) # tensor([-49.7500, -49.7500,   0.0000], device='cuda:1')
        # print(self.dx) # tensor([ 0.5000,  0.5000, 20.0000], device='cuda:1')
        # i think, frustum is under image coordinate, thus, HxWxD
        # geom_feats is under XYZ coordinate, thus, 200x200 in BEV
        geom_feats = geom_feats.view(Nprime, 3)
        batch_ix = torch.cat([torch.full([Nprime // B, 1], ix,
                                         device=x.device, dtype=torch.long) for ix in range(B)])
        geom_feats = torch.cat((geom_feats, batch_ix), 1)
        # print(self.nx) # tensor([200, 200,   1], device='cuda:1')
        # filter out points that are outside box
        kept = (geom_feats[:, 0] >= 0) & (geom_feats[:, 0] < self.nx[0]) \
               & (geom_feats[:, 1] >= 0) & (geom_feats[:, 1] < self.nx[1]) \
               & (geom_feats[:, 2] >= 0) & (geom_feats[:, 2] < self.nx[2])
        x = x[kept]
        geom_feats = geom_feats[kept]

        # get tensors from the same voxel next to each other
        ranks = geom_feats[:, 0] * (self.nx[1] * self.nx[2] * B) \
                + geom_feats[:, 1] * (self.nx[2] * B) \
                + geom_feats[:, 2] * B \
                + geom_feats[:, 3]
        sorts = ranks.argsort()  # from small to big
        x, geom_feats, ranks = x[sorts], geom_feats[sorts], ranks[sorts]
        # print(geom_feats.shape, x.shape) # shape: 7216, 4
        # print(geom_feats[:5, :])
        # print(ranks[:5])
        # for example, the geom_feats[:5, :] are all [111, 95, 0, 0]
        # with cumsum trick, the geom_feats[:5, :] are [111, 95, 0, 0] etc, from 95 to 99
        # cumsum trick
        # camera points decrease rapidly
        if not self.use_quickcumsum:
            x, geom_feats = cumsum_trick(x, geom_feats, ranks)
        else:
            x, geom_feats = QuickCumsum.apply(x, geom_feats, ranks)
        # cumsum trick: |----point A----| the feature is the sum of the feature of points whose position is A
        # print(geom_feats.shape) # shape: 1266, 4
        # griddify (B x C x Z x X x Y)
        final = torch.zeros((B, C, self.nx[2], self.nx[0], self.nx[1]), device=x.device)
        final[geom_feats[:, 3], :, geom_feats[:, 2], geom_feats[:, 0], geom_feats[:, 1]] = x

        # collapse Z
        final = torch.cat(final.unbind(dim=2), 1)

        return final

    def get_voxels(self, x):
        geom = self.get_geometry(x)
        x = self.get_cam_feats(x)

        x = self.voxel_pooling(geom, x)

        return x

    def forward(self, x, radars):
        x = self.get_voxels(x)

        x = self.bevencode(x)
        return x


def compile_model(grid_conf, data_aug_conf, outC):
    return LiftSplatShoot(grid_conf, data_aug_conf, outC)
