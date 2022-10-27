"""
Copyright (C) 2020 NVIDIA Corporation.  All rights reserved.
Licensed under the NVIDIA Source Code License. See LICENSE at https://github.com/nv-tlabs/lift-splat-shoot.
Authors: Jonah Philion and Sanja Fidler
"""

import torch
from time import time
from tensorboardX import SummaryWriter
import numpy as np
import os

from .models import compile_model
from .data import compile_data
from .tools import SimpleLoss, get_batch_iou, get_val_info

from .dataloader import get_loader
from .radar_loader import radar_preprocessing

class CrossEntropyLoss2d(torch.nn.Module):
    def __init__(self, weight=None):
        super(CrossEntropyLoss2d, self).__init__()
        self.loss = torch.nn.NLLLoss2d(weight)
    def forward(self, outputs, targets, mask):
        t = targets * mask
        return self.loss(torch.nn.functional.log_softmax(outputs*mask, dim=1), t[:, 0].long())

class BCELoss(torch.nn.Module):
    def __init__(self):
        super(BCELoss, self).__init__()
        self.loss_fn = torch.nn.BCELoss()

    def forward(self, ypred, ytgt, mask):
        loss = self.loss_fn(ypred.sigmoid() * mask, ytgt * mask)
        return loss

def train(gpuid=1,

          H=512, W=1024,
          resize_lim=(0.193, 0.225),
          final_dim=(512, 1024),
          bot_pct_lim=(0.0, 0.22),
          rot_lim=(-5.4, 5.4),
          rand_flip=True,
          ncams=1,
          max_grad_norm=5.0,
          pos_weight=2.13,
          logdir='./runs',

          xbound=[-75.0, 75.0, 0.25],
          ybound=[0.0, 75.0, 0.25],
          zbound=[-10.0, 10.0, 20.0],
          dbound=[4.0, 75.0, 1.0],

          bsz=2,
          nworkers=10,
          lr=1e-3,
          weight_decay=1e-7,
          ):
    grid_conf = {
        'xbound': xbound,
        'ybound': ybound,
        'zbound': zbound,
        'dbound': dbound,
    }
    data_aug_conf = {
        'resize_lim': resize_lim,
        'final_dim': final_dim,
        'rot_lim': rot_lim,
        'H': H, 'W': W,
        'rand_flip': rand_flip,
        'bot_pct_lim': bot_pct_lim,
        'cams': ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
                 'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT'],
        'Ncams': ncams,
    }
    # INIT datasetd
    args = {
        # img -> (600, 1200), others -> (301, 601)
        'crop_h': 600,
        'crop_w': 300,
        'no_aug': None,
        'data_path': '/media/personal_data/lizc/2/lift-splat-shoot/dataset',
        'rotate': False,
        'flip': None,  # if 'hflip', the inverse-projection will ?
        'batch_size': bsz,
        'nworkers': 4,
        'val_batch_size': bsz,
        'nworkers_val': 4,

    }
    dataset = radar_preprocessing(args['data_path'])
    dataset.prepare_dataset()
    trainloader, validloader = get_loader(args, dataset)
    # trainloader, valloader = compile_data(version, dataroot, data_aug_conf=data_aug_conf,
    #                                       grid_conf=grid_conf, bsz=bsz, nworkers=nworkers,
    #                                       parser_name='segmentationdata')
    # for batchi, (imgs, rots, trans, intrins, post_rots, post_trans, binimgs) in enumerate(trainloader):
    #     print(imgs.shape, binimgs.shape, post_rots.shape)
    #     raise NotImplemented
    device = torch.device('cpu') if gpuid < 0 else torch.device(f'cuda:{gpuid}')

    model = compile_model(grid_conf, data_aug_conf, outC=1)
    model.to(device)

    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # loss_fn = SimpleLoss(pos_weight).cuda(gpuid)
    # loss_fn = CrossEntropyLoss2d().cuda(gpuid)
    loss_fn = BCELoss().cuda(gpuid)

    writer = SummaryWriter(logdir=logdir)
    val_step = 1000

    model.train()
    counter = 0
    for epoch in range(620):
        np.random.seed()
        for batchi, (imgs, radars, lidars, masks) in enumerate(trainloader):
            t0 = time()
            opt.zero_grad()
            preds = model(imgs.to(device),
                          radars.to(device)
                          )
            lidars = lidars.to(device)
            masks = masks.to(device)
            loss = loss_fn(preds, lidars, masks)  # insert mask in loss_fn
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            opt.step()
            counter += 1
            t1 = time()

            if counter % 10 == 0:
                print(epoch, counter, loss.item())
                writer.add_scalar('train/loss', loss, counter)

            if counter % 50 == 0:
                _, _, iou = get_batch_iou(preds, lidars, masks)
                writer.add_scalar('train/iou', iou, counter)
                writer.add_scalar('train/epoch', epoch, counter)
                writer.add_scalar('train/step_time', t1 - t0, counter)

            # if counter % val_step == 0:
            #     val_info = get_val_info(model, valloader, loss_fn, device)
            #     print('VAL', val_info)
            #     writer.add_scalar('val/loss', val_info['loss'], counter)
            #     writer.add_scalar('val/iou', val_info['iou'], counter)

            # if counter % val_step == 0:
            #     model.eval()
            #     mname = os.path.join(logdir, "model{}.pt".format(counter))
            #     print('saving', mname)
            #     torch.save(model.state_dict(), mname)
            #     model.train()
        if epoch % 100 == 0:
            model.eval()
            mname = os.path.join(logdir, "model-{}.pt".format(epoch))
            print('saving', mname)
            torch.save(model.state_dict(), mname)
            model.train()
    # model.eval()
    # mname = os.path.join(logdir, "model-{}.pt".format(epoch))
    # print('saving', mname)
    # torch.save(model.state_dict(), mname)
    # model.train()