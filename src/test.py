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
from PIL import Image
from .models import compile_model
from .data import compile_data
from .tools import SimpleLoss, get_batch_iou, get_val_info

from .dataloader import get_loader
from .radar_loader import radar_preprocessing


def model_test(gpuid=1,

               H=600, W=1200,
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

               bsz=1,
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
    device = torch.device('cpu') if gpuid < 0 else torch.device(f'cuda:{gpuid}')

    model = compile_model(grid_conf, data_aug_conf, outC=1)
    model.to(device)

    model = compile_model(grid_conf, data_aug_conf, outC=1)
    print('loading')
    model.load_state_dict(torch.load('./runs/model-3000.pt'))
    model.to(device)

    model.eval()
    for batchi, (imgs, radars, lidars, masks) in enumerate(validloader):
        preds = model(imgs.to(device),
                      radars.to(device)
                      )
        lidars = lidars.to(device)
        masks = masks.to(device)

        p = preds[0, 0, :, :].detach().cpu().numpy()
        m = masks[0, 0, :, :].detach().cpu().numpy()
        p = p * m
        print(p.shape)
        for i in range(p.shape[0]):
            for j in range(p.shape[1]):
                if p[i, j] > 0.5:
                    p[i, j] = 1
                else:
                    p[i, j] = 0
        Image.fromarray(np.uint8(p * 255).T[::-1, :]).save(f'results/val/{batchi}.png')
        # raise NotImplemented
