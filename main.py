"""
Copyright (C) 2020 NVIDIA Corporation.  All rights reserved.
Licensed under the NVIDIA Source Code License. See LICENSE at https://github.com/nv-tlabs/lift-splat-shoot.
Authors: Jonah Philion and Sanja Fidler
"""

from fire import Fire

import src


if __name__ == '__main__':
    Fire({
        'lidar_check': src.explore.lidar_check,
        'cumsum_check': src.explore.cumsum_check,

        'train': src.train.train,
        'eval_model_iou': src.explore.eval_model_iou,
        'viz_model_preds': src.explore.viz_model_preds,
        'model_test': src.test.model_test,
    })

# python main.py eval_model_iou mini --modelf=model525000.pt --dataroot=mini
# python main.py lidar_check mini --dataroot=mini --viz_train=False
# python main.py train mini --dataroot=mini --logdir=./runs --gpuids=0 tensorboard --logdir=./runs --bind_all