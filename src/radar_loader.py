"""
Author: Wouter Van Gansbeke
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
"""

import os
import sys
import re

sys.path.insert(1, os.path.join(sys.path[0], '..'))


class radar_preprocessing(object):
    def __init__(self, dataset_path):
        self.train_paths = {'img': [], 'lidar': [], 'radar': [], 'mask': []}
        self.val_paths = {'img': [], 'lidar': [], 'radar': [], 'mask': []}
        self.dataset_path = dataset_path

    def get_paths(self):
        # train and validation dirs
        for type_set in os.listdir(self.dataset_path):
            for root, dirs, files in os.walk(os.path.join(self.dataset_path, type_set)):
                self.train_paths['radar'].extend(sorted([os.path.join(root, file) for file in files
                                                         if re.search('radar', root)
                                                         and re.search('train', root)
                                                         and re.search('png', file)]))
                self.val_paths['radar'].extend(sorted([os.path.join(root, file) for file in files
                                                       if re.search('radar', root)
                                                       and re.search('val', root)
                                                       and re.search('png', file)]))
                self.train_paths['lidar'].extend(sorted([os.path.join(root, file) for file in files
                                                         if re.search('lidar', root)
                                                         and re.search('train', root)
                                                         and re.search('png', file)]))
                self.val_paths['lidar'].extend(sorted([os.path.join(root, file) for file in files
                                                       if re.search('lidar', root)
                                                       and re.search('val', root)
                                                       and re.search('png', file)]))
                self.train_paths['mask'].extend(sorted([os.path.join(root, file) for file in files
                                                        if re.search('mask', root)
                                                        and re.search('train', root)
                                                        and re.search('png', file)]))
                self.val_paths['mask'].extend(sorted([os.path.join(root, file) for file in files
                                                      if re.search('mask', root)
                                                      and re.search('val', root)
                                                      and re.search('png', file)]))
                self.train_paths['img'].extend(sorted([os.path.join(root, file) for file in files
                                                       if re.search('img', root)
                                                       and re.search('train', root)
                                                       and re.search('png', file)]))
                self.val_paths['img'].extend(sorted([os.path.join(root, file) for file in files
                                                     if re.search('img', root)
                                                     and re.search('val', root)
                                                     and re.search('png', file)]))

    def prepare_dataset(self):
        self.get_paths()
        print('img   in training dataset: ', len(self.train_paths['img']))
        print('radar in training dataset: ', len(self.train_paths['radar']))
        print('lidar in training dataset: ', len(self.train_paths['lidar']))
        print('mask  in training dataset: ', len(self.train_paths['mask']))
        print('img   in val/test dataset: ', len(self.val_paths['img']))
        print('radar in val/test dataset: ', len(self.val_paths['radar']))
        print('lidar in val/test dataset: ', len(self.val_paths['lidar']))
        print('mask  in val/test dataset: ', len(self.val_paths['mask']))


if __name__ == '__main__':

    # Imports
    import tqdm
    from PIL import Image
    import os
    import argparse
    from utils import str2bool

    # arguments
    parser = argparse.ArgumentParser(description='Preprocess')
    parser.add_argument("--png2img", type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument("--calc_params", type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument('--num_samples', default=0, type=int, help='number of samples')
    parser.add_argument('--datapath', default='../dataset')
    parser.add_argument('--dest', default='/usr/data/tmp/')
    args = parser.parse_args()

    dataset = radar_preprocessing(args.datapath)
    dataset.prepare_dataset()
    # if args.png2img:
    #     os.makedirs(os.path.join(args.dest, 'Rgb'), exist_ok=True)
    #     destination_train = os.path.join(args.dest, 'Rgb/train')
    #     destination_valid = os.path.join(args.dest, 'Rgb/val')
    #     dataset.convert_png_to_rgb(dataset.train_paths['img'], destination_train)
    #     dataset.convert_png_to_rgb(dataset.val_paths['img'], destination_valid)
    if args.calc_params:
        import matplotlib.pyplot as plt

        params = dataset.compute_mean_std()
        mu_std = params[0:2]
        max_lst = params[-1]
        print('Means and std equals {} and {}'.format(*mu_std))
        plt.hist(max_lst, bins='auto')
        plt.title('Histogram for max depth')
        plt.show()
        # mean, std = 14.969576188369581, 11.149000139428104
        # Normalized
        # mean, std = 0.17820924033773314, 0.1327261921360489
    # if args.num_samples != 0:
    #     print("Making downsampled dataset")
    #     os.makedirs(os.path.join(args.dest), exist_ok=True)
    #     destination_train = os.path.join(args.dest, 'train')
    #     destination_valid = os.path.join(args.dest, 'val')
    #     dataset.downsample(dataset.train_paths['lidar_in'], destination_train, args.num_samples)
    #     dataset.downsample(dataset.val_paths['lidar_in'], destination_valid, args.num_samples)
