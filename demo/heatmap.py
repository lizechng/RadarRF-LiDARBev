import os
import cv2 as cv
import random
import pandas as pd
from tqdm import tqdm
import plotly.graph_objects as go
import plotly.io as pio
import numpy as np
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import scipy
import scipy.ndimage as ndimage
import scipy.ndimage.filters as filters



def load_TIRadarHeatmap(heatmap_path):
    '''
    read TI radar heatmap
    :param heatmap_path: str - TI radar heatmap path
    :return: dict(np.array)
    '''
    data = np.fromfile(heatmap_path, dtype='float32')
    # data = data.reshape((232, 4*257)).T
    # data = data.reshape((4, 257, 232))
    data = data.reshape((4*257, 232), order='F')
    data = data.reshape((4, 257, 232))
    res = {
        "heatmap_static": data[0, :, :],
        "heatmap_dynamic": data[1, :, :],
        "x_bins": data[2, :, :],
        "y_bins": data[3, :, :],
    }
    return res


def get_data(mmxx, mmyy, mmzz):

    df = pd.DataFrame(data=[v for v in zip(mmxx, mmyy, mmzz)], columns=['x', 'y', 'z'])
    return df

def plot_heatmap(heatmap, TI_heatmap_static):

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    #  cmap=plt.get_cmap('rainbow')
    # heatmap_static, heatmap_dynamic
    surf = ax.plot_surface(heatmap['x_bins'], heatmap['y_bins'], heatmap['heatmap_dynamic'])
    ax.view_init(elev=90, azim=-90)
    plt.savefig('./heatmap.png')
    plt.show()


def imshow_heatmap(heatmap):
    fig, ax = plt.subplots()
    im = ax.imshow(heatmap, interpolation='gaussian')
    plt.axis('off')
    # plt.savefig('/home/zhangq/Desktop/zhangq/UsefulCode/TEST.png', bbox_inches='tight', pad_inches=-0.1)


def test():


    # base_path = '/media/ourDataset/v1.0_label/20211027_1_group0021_123frames_25labeled'
    base_path = './Dataset/20211027_1_group0000'
    # base_path = '/media/ourDataset/v1.0_label/20211027_1_group0026_144frames_29labeled'
    # base_path = '/media/ourDataset/v1.0_label/20211027_1_group0028_134frames_27labeled'
    # base_path = '/home/zhangq/Desktop/zhangq/heatmap_check/data_test/heatmap_example'
    # base_path = '/media/ourDataset/v1.0_label/20211027_1_group0010_148frames_30labeled'
    # base_path = './20211027_2_group0051_72frames_15labeled'
    save_path = './save/test/'

    folders = os.listdir(base_path)
    folders = sorted(folders)
    frame_num = -1
    for folder in folders:
        frame_num += 1
        # if frame_num > 1:
        #     break
        print('frame_num:', frame_num)

        camera_path = os.path.join(base_path, folder, 'LeopardCamera1')
        for file in os.listdir(camera_path):
            if file[-3:] == 'png':
                img_path = os.path.join(camera_path, file)
        lidar_path = os.path.join(base_path, folder, 'VelodyneLidar')
        for file in os.listdir(lidar_path):
            if file[-3:] == 'pcd':
                Lidar_pcd_path = os.path.join(lidar_path, file)
            if file[-4:] == 'json':
                Lidar_json_path = os.path.join(lidar_path, file)
        ti_path = os.path.join(base_path, folder, 'TIRadar')
        for file in os.listdir(ti_path):
            if file[-3:] == 'pcd':
                TI_pcd_path = os.path.join(ti_path, file)
            if file[-4:] == 'json':
                TI_radar_json_path = os.path.join(ti_path, file)
            if file[-11:] == 'heatmap.bin':
                TI_heatmap_path = os.path.join(ti_path, file)

        # img = cv.imread(img_path)
        # width, height = img.shape[1], img.shape[0]


        TI_heatmap_data = load_TIRadarHeatmap(TI_heatmap_path)
        TI_heatmap_static = TI_heatmap_data['heatmap_static']
        TI_heatmap_dynamic = TI_heatmap_data['heatmap_dynamic']
        X_bins = TI_heatmap_data['x_bins']
        Y_bins = TI_heatmap_data['y_bins']
        # angel = np.arctan2(X_bins, Y_bins) / (np.pi/2)
        # distance = np.sqrt(X_bins**2 + Y_bins**2)
        # print('X_bins:', X_bins.shape)
        # print('Y_bins:', Y_bins.shape)


        # TEST = X_bins[128:, :]
        TI_heatmap_dynamic = np.vstack((TI_heatmap_dynamic[128:, :], TI_heatmap_dynamic[0:128, :]))
        TI_heatmap_dynamic = TI_heatmap_dynamic.T
        # TI_heatmap_dynamic = TI_heatmap_dynamic[0:128, 128-64:128+64]
        TI_heatmap_dynamic = TI_heatmap_dynamic[::-1, :]
        TI_heatmap_dynamic = TI_heatmap_dynamic/np.max(TI_heatmap_dynamic)



        TI_heatmap_static = np.vstack((TI_heatmap_static[128:, :], TI_heatmap_static[0:128, :]))
        TI_heatmap_static = TI_heatmap_static.T
        # TI_heatmap_static = TI_heatmap_static[0:128, 128 - 64:128 + 64]
        TI_heatmap_static = TI_heatmap_static[::-1, :]
        TI_heatmap_static = TI_heatmap_static / np.max(TI_heatmap_static)

        # print('TI_heatmap_dynamic:', TI_heatmap_dynamic.shape)
        # print('TI_heatmap_static:', TI_heatmap_static.shape)

        # plt.pcolor(X_bins, Y_bins, TI_heatmap_dynamic)
        imshow_heatmap(TI_heatmap_dynamic)
        plt.show()

        # neighborhood_size = 5
        # threshold = 0.4
        # TI_heatmap_dynamic[TI_heatmap_dynamic < threshold] = 0.
        # dynamic_max = filters.maximum_filter(TI_heatmap_dynamic, neighborhood_size)
        # maxima = (TI_heatmap_dynamic == dynamic_max)
        # maxima[TI_heatmap_dynamic == 0.] = False
        # print('where:', np.where(maxima==True))
        #
        # labeled, num_objects = ndimage.label(maxima)
        #
        # slices = ndimage.find_objects(labeled)
        # x, y = [], []
        # for dx, dy in slices:
        #     x_center = (dx.start + dx.stop - 1) / 2
        #     x.append(x_center)
        #     y_center = (dy.start + dy.stop - 1) / 2
        #     y.append(y_center)
        # # print('slices:', slices)
        # x = np.array(x).astype(np.int32)
        # y = np.array(y).astype(np.int32)
        # print('TI_heatmap_dynamic_peak:', TI_heatmap_dynamic[x ,y])
        # print('x:', x, 'y:', y)


        # print(Y_bins.shape)
        # ax = plt.subplot(111)
        # plt.xlim(np.min(X_bins), np.max(X_bins))
        # plt.ylim(np.min(Y_bins), np.max(Y_bins))
        # TI_heatmap_dynamic = TI_heatmap_dynamic / np.max(TI_heatmap_dynamic)
        # # ax.fill_between()
        #
        # plt.show()
        # print('X_bins:', X_bins)
        # print('Y_bins:', Y_bins)
        # print('angel:', angel)
        # print('distance:', distance)
        # plot_heatmap(TI_heatmap_data, TI_heatmap_static)

        # img = cv.resize(img, (1080, 720))
        # cv.imshow('test-img', img)
        if cv.waitKey(0) & 0xFF == 27:
            break


    print('test done.')


if __name__ == "__main__":
    test()


