import os

import numpy as np
from torch.utils.data import Dataset

class self_dataset(Dataset):
    def __init__(self, dataset_root):
        self.dataset_root = dataset_root

        self.radar_data_root = os.path.join(dataset_root, 'radar')
        self.radar_data = os.listdir(self.radar_data_root)
        self.radar_data.sort()
        self.radar_data_pathes = [os.path.join(self.radar_data_root, frame) for frame in self.radar_data]

        self.lidar_data_root = os.path.join(dataset_root, 'lidar')
        self.lidar_data = os.listdir(self.lidar_data_root)
        self.lidar_data.sort()
        self.lidar_data_pathes = [os.path.join(self.lidar_data_root, frame) for frame in self.lidar_data]

        assert len(self.radar_data_pathes) == len(self.lidar_data_pathes)
        self.length = len(self.radar_data)

        self.common_info = dict(np.load(os.path.join(self.dataset_root, 'common_info.npz')))

    def __len__(self):
        return self.length

    def __getitem__(self, item):
        radar_data = np.load(self.radar_data_pathes[item])  # (range, doppler, azimuth, elevation)
        radar_data = np.swapaxes(radar_data, 2, 3)  # (range, doppler, elevation, azimuth)
        radar_data = np.swapaxes(radar_data, 0, 1)  # (doppler, range, elevation, azimuth)
        # radar_data_v_zero = radar_data[radar_data.shape[0] // 2, :, :, :]
        # mask_v_nonzero = (np.arange(radar_data.shape[0]) != radar_data.shape[0] / 2)
        # radar_data_v_nonzero = radar_data[mask_v_nonzero, :, :, :].sum(axis=0)
        # radar_data = np.stack((radar_data_v_zero, radar_data_v_nonzero), axis=0)

        lidar_data = np.load(self.lidar_data_pathes[item])  # (elevation, azimuth, channel)
        lidar_data = np.swapaxes(lidar_data, 0, 2)
        lidar_data = np.swapaxes(lidar_data, 1, 2)  # (channel, elevation, azimuth)

        return radar_data, lidar_data


if __name__ == '__main__':
    dataset_train = self_dataset('../data/mmwave_augmentation/train')

    radar_data, lidar_data = dataset_train[0]  # ego dynamic

    import matplotlib.pyplot as plt
    img = np.stack((lidar_data[3, :, :], lidar_data[4, :, :], lidar_data[5, :, :]))
    img = np.swapaxes(img, 0, 2)
    img = np.swapaxes(img, 0, 1)
    fig1 = plt.figure()
    ax1 = fig1.add_axes([0.1, 0.1, 0.8, 0.8])
    ax1.imshow(img)

    heatmap = radar_data.sum(axis=0).sum(axis=1)
    r = np.tile(dataset_train.common_info['radar_range_bins'], (heatmap.shape[1], 1)).T
    azimuth = np.tile(dataset_train.common_info['radar_azimuth_bins'], (heatmap.shape[0], 1))
    x = r * np.sin(azimuth / 180 * np.pi)
    y = r * np.cos(azimuth / 180 * np.pi)

    fig2 = plt.figure()
    fig2_ax1 = fig2.add_subplot(projection='3d')
    fig2_ax1.plot_surface(x, y, heatmap, cmap='jet')
    fig2_ax1.view_init(elev=90, azim=-90)


    def polar_to_rectangular(aer):
        '''
            transform pointcloud from polar coordinate to rectangular coordinate
        :param aer: (azimuth, elevation, range)
        :return xyz: (x, y, z)
        '''
        if aer.shape[0] == 0:
            return np.zeros((0, 3))

        azimuth = aer[:, 0].reshape((-1, 1))
        elevation = aer[:, 1].reshape((-1, 1))
        r = aer[:, 2].reshape((-1, 1))

        x = r * np.cos(elevation / 180 * np.pi) * np.sin(azimuth / 180 * np.pi)
        y = r * np.cos(elevation / 180 * np.pi) * np.cos(azimuth / 180 * np.pi)
        z = r * np.sin(elevation / 180 * np.pi)
        xyz = np.hstack((x, y, z))

        return xyz

    lidar_aer = lidar_data[:3, :, :].reshape((3, -1)).T
    lidar_xyz = polar_to_rectangular(lidar_aer)

    from mpl_toolkits.mplot3d import axes3d

    def show_pointcloud(xyz):
        '''
            show pointcloud using plt
        :param xyz: (n, 3[x, y, z])
        :return:
        '''
        fig = plt.figure()
        ax = axes3d.Axes3D(fig)

        ax.scatter3D(xyz[:, 0], xyz[:, 1], xyz[:, 2],
                     c=xyz[:, 2], s=1)

        x_scale = xyz[:, 0].max() - xyz[:, 0].min()
        y_scale = xyz[:, 1].max() - xyz[:, 1].min()
        z_scale = xyz[:, 2].max() - xyz[:, 2].min()
        plt.gca().set_box_aspect((x_scale, y_scale, z_scale))
        plt.xlabel('x')
        plt.ylabel('y')

    # show lidar pointcloud
    show_pointcloud(lidar_xyz)

    # show lidar pointcloud with heatmap
    fig4 = plt.figure()
    fig4_ax1 = axes3d.Axes3D(fig4)
    fig4_ax1.scatter3D(lidar_xyz[:, 0], lidar_xyz[:, 1], lidar_xyz[:, 2], c='k', s=1)
    fig4_ax1.scatter3D(x, y, lidar_xyz[:, 2].min(), c=heatmap, cmap='jet')
    x_scale = lidar_xyz[:, 0].max() - lidar_xyz[:, 0].min()
    y_scale = lidar_xyz[:, 1].max() - lidar_xyz[:, 1].min()
    z_scale = lidar_xyz[:, 2].max() - lidar_xyz[:, 2].min()
    plt.gca().set_box_aspect((x_scale, y_scale, z_scale))
    plt.xlabel('x')
    plt.ylabel('y')
    fig4_ax1.view_init(elev=74, azim=-90)

    # show Range-Doppler Curves
    sig_integrate = radar_data.sum(axis=-1).sum(axis=-1)
    fig5 = plt.figure()
    fig5_ax1 = fig5.add_axes([0.1, 0.1, 0.8, 0.8])
    for i in range(radar_data.shape[0]):
        if i == radar_data.shape[0] / 2:
            fig5_ax1.plot(dataset_train.common_info['radar_range_bins'], sig_integrate[i, :], color='black', linewidth=2,
                          label='v={:.2f}'.format(dataset_train.common_info['radar_velocity_bins'][i]))
        else:
            fig5_ax1.plot(dataset_train.common_info['radar_range_bins'], sig_integrate[i, :], linewidth=1,
                          label='v={:.2f}'.format(dataset_train.common_info['radar_velocity_bins'][i]))
    fig5_ax1.grid(True)
    fig5_ax1.set_xlabel('range(m)')
    fig5_ax1.set_ylabel('energy')
    plt.title('Range-Doppler Curves')

    sig_integrate_db = 20 * np.log10(radar_data.sum(axis=-1).sum(axis=-1))
    fig6 = plt.figure()
    fig6_ax1 = fig6.add_axes([0.1, 0.1, 0.8, 0.8])
    for i in range(radar_data.shape[0]):
        if i == radar_data.shape[0] / 2:
            fig6_ax1.plot(dataset_train.common_info['radar_range_bins'], sig_integrate_db[i, :], color='black', linewidth=2,
                          label='v={:.2f}'.format(dataset_train.common_info['radar_velocity_bins'][i]))
        else:
            fig6_ax1.plot(dataset_train.common_info['radar_range_bins'], sig_integrate_db[i, :], linewidth=1,
                          label='v={:.2f}'.format(dataset_train.common_info['radar_velocity_bins'][i]))
    fig6_ax1.grid(True)
    fig6_ax1.set_xlabel('range(m)')
    fig6_ax1.set_ylabel('energy(dB)')
    plt.title('Range-Doppler Curves')

    sig_integrate_v_zero = sig_integrate[radar_data.shape[0] // 2, :]
    mask = (np.arange(radar_data.shape[0]) != radar_data.shape[0] / 2)
    sig_integrate_v_nonzero = sig_integrate[mask, :].sum(axis=0)
    fig7 = plt.figure()
    fig7_ax1 = fig7.add_axes([0.1, 0.1, 0.8, 0.8])
    fig7_ax1.plot(dataset_train.common_info['radar_range_bins'], sig_integrate_v_zero, color='black', linewidth=2)
    fig7_ax1.plot(dataset_train.common_info['radar_range_bins'], sig_integrate_v_nonzero, linewidth=1)
    fig7_ax1.grid(True)
    fig7_ax1.set_xlabel('range(m)')
    fig7_ax1.set_ylabel('energy(dB)')
    plt.title('Range-Doppler Curves')

    # show lidar pointcloud with heatmap_v_zero / heatmap_v_nonzero
    heatmap_v_zero = radar_data[radar_data.shape[0] // 2, :, :, :].sum(axis=1)
    mask = (np.arange(radar_data.shape[0]) != radar_data.shape[0] / 2)
    heatmap_v_nonzero = radar_data[mask, :, :, :].sum(axis=0).sum(axis=1)
    r = np.tile(dataset_train.common_info['radar_range_bins'], (heatmap_v_zero.shape[1], 1)).T
    azimuth = np.tile(dataset_train.common_info['radar_azimuth_bins'], (heatmap_v_zero.shape[0], 1))
    x = r * np.sin(azimuth / 180 * np.pi)
    y = r * np.cos(azimuth / 180 * np.pi)

    fig8 = plt.figure('heatmap_v_zero')
    fig8_ax1 = axes3d.Axes3D(fig8)
    fig8_ax1.scatter3D(lidar_xyz[:, 0], lidar_xyz[:, 1], lidar_xyz[:, 2], c='k', s=1)
    fig8_ax1.scatter3D(x, y, lidar_xyz[:, 2].min(), c=heatmap_v_zero, cmap='jet')
    x_scale = lidar_xyz[:, 0].max() - lidar_xyz[:, 0].min()
    y_scale = lidar_xyz[:, 1].max() - lidar_xyz[:, 1].min()
    z_scale = lidar_xyz[:, 2].max() - lidar_xyz[:, 2].min()
    plt.gca().set_box_aspect((x_scale, y_scale, z_scale))
    plt.xlabel('x')
    plt.ylabel('y')
    fig8_ax1.view_init(elev=74, azim=-90)

    fig9 = plt.figure('heatmap_v_nonzero')
    fig9_ax1 = axes3d.Axes3D(fig9)
    fig9_ax1.scatter3D(lidar_xyz[:, 0], lidar_xyz[:, 1], lidar_xyz[:, 2], c='k', s=1)
    fig9_ax1.scatter3D(x, y, lidar_xyz[:, 2].min(), c=heatmap_v_nonzero, cmap='jet')
    x_scale = lidar_xyz[:, 0].max() - lidar_xyz[:, 0].min()
    y_scale = lidar_xyz[:, 1].max() - lidar_xyz[:, 1].min()
    z_scale = lidar_xyz[:, 2].max() - lidar_xyz[:, 2].min()
    plt.gca().set_box_aspect((x_scale, y_scale, z_scale))
    plt.xlabel('x')
    plt.ylabel('y')
    fig9_ax1.view_init(elev=74, azim=-90)


    plt.show()



    print('done')

