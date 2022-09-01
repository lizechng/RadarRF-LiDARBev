import os
import numpy as np
import math
import struct
import glob
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d

def pointcloud_transform(pointcloud, transform_matrix):
    '''
        transform pointcloud from coordinate1 to coordinate2 according to transform_matrix
    :param pointcloud: (x, y, z, ...)
    :param transform_matrix:
    :return pointcloud_transformed: (x, y, z, ...)
    '''
    n_points = pointcloud.shape[0]
    xyz = pointcloud[:, :3]
    xyz1 = np.vstack((xyz.T, np.ones((1, n_points))))
    xyz1_transformed = np.matmul(transform_matrix, xyz1)
    pointcloud_transformed = np.hstack((
        xyz1_transformed[:3, :].T,
        pointcloud[:, 3:]
    ))
    return pointcloud_transformed

def load_VelodyneLidarPcd(pcd_path):
    '''
    read velodyne lidar pcd
    :param pcd_path: str-velodyne lidar pcd path
    :return: np.array-(n, 6) x,y,z,intensity,ring,time
    '''
    with open(pcd_path, "rb") as f:
        f.seek(212, 0)
        data = f.read()
    num_points = math.floor(len(data) / 22)
    data_valid_len = num_points * 22
    data = struct.unpack('<' + 'ffffhf' * num_points, data[:data_valid_len])
    data = np.array(data).reshape((-1, 6))
    return data


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

def main():
    # lidar_path = glob.glob(os.path.join('./data_test/lizc', '*.pcd'))[0]
    lidar_path = './Dataset/20211027_1_group0000/group0000_frame0000/VelodyneLidar/1635317963.762042046.pcd'
    # read lidar_pcd
    pointcloud = load_VelodyneLidarPcd(lidar_path)

    VelodyneLidar_to_LeopardCamera1_TransformMatrix = np.array(
        [
            [1878.13694567694, -1900.51229133004, -3.76435830142225, -1450.90535286182],
            [962.699273002558, 8.21051460132727, -1996.69155762544, -1289.34979520961],
            [0.998966844122018, 0.0410508475655525, -0.0194954420069267, -0.515288952567310]
        ]
    )
    TIRadar_to_LeopardCamera1_TransformMatrix = np.array(
        [
            [2019.61363529422, 1745.88166828988, -111.433796801951, -419.938881768377],
            [26.0193673714885, 870.796981112031, -2038.30078479358, -120.997110351106],
            [0.0244308479903333, 0.997614077965117, -0.0645700016438225, -0.00641535834610336]
        ]
    )
    LeopardCamera1_IntrinsicMatrix = np.array(
        [
            [1976.27129878769, 0, 1798.25228491297],
            [0, 1977.80114435384, 1000.96808764067],
            [0, 0, 1]
        ]
    )
    VelodyneLidar_to_TIRadar_TransformMatrix = np.matmul(
        np.linalg.inv(
            np.vstack((np.matmul(np.linalg.inv(LeopardCamera1_IntrinsicMatrix),
                                 TIRadar_to_LeopardCamera1_TransformMatrix), np.array([[0, 0, 0, 1]])))
        ),
        np.vstack((np.matmul(np.linalg.inv(LeopardCamera1_IntrinsicMatrix),
                             VelodyneLidar_to_LeopardCamera1_TransformMatrix), np.array([[0, 0, 0, 1]])))
    )

    pointcloud_in_TIRadar_coordinate = pointcloud_transform(pointcloud, VelodyneLidar_to_TIRadar_TransformMatrix)

    # interest zone
    x_zone = [-80, 80]
    y_zone = [0, 80]
    z_zone = [-np.inf, np.inf]
    mask = np.logical_and(
        pointcloud_in_TIRadar_coordinate[:, 0] >= x_zone[0],
        pointcloud_in_TIRadar_coordinate[:, 0] <= x_zone[1]
    )
    mask = np.logical_and(
        mask,
        np.logical_and(
            pointcloud_in_TIRadar_coordinate[:, 1] >= y_zone[0],
            pointcloud_in_TIRadar_coordinate[:, 1] <= y_zone[1]
        )
    )
    mask = np.logical_and(
        mask,
        np.logical_and(
            pointcloud_in_TIRadar_coordinate[:, 2] >= z_zone[0],
            pointcloud_in_TIRadar_coordinate[:, 2] <= z_zone[1]
        )
    )
    lidar_xyz = pointcloud_in_TIRadar_coordinate[mask]

    # read heatmap
    # heatmap_path = glob.glob(os.path.join('./data_test/lizc', '*.heatmap.bin'))[0]
    heatmap_path = './Dataset/20211027_1_group0000/group0000_frame0000/TIRadar/1635317963.771.heatmap.bin'
    heatmap = load_TIRadarHeatmap(heatmap_path)

    heatmap_static = np.stack((heatmap['x_bins'], heatmap['y_bins'], heatmap['heatmap_static'])).reshape((3, -1)).T
    heatmap_dynamic = np.stack((heatmap['x_bins'], heatmap['y_bins'], heatmap['heatmap_dynamic'])).reshape((3, -1)).T
    
    fig1 = plt.figure('heatmap_static')
    fig1_ax1 = axes3d.Axes3D(fig1)
    fig1_ax1.scatter3D(lidar_xyz[:, 0], lidar_xyz[:, 1], lidar_xyz[:, 2], c='k', s=1)
    fig1_ax1.scatter3D(heatmap_static[:, 0], heatmap_static[:, 1], lidar_xyz[:, 2].min(), c=heatmap_static[:, 2], cmap='jet')
    x_scale = lidar_xyz[:, 0].max() - lidar_xyz[:, 0].min()
    y_scale = lidar_xyz[:, 1].max() - lidar_xyz[:, 1].min()
    z_scale = lidar_xyz[:, 2].max() - lidar_xyz[:, 2].min()
    plt.gca().set_box_aspect((x_scale, y_scale, z_scale))
    plt.xlabel('x')
    plt.ylabel('y')
    fig1_ax1.view_init(elev=74, azim=-90)

    fig2 = plt.figure('heatmap_dynamic')
    fig2_ax1 = axes3d.Axes3D(fig2)
    fig2_ax1.scatter3D(lidar_xyz[:, 0], lidar_xyz[:, 1], lidar_xyz[:, 2], c='k', s=1)
    fig2_ax1.scatter3D(heatmap_dynamic[:, 0], heatmap_dynamic[:, 1], lidar_xyz[:, 2].min(), c=heatmap_dynamic[:, 2], cmap='jet')
    x_scale = lidar_xyz[:, 0].max() - lidar_xyz[:, 0].min()
    y_scale = lidar_xyz[:, 1].max() - lidar_xyz[:, 1].min()
    z_scale = lidar_xyz[:, 2].max() - lidar_xyz[:, 2].min()
    plt.gca().set_box_aspect((x_scale, y_scale, z_scale))
    plt.xlabel('x')
    plt.ylabel('y')
    fig2_ax1.view_init(elev=74, azim=-90)
    
    plt.show()
    

    print('done')

if __name__ == '__main__':
    main()
