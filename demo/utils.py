import json
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d

def load_json(json_path):
    '''
    read json
    :param json_path: str-json path
    :return: dict-json content
    '''
    with open(json_path, "r") as f:
        data = json.load(f)
    return data

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

def transform_xyz_to_uv(xyz, transform_matrix):
    '''

    :param xyz: (n, 3)
    :param transform_matrix: (3, 4)
    :return:
    '''
    n_points = xyz.shape[0]
    UVZ = np.matmul(transform_matrix,
                                np.vstack((xyz.T, np.ones((1, n_points)))))
    uv1 = UVZ / UVZ[2, :]
    uv = uv1[0:2, :].T
    uv = np.round(uv).astype('int')

    return uv

def rectangular_to_polar(xyz):
    '''
        transform pointcloud from rectangular coordinate to polar coordinate
    :param xyz: (x, y, z)
    :return aer:  (azimuth, elevation, range)
    '''
    x = xyz[:, 0].reshape((-1, 1))
    y = xyz[:, 1].reshape((-1, 1))
    z = xyz[:, 2].reshape((-1, 1))

    r = np.sqrt(x * x + y * y + z * z)
    azimuth = np.arctan(x / y) / np.pi * 180
    elevation = np.arctan(z / np.sqrt(x * x + y * y)) / np.pi * 180
    aer = np.hstack((azimuth, elevation, r))

    return aer

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

def show_pointcloud(xyz, text, elev_init=90, azim_init=-90):
    '''
        show pointcloud using plt
    :param xyz: (n, 3[x, y, z])
    :return:
    '''
    fig = plt.figure(text)
    ax = axes3d.Axes3D(fig)

    ax.scatter3D(xyz[:, 0], xyz[:, 1], xyz[:, 2],
                 c=xyz[:, 2], s=1)

    x_scale = xyz[:, 0].max() - xyz[:, 0].min()
    y_scale = xyz[:, 1].max() - xyz[:, 1].min()
    z_scale = xyz[:, 2].max() - xyz[:, 2].min()
    plt.gca().set_box_aspect((x_scale, y_scale, z_scale))
    plt.xlabel('x')
    plt.ylabel('y')
    ax.view_init(elev=elev_init, azim=azim_init)


def draw_line(corners, corner1_id, corner2_id):
    ax = plt.gca()
    ax.plot3D(
        np.array([corners[0, corner1_id], corners[0, corner2_id]]),
        np.array([corners[1, corner1_id], corners[1, corner2_id]]),
        np.array([corners[2, corner1_id], corners[2, corner2_id]]),
        'red'
    )

def draw_bbox(corners):
    draw_line(corners, 0, 1)
    draw_line(corners, 1, 2)
    draw_line(corners, 2, 3)
    draw_line(corners, 3, 0)

    draw_line(corners, 0, 4)
    draw_line(corners, 1, 5)
    draw_line(corners, 2, 6)
    draw_line(corners, 3, 7)

    draw_line(corners, 4, 5)
    draw_line(corners, 5, 6)
    draw_line(corners, 6, 7)
    draw_line(corners, 7, 4)

def show_pointcloud_with_boxes(xyz, text, bboxes, elev_init=90, azim_init=-90):
    '''
        show pointcloud with 3d bboxes
    :param xyz: (n, 3(x,y,z))
    :param text:
    :param bboxes: list [[x,y,z,l,w,h,alpha], [], ...]
    :param elev_init:
    :param azim_init:
    :return:
    '''
    show_pointcloud(xyz, text, elev_init=elev_init, azim_init=azim_init)

    for i in range(len(bboxes)):
        x, y, z, l, w, h, alpha = bboxes[i]
        corners = np.array([
            [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2],
            [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2],
            [h / 2, h / 2, h / 2, h / 2, -h / 2, -h / 2, -h / 2, -h / 2]
        ])
        rotation_matrix = np.array([
            [np.cos(alpha), -np.sin(alpha), 0],
            [np.sin(alpha), np.cos(alpha), 0],
            [0, 0, 1]
        ])
        corners_rotated = np.matmul(rotation_matrix, corners)
        corners_rotated_translated = corners_rotated + np.array([[x], [y], [z]])
        draw_bbox(corners_rotated_translated)

def get_pcd_in_bboxes(pcd, bboxes):
    '''
        get pointcloud in bboxes
    :param pcd: (n, 3(x,y,z))
    :param bboxes: list [[x,y,z,l,w,h,alpha], [], ...]
    :return pts_in_boxes: (n, 3(x,y,z))
    :return pts_object: list [(n, 3(x,y,z)), ...]
    '''
    bboxes = np.array(bboxes)
    n_bboxes = bboxes.shape[0]

    pts = pcd[:, :3]
    pts = pts.T
    pts = np.expand_dims(pts, axis=0)
    pts = pts.repeat(n_bboxes, axis=0)
    pts_shift = pts - np.expand_dims(bboxes[:, :3], axis=2)
    rotation_matrixes = np.array(
        [[[np.cos(-alpha), -np.sin(-alpha), 0], [np.sin(-alpha), np.cos(-alpha), 0], [0, 0, 1]] for alpha in
         bboxes[:, 6]]
    )
    pts_shift_rotated = np.zeros(pts_shift.shape)
    bbox_indexes = np.arange(n_bboxes)
    pts_shift_rotated[bbox_indexes, :, :] = np.matmul(
        rotation_matrixes[bbox_indexes, :, :],
        pts_shift[bbox_indexes, :, :]
    )

    mask_x = np.abs(pts_shift_rotated[:, 0, :]) <= np.expand_dims(bboxes[:, 3] / 2, axis=1)
    mask_y = np.abs(pts_shift_rotated[:, 1, :]) <= np.expand_dims(bboxes[:, 4] / 2, axis=1)
    mask_z = np.abs(pts_shift_rotated[:, 2, :]) <= np.expand_dims(bboxes[:, 5] / 2, axis=1)
    mask = np.logical_and(
        mask_x,
        np.logical_and(mask_y, mask_z)
    )
    pts_object = [pcd[mask[i, :], :] for i in range(n_bboxes)]
    pts_in_boxes = pcd[mask.sum(axis=0) == 1, :]

    return pts_in_boxes, pts_object

