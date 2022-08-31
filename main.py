# Semantic-guided depth completion
import numpy as np
import cv2
import json
from PIL import Image
import os
import matplotlib.pyplot as plt
import re
import warnings

numpy_pcd_type_mappings = [(np.dtype('float32'), ('F', 4)),
                           (np.dtype('float64'), ('F', 8)),
                           (np.dtype('uint8'), ('U', 1)),
                           (np.dtype('uint16'), ('U', 2)),
                           (np.dtype('uint32'), ('U', 4)),
                           (np.dtype('uint64'), ('U', 8)),
                           (np.dtype('int16'), ('I', 2)),
                           (np.dtype('int32'), ('I', 4)),
                           (np.dtype('int64'), ('I', 8))]
pcd_type_to_numpy_type = dict((q, p) for (p, q) in numpy_pcd_type_mappings)


def parse_header(lines):
    '''Parse header of PCD files'''
    metadata = {}
    for ln in lines:
        if ln.startswith('#') or len(ln) < 2:
            continue
        match = re.match('(\w+)\s+([\w\s\.]+)', ln)
        if not match:
            warnings.warn(f'warning: cannot understand line: {ln}')
            continue
        key, value = match.group(1).lower(), match.group(2)
        if key == 'version':
            metadata[key] = value
        elif key in ('fields', 'type'):
            metadata[key] = value.split()
        elif key in ('size', 'count'):
            metadata[key] = map(int, value.split())
        elif key in ('width', 'height', 'points'):
            metadata[key] = int(value)
        elif key == 'viewpoint':
            metadata[key] = map(float, value.split())
        elif key == 'data':
            metadata[key] = value.strip().lower()

    if 'count' not in metadata:
        metadata['count'] = [1] * len(metadata['fields'])
    if 'viewpoint' not in metadata:
        metadata['viewpoint'] = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
    if 'version' not in metadata:
        metadata['version'] = '.7'
    return metadata


def _build_dtype(metadata):
    fieldnames = []
    typenames = []
    for f, c, t, s in zip(metadata['fields'],
                          metadata['count'],
                          metadata['type'],
                          metadata['size']):
        np_type = pcd_type_to_numpy_type[(t, s)]
        if c == 1:
            fieldnames.append(f)
            typenames.append(np_type)
        else:
            fieldnames.extend(['%s_%04d' % (f, i) for i in range(c)])
            typenames.extend([np_type] * c)
    dtype = np.dtype(list(zip(fieldnames, typenames)))
    return dtype


def parse_ascii_pc_data(f, dtype, metadata):
    # for radar point
    return np.loadtxt(f, dtype=dtype, delimiter=' ')


def parse_binary_pc_data(f, dtype, metadata):
    # for lidar point
    rowstep = metadata['points'] * dtype.itemsize
    buf = f.read(rowstep)
    return np.fromstring(buf, dtype=dtype)


def parse_binary_compressed_pc_data(f, dtype, metadata):
    raise NotImplemented


def read_pcd(pcd_path, pts_view=False):
    # pcd = o3d.io.read_point_cloud(pcd_path)
    f = open(pcd_path, 'rb')
    header = []
    while True:
        ln = f.readline().strip()  # ln is bytes
        ln = str(ln, encoding='utf-8')
        header.append(ln)
        # print(type(ln), ln)
        if ln.startswith('DATA'):
            metadata = parse_header(header)
            dtype = _build_dtype(metadata)
            break
    if metadata['data'] == 'ascii':
        pc_data = parse_ascii_pc_data(f, dtype, metadata)
    elif metadata['data'] == 'binary':
        pc_data = parse_binary_pc_data(f, dtype, metadata)
    elif metadata['data'] == 'binary_compressed':
        pc_data = parse_binary_compressed_pc_data(f, dtype, metadata)
    else:
        print('DATA field is neither "ascii" or "binary" or "binary_compressed"')

    points = np.concatenate([pc_data[metadata['fields'][0]][:, None],
                             pc_data[metadata['fields'][1]][:, None],
                             pc_data[metadata['fields'][2]][:, None],
                             pc_data[metadata['fields'][3]][:, None]], axis=-1)
    print(f'pcd points: {points.shape}')

    if pts_view:
        ptsview(points)
    return points


def pts2camera(pts, img_path, calib_path, matrix=None):
    depth_img = np.zeros((704, 2000, 3), dtype=np.uint8)
    img = Image.open(img_path)
    print(f'img: {img.size}')
    width, height = img.size
    if matrix is None:
        try:
            matrix = json.load(open(calib_path))['VelodyneLidar_to_LeopardCamera1_TransformMatrix']
        except:
            matrix = json.load(open(calib_path))['OCULiiRadar_to_LeopardCamera1_TransformMatrix']
    matrix = np.asarray(matrix)
    from scipy.linalg import pinv
    inv = pinv(matrix)
    n = pts.shape[0]
    pts = np.hstack((pts, np.ones((n, 1))))
    print(pts.shape, matrix.shape)
    pts_2d = np.dot(pts, np.transpose(matrix))
    print('pts_2d\n', pts_2d)
    pts_2d[:, 0] = pts_2d[:, 0] / pts_2d[:, 2]
    pts_2d[:, 1] = pts_2d[:, 1] / pts_2d[:, 2]
    mask = (pts_2d[:, 0] < 2750) & (pts_2d[:, 1] < 1204) & \
           (pts_2d[:, 0] > 750) & (pts_2d[:, 1] > 500) & \
           (pts_2d[:, 2] > 5) & (pts_2d[:, 2] <= 255)
    pts_2d = pts_2d[mask, :]
    pts_2d[:, 0] -= 750
    pts_2d[:, 1] -= 500
    # pts_2d = pts_2d[mask, :]
    cmap = plt.cm.get_cmap('hsv', 256)
    cmap = np.array([cmap(i) for i in range(256)])[:, :3] * 255
    img = np.asarray(img)
    image = np.asarray(img)[500:1204, 750:2750, :]
    for i in range(pts_2d.shape[0]):
        depth = pts_2d[i, 2]
        color = cmap[int(depth), :]
        cv2.circle(depth_img, (int(np.round(pts_2d[i, 0])),
                               int(np.round(pts_2d[i, 1]))),
                   2, color=tuple([int(depth), int(depth), int(depth)]), thickness=-1)
    depth_img = depth_img[::2, ::2, 0]
    image_img = image[::2, ::2]
    print(depth_img.shape, image_img.shape)
    print(matrix)
    print(inv)
    print(np.dot(matrix, inv))
    pc = []
    for i in range(depth_img.shape[0]):
        for j in range(depth_img.shape[1]):
            v = (i * 2 + 500) * depth_img[i, j]
            u = (j * 2 + 750) * depth_img[i, j]
            p = np.dot(inv, np.asarray([u, v, depth_img[i, j]]).reshape(3, 1))
            # if p[0, 0] !=0:
            #     print(p.reshape(4, ))
            pc.append(list(p.reshape(4, )))
    pc = np.asarray(pc)
    print(pc.shape)
    Image.fromarray(depth_img).show()
    Image.fromarray(image_img).show()


def pts2bev(pts):
    side_range = (-75, 75)
    fwd_range = (0, 75)
    height_range = (-2, 5)
    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
    # print(np.min(x), np.max(x)) # -199.20, 197.77
    # print(np.min(y), np.max(y)) # -185.53, 196.74
    # print(np.min(z), np.max(z)) # -5.02, 39.75
    f_filter = np.logical_and(x > fwd_range[0], x < fwd_range[1])
    s_filter = np.logical_and(y > side_range[0], y < side_range[1])
    h_filter = np.logical_and(z > height_range[0], z < height_range[1])
    filter = np.logical_and(f_filter, s_filter)
    filter = np.logical_and(filter, h_filter) # height filter
    indices = np.argwhere(filter).flatten()
    x, y, z = x[indices], y[indices], z[indices]

    res = 0.25
    x_img = (-y / res).astype(np.int32)
    y_img = (-x / res).astype(np.int32)
    x_img = x_img - int(np.floor(side_range[0]) / res)
    y_img = y_img + int(np.floor(fwd_range[1]) / res)


    # pixel_value = np.clip(a=z, a_max=height_range[1], a_min=height_range[0])
    pixel_value = 250 #z

    def scale_to_255(a, min, max, dtype=np.uint8):
        return ((a - min) / float(max - min) * 255).astype(dtype)

    # pixel_value = scale_to_255(pixel_value, height_range[0], height_range[1])
    # pixel_value = (pixel_value - np.min(pixel_value)) / (np.max(pixel_value) - np.min(pixel_value)) * 255
    x_max = int((side_range[1] - side_range[0]) / res) + 1
    y_max = int((fwd_range[1] - fwd_range[0]) / res) + 1

    im = np.zeros([y_max, x_max], dtype=np.uint8)
    im[y_img, x_img] = pixel_value
    # im = (im - np.min(im)) / (np.max(im) - np.min(im)) * 255
    print(im.shape)
    # plt.imshow(im, cmap='jet')
    # plt.show()
    return im[:, ::-1]


def lidar_radar_diff(img_path, lidar_pts, radar_pts, calib_lidar, calib_radar):
    img = Image.open(img_path)
    # img.show()
    width, height = img.size
    lidar_img = np.zeros((height, width, 3), dtype=np.uint8)
    lidar_value = np.zeros((width, height), dtype=np.float32)
    radar_img = np.zeros((height, width, 3), dtype=np.uint8)
    radar_value = np.zeros((width, height, 2), dtype=np.float32)

    lidar_img = np.asarray(img).copy()
    radar_img = np.asarray(img).copy()
    correct_img = np.asarray(img).copy()

    matrix_lidar = json.load(open(calib_lidar))['VelodyneLidar_to_LeopardCamera1_TransformMatrix']
    try:
        matrix_radar = json.load(open(calib_radar))['OCULiiRadar_to_LeopardCamera1_TransformMatrix']
    except:
        matrix_radar = json.load(open(calib_ti))['TIRadar_to_LeopardCamera1_TransformMatrix']
    matrix_lidar = np.asarray(matrix_lidar)
    matrix_radar = np.asarray(matrix_radar)

    cmap = plt.cm.get_cmap('hsv', 256)
    cmap = np.array([cmap(i) for i in range(256)])[:, :3] * 255

    n = lidar_pts.shape[0]
    lidar_pts = np.hstack((lidar_pts[:, :3], np.ones((n, 1))))
    lidar_2d = np.dot(lidar_pts, np.transpose(matrix_lidar))
    lidar_2d[:, 0] = lidar_2d[:, 0] / lidar_2d[:, 2]
    lidar_2d[:, 1] = lidar_2d[:, 1] / lidar_2d[:, 2]
    mask = (lidar_2d[:, 0] < width) & (lidar_2d[:, 1] < height) & \
           (lidar_2d[:, 0] > 0) & (lidar_2d[:, 1] > 0) & \
           (lidar_2d[:, 2] > 5) & (lidar_2d[:, 2] < 80)
    lidar_2d = lidar_2d[mask, :]

    for i in range(lidar_2d.shape[0]):
        depth = lidar_2d[i, 2]
        # depth = pts_2d[i,2]
        color = cmap[int(3 * depth), :]
        cv2.circle(lidar_img,
                   (int(np.round(lidar_2d[i, 0])), int(np.round(lidar_2d[i, 1]))),
                   1, color=tuple(color), thickness=-1)
        lidar_value[int(np.floor(lidar_2d[i, 0])), int(np.floor(lidar_2d[i, 1]))] = depth
    Image.fromarray(lidar_img).show()
    # Image.fromarray(lidar_value).show()

    n = radar_pts.shape[0]
    velo = radar_pts[:, 3:4]
    radar_pts = np.hstack((radar_pts[:, :3], np.ones((n, 1))))
    radar_2d = np.dot(radar_pts, np.transpose(matrix_radar))
    radar_2d[:, 0] = radar_2d[:, 0] / radar_2d[:, 2]
    radar_2d[:, 1] = radar_2d[:, 1] / radar_2d[:, 2]
    mask = (radar_2d[:, 0] < width) & (radar_2d[:, 1] < height) & \
           (radar_2d[:, 0] > 0) & (radar_2d[:, 1] > 0) & \
           (radar_2d[:, 2] > 5) & (radar_2d[:, 2] < 80)
    radar_2d = radar_2d[mask, :]
    velo = velo[mask, :]
    radar_2d = np.concatenate([radar_2d, velo], axis=-1)

    print(f'radar_2d number: {radar_2d.shape[0]}, lidar_2d number: {lidar_2d.shape[0]}')
    for i in range(radar_2d.shape[0]):
        depth = radar_2d[i, 2]
        velo = radar_2d[i, 3]
        color = cmap[int(3 * depth), :]
        cv2.circle(radar_img,
                   (int(np.round(radar_2d[i, 0])), int(np.round(radar_2d[i, 1]))),
                   3, color=tuple(color), thickness=-1)
        # print(int(np.round(radar_2d[i, 0])), int(np.round(radar_2d[i, 1])))
        radar_value[int(np.floor(radar_2d[i, 0])), int(np.floor(radar_2d[i, 1])), 0] = depth
        radar_value[int(np.floor(radar_2d[i, 0])), int(np.floor(radar_2d[i, 1])), 1] = velo
    Image.fromarray(radar_img).show()
    # print(radar_value.shape)
    radar_list, min_list, velo_list, max_list = [], [], [], []
    for u in range(radar_value.shape[0]):
        for v in range(radar_value.shape[1]):
            if radar_value[u, v, 0] != 0:
                # print(f'radar depth: {radar_value[u, v, 0]}, radar velo: {radar_value[u, v, 1]}')
                diff = []
                update_flag = False
                for i in range(u - 6, u + 6):
                    for j in range(v - 6, v + 6):
                        if i >= width or j >= height:
                            continue
                        if lidar_value[i, j] != 0:
                            diff.append((lidar_value[i, j] - radar_value[u, v, 0]))
                            if abs(lidar_value[i, j] - radar_value[u, v, 0]) <= radar_value[u, v, 1]:
                                update_flag = True
                            # print(lidar_value[i, j], end=' ')
                if update_flag:
                    color = cmap[int(3 * radar_value[u, v, 0]), :]
                    cv2.circle(correct_img,
                               (u, v),
                               3, color=tuple(color), thickness=-1)
                if len(diff) > 0:
                    # print(f'min: {min(diff)}, max: {max(diff)}, mean: {sum(diff)/len(diff)}')
                    min_list.append(radar_value[u, v, 0] + min(diff))
                    radar_list.append(radar_value[u, v, 0])
                    max_list.append(radar_value[u, v, 0] + max(diff))
                    velo_list.append(radar_value[u, v, 1])

    Image.fromarray(correct_img).show()

    idx = sorted(range(len(radar_list)), key=lambda k: radar_list[k])
    radar_list = np.asarray(radar_list)[idx]
    min_list = np.asarray(min_list)[idx]
    max_list = np.asarray(max_list)[idx]
    velo_list = np.asarray(velo_list)[idx]

    plt.plot(radar_list, radar_list + velo_list, '--', linewidth=1, color='sandybrown')
    plt.plot(radar_list, radar_list - velo_list, '--', linewidth=1, color='sandybrown')
    plt.fill_between(radar_list, radar_list, radar_list - velo_list, color='peachpuff')
    plt.fill_between(radar_list, radar_list, radar_list + velo_list, color='indianred')

    plt.plot(radar_list, radar_list, '--', linewidth=1, color='darkcyan')
    plt.scatter(radar_list, min_list, marker='v', c='darkcyan', s=24)
    plt.scatter(radar_list, max_list, marker='^', c='darkcyan', s=24)

    for i in range(len(radar_list)):
        plt.plot([radar_list[i], radar_list[i]], [min_list[i], max_list[i]], '-', linewidth=1, color='darkcyan')
        # plt.plot(radar_list[i], max_list[i], 'b-', linewidth=1)
    # print(radar_list[:5], lidar_list[:5], err_list[:5])
    # plt.errorbar(radar_list, lidar_list, yerr=err_list, label='None')
    plt.show()


def img_sequence(img_path):
    img = cv2.imread(img_path)
    cv2.namedWindow('img', cv2.WINDOW_KEEPRATIO)
    cv2.imshow('img', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def pts_sequence(img_path, pts, calib):
    img = Image.open(img_path)
    width, height = img.size

    depth_img = np.zeros((height, width, 3), np.uint8)

    calib_dict = json.load(open(calib))
    if 'VelodyneLidar_to_LeopardCamera1_TransformMatrix' in calib_dict.keys():
        matrix = calib_dict['VelodyneLidar_to_LeopardCamera1_TransformMatrix']
    if 'OCULiiRadar_to_LeopardCamera1_TransformMatrix' in calib_dict.keys():
        matrix = calib_dict['OCULiiRadar_to_LeopardCamera1_TransformMatrix']
    if 'TIRadar_to_LeopardCamera1_TransformMatrix' in calib_dict.keys():
        matrix = calib_dict['TIRadar_to_LeopardCamera1_TransformMatrix']

    matrix = np.asarray(matrix)

    cmap = plt.cm.get_cmap('hsv', 256)
    cmap = np.array([cmap(i) for i in range(256)])[:, :3] * 255

    n = pts.shape[0]
    pts = np.hstack((pts[:, :3], np.ones((n, 1))))
    pts_2d = np.dot(pts, np.transpose(matrix))
    pts_2d[:, 0] = pts_2d[:, 0] / pts_2d[:, 2]
    pts_2d[:, 1] = pts_2d[:, 1] / pts_2d[:, 2]
    mask = (pts_2d[:, 0] < width) & (pts_2d[:, 1] < height) & \
           (pts_2d[:, 0] > 0) & (pts_2d[:, 1] > 0) & \
           (pts_2d[:, 2] > 5) & (pts_2d[:, 2] < 80)
    pts_2d = pts_2d[mask, :]
    img = np.asarray(img)
    for i in range(pts_2d.shape[0]):
        depth = pts_2d[i, 2]
        color = cmap[int(3 * depth), :]
        cv2.circle(img,
                   (int(np.round(pts_2d[i, 0])), int(np.round(pts_2d[i, 1]))),
                   3, color=tuple(color), thickness=-1)
        # depth_img[int(np.floor(pts_2d[i, 0])), int(np.floor(pts_2d[i, 1]))] = [depth, depth, depth]

    # Image.fromarray(img).save(f'./output_radar/{img_path.split("/")[-1]}')
    img_crop = img[500:1200, 750:2750, :]
    img_crop = Image.fromarray(img_crop)
    img_crop.show()


def data_structure(img_path, lidar_pts, radar_pts, calib_lidar, calib_radar):
    match = re.search(r'group\d\d\d\d', img_path)
    group = match.group()
    timestamp = img_path.split('LeopardCamera1/')[1][:-4].replace('.', '_')

    img = Image.open(img_path)

    res_name = f'{group}_{timestamp}'
    print(res_name)
    lidar_img = np.zeros((704, 2000, 3), dtype=np.uint8)
    color_img = np.zeros((704, 2000, 3), dtype=np.uint8)
    radarD_img = np.zeros((704, 2000, 3), dtype=np.uint8)
    radarV_img = np.zeros((704, 2000, 3), dtype=np.uint8)

    matrix_lidar = json.load(open(calib_lidar))['VelodyneLidar_to_LeopardCamera1_TransformMatrix']
    try:
        matrix_radar = json.load(open(calib_radar))['OCULiiRadar_to_LeopardCamera1_TransformMatrix']
    except:
        matrix_radar = json.load(open(calib_ti))['TIRadar_to_LeopardCamera1_TransformMatrix']
    matrix_lidar = np.asarray(matrix_lidar)
    matrix_radar = np.asarray(matrix_radar)

    n = lidar_pts.shape[0]
    lidar_pts = np.hstack((lidar_pts[:, :3], np.ones((n, 1))))
    lidar_2d = np.dot(lidar_pts, np.transpose(matrix_lidar))
    lidar_2d[:, 0] = lidar_2d[:, 0] / lidar_2d[:, 2]
    lidar_2d[:, 1] = lidar_2d[:, 1] / lidar_2d[:, 2]
    mask = (lidar_2d[:, 0] < 2750) & (lidar_2d[:, 1] < 1204) & \
           (lidar_2d[:, 0] > 750) & (lidar_2d[:, 1] > 500) & \
           (lidar_2d[:, 2] > 5) & (lidar_2d[:, 2] <= 255)
    lidar_2d = lidar_2d[mask, :]
    lidar_2d[:, 0] -= 750
    lidar_2d[:, 1] -= 500
    cmap = plt.cm.get_cmap('hsv', 256)
    cmap = np.array([cmap(i) for i in range(256)])[:, :3] * 255

    image = np.asarray(img)[500:1204, 750:2750, :]
    Image.fromarray(image).save(f'./DepthTrainData/image/train/{res_name}.png')

    for i in range(lidar_2d.shape[0]):
        depth = lidar_2d[i, 2]
        color = cmap[int(depth), :]
        cv2.circle(lidar_img,
                   (int(np.round(lidar_2d[i, 0])), int(np.round(lidar_2d[i, 1]))),
                   1, color=tuple([int(depth), int(depth), int(depth)]), thickness=-1)
    Image.fromarray(lidar_img).convert('L').save(f'./DepthTrainData/lidar/train/{res_name}.png')

    for i in range(lidar_2d.shape[0]):
        depth = lidar_2d[i, 2]
        color = cmap[int(depth), :]
        cv2.circle(color_img,
                   (int(np.round(lidar_2d[i, 0])), int(np.round(lidar_2d[i, 1]))),
                   1, color=tuple(color), thickness=-1)
    Image.fromarray(color_img).convert('RGB').save(f'./DepthTrainData/lidar/color/{res_name}.png')

    n = radar_pts.shape[0]
    velo = radar_pts[:, 3:4]
    radar_pts = np.hstack((radar_pts[:, :3], np.ones((n, 1))))
    radar_2d = np.dot(radar_pts, np.transpose(matrix_radar))
    radar_2d[:, 0] = radar_2d[:, 0] / radar_2d[:, 2]
    radar_2d[:, 1] = radar_2d[:, 1] / radar_2d[:, 2]
    mask = (radar_2d[:, 0] < 2750) & (radar_2d[:, 1] < 1204) & \
           (radar_2d[:, 0] > 750) & (radar_2d[:, 1] > 500) & \
           (radar_2d[:, 2] > 5) & (radar_2d[:, 2] <= 255)
    radar_2d = radar_2d[mask, :]
    velo = velo[mask, :]
    radar_2d = np.concatenate([radar_2d, velo], axis=-1)
    radar_2d[:, 0] -= 750
    radar_2d[:, 1] -= 500
    radar_2d[:, 3] += 128
    for i in range(radar_2d.shape[0]):
        depth = radar_2d[i, 2]
        velo = radar_2d[i, 3]
        color = cmap[int(depth), :]
        cv2.circle(radarD_img,
                   (int(np.round(radar_2d[i, 0])), int(np.round(radar_2d[i, 1]))),
                   1, color=tuple([int(depth), int(depth), int(depth)]), thickness=-1)
        cv2.circle(radarV_img,
                   (int(np.round(radar_2d[i, 0])), int(np.round(radar_2d[i, 1]))),
                   1, color=tuple([int(velo), int(velo), int(velo)]), thickness=-1)
    Image.fromarray(radarD_img).convert('L').save(f'./DepthTrainData/depth/train/{res_name}.png')
    Image.fromarray(radarV_img).convert('L').save(f'./DepthTrainData/velo/train/{res_name}.png')


def loadTIRadarHeatmap(heatmap_path):
    '''
    read TI radar heatmap
    :param heatmap_path: str - TI radar heatmap path
    :return: dict(np.array)
    '''
    data = np.fromfile(heatmap_path, dtype='float32')
    print(data.shape)
    data = data.reshape((4 * 257, 232), order='F')
    data = data.reshape((4, 257, 232))
    res = {
        "heatmap_static": data[0, :, :],
        "heatmap_dynamic": data[1, :, :],
        "x_bins": data[2, :, :],
        "y_bins": data[3, :, :],
    }
    return res


def generateRadarBev(pth):
    pth = './Dataset/20211027_1_group0021/group0021_frame0000/TIRadar/1635319097.410.heatmap.bin'
    res = loadTIRadarHeatmap(pth)
    # shape: 257, 232
    x_bins = res['x_bins']
    y_bins = res['y_bins']
    static = res['heatmap_static']
    dynamic = res['heatmap_dynamic']

    map = np.zeros((151, 76))
    print(map.shape)
    for i in range(x_bins.shape[0]):
        for j in range(x_bins.shape[1]):
            map[int(np.ceil(x_bins[i, j])) + 75, int(np.ceil(y_bins[i, j]))] = dynamic[i, j]

    data = map.T[::-1, :]
    rb = data / np.max(data)
    mask = np.where(rb > 0.5, rb, 0)
    ######## Plot 3d #############
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    # ax.scatter(x_bins, y_bins, dynamic, c=dynamic, s=1)
    ax.plot_surface(x_bins, y_bins, dynamic, cmap=plt.get_cmap('rainbow'))
    ax.view_init(elev=90, azim=-90)
    plt.show()

    ##############################
    return rb, mask


def calibrationLidRad(pth, pts):
    pth = './Dataset/20211027_1_group0021/group0021_frame0000/TIRadar/1635319097.410.heatmap.bin'
    res = loadTIRadarHeatmap(pth)
    # shape: 257, 232
    x_bins = res['x_bins']
    y_bins = res['y_bins']
    static = res['heatmap_static']
    dynamic = res['heatmap_dynamic']
    ######## Lidar Filter ########

    side_range = (-75, 75)
    fwd_range = (0, 75)
    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
    f_filter = np.logical_and(x > fwd_range[0], x < fwd_range[1])
    s_filter = np.logical_and(y > side_range[0], y < side_range[1])
    filter = np.logical_and(f_filter, s_filter)
    indices = np.argwhere(filter).flatten()
    x, y, z = x[indices], y[indices], z[indices]

    dynamic = dynamic / np.max(dynamic)
    ######## Cv2.remap ###########
    tmp = cv2.remap(dynamic, y_bins, x_bins, cv2.INTER_LINEAR)
    plt.imshow(tmp)
    plt.show()
    return
    ######## Plot 3d #############
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    # ax.scatter(x_bins, y_bins, dynamic, c=dynamic, s=1)
    # ax.scatter3D(y_bins, x_bins, 3, c=dynamic, cmap='jet')
    ax.plot_surface(y_bins, x_bins, dynamic, cmap='jet')
    ax.scatter3D(x, y, z, c='k', s=1)
    ax.view_init(elev=74, azim=-90)
    plt.show()

    ##############################


def radar_polar_to_cartesian(pth=None, cart_pixel_width=601, cart_pixel_height=301):
    # pth = './Dataset/20211027_1_group0021/group0021_frame0000/TIRadar/1635319097.410.heatmap.bin'
    res = loadTIRadarHeatmap(pth)
    # shape: 257, 232
    x_bins = res['x_bins']
    y_bins = res['y_bins']
    static = res['heatmap_static']
    dynamic = res['heatmap_dynamic']  # shape: 257, 232
    ############ Setting ##############
    # cart_pixel_width, cart_pixel_height = 600, 300
    # cart_resolution = 0.25
    ###################################
    coords_x = np.linspace(-75, 75, cart_pixel_width, dtype=np.float32)
    coords_y = np.linspace(0, 75, cart_pixel_height, dtype=np.float32)
    Y, X = np.meshgrid(coords_y, coords_x)
    sample_range = np.sqrt(Y * Y + X * X)  # shape: 600, 300
    sample_angle = np.arctan2(Y, X) / np.pi * 180

    # Interpolate Radar Data Coordinates
    angle = (np.arctan2(y_bins, x_bins) / np.pi * 180)[:, 0]  # shape: 257,
    distance = np.sqrt(x_bins ** 2 + y_bins ** 2)[0, :]  # shape: 232,
    anglx = np.arange(0, 257)
    distancx = np.arange(0, 232)

    sample_u = np.interp(sample_range, distance, distancx).astype(np.float32)
    sample_v = np.interp(sample_angle, angle, anglx).astype(np.float32)

    # We clip the sample points to the minimum sensor reading range so that we
    # do not have undefined results in the centre of the image. In practice
    # this region is simply undefined.
    sample_u[sample_u < 0] = 0

    ####### Heatmap normalization ######################
    hm = static + dynamic
    hm = np.uint8(hm / np.max(hm) * 255)
    hm = np.expand_dims(hm, -1)

    ####### Heatmap remap ##############################
    polar_to_cart_warp = np.stack((sample_u, sample_v), -1)
    cart_img = np.expand_dims(cv2.remap(hm, polar_to_cart_warp, None, cv2.INTER_LINEAR), -1)
    cart_im = cart_img[:, :, 0]
    # plt.imshow(cart_im.T[::-1, :], cmap='jet')
    # plt.show()
    return cart_im.T[::-1, ::-1]


if __name__ == '__main__':
    # radar_polar_to_cartesian()
    # raise NotImplemented
    # generateRadarBev(1)
    # raise NotImplemented
    # Build Training Dataset
    # if not os.path.exists('./DepthTrainData/lidar/color'):
    #     os.makedirs('./DepthTrainData/lidar/color', exist_ok=True)
    # if not os.path.exists('./DepthTrainData/image/train'):
    #     os.makedirs('./DepthTrainData/image/train', exist_ok=True)
    # if not os.path.exists('./DepthTrainData/depth/train'):
    #     os.makedirs('./DepthTrainData/depth/train', exist_ok=True)
    # if not os.path.exists('./DepthTrainData/velo/train'):
    #     os.makedirs('./DepthTrainData/velo/train', exist_ok=True)
    # if not os.path.exists('./DepthTrainData/segmentation/train'):
    #     os.makedirs('./DepthTrainData/segmentation/train', exist_ok=True)
    # if not os.path.exists('./DepthTrainData/lidar/train'):
    #     os.makedirs('./DepthTrainData/lidar/train', exist_ok=True)
    #
    base_path = './Dataset'
    groups = os.listdir(base_path)
    groups = sorted(groups)
    for group in groups:
        group_path = os.path.join(base_path, group)
        folders = os.listdir(group_path)
        folders = sorted(folders)
        for folder in folders:
            camera_path = os.path.join(group_path, folder, 'LeopardCamera1')
            for file in os.listdir(camera_path):
                if file[-3:] == 'png':
                    img_path = os.path.join(camera_path, file)
            lidar_path = os.path.join(group_path, folder, 'VelodyneLidar')
            for file in os.listdir(lidar_path):
                if file[-3:] == 'pcd':
                    pcd_lidar = os.path.join(lidar_path, file)
                if file[-4:] == 'json':
                    calib_lidar = os.path.join(lidar_path, file)
            radar_path = os.path.join(group_path, folder, 'OCULiiRadar')
            for file in os.listdir(radar_path):
                if file[-3:] == 'pcd':
                    pcd_radar = os.path.join(radar_path, file)
                if file[-4:] == 'json':
                    calib_radar = os.path.join(radar_path, file)

            ti_path = os.path.join(group_path, folder, 'TIRadar')
            for file in os.listdir(ti_path):
                if file[-3:] == 'pcd':
                    pcd_ti = os.path.join(ti_path, file)
                if file[-4:] == 'json':
                    calib_ti = os.path.join(ti_path, file)
                if file[-3:] == 'bin':
                    hm_ti = os.path.join(ti_path, file)
            # data_structure(img_path, read_pcd(pcd_lidar), read_pcd(pcd_radar), calib_lidar, calib_radar)
            # pts2camera(read_pcd(pcd_lidar)[:, :3], img_path, calib_lidar)
            # calibrationLidRad(hm_ti, read_pcd(pcd_lidar))
            cart_im = radar_polar_to_cartesian(hm_ti)
            lid_im = pts2bev(read_pcd(pcd_lidar))
            # rb, mask = generateRadarBev(hm_ti)
            Image.fromarray(np.uint8(lid_im)).save('Bev.png')
            cm = plt.get_cmap('jet')
            colored_image = cm(cart_im)
            Image.fromarray((colored_image[:, :, :3]*255).astype(np.uint8)).save('Radar.png')
            # Image.fromarray(np.uint8(rb*255)).show()
            break
        break
    # Copy Segmentationn Dataset
    # seg_path = '/media/personal_data/lizc/1/erfnet_pytorch/dataset/predict_out/gray/'
    # groups = os.listdir(seg_path)
    # groups = sorted(groups)
    # for group in groups:
    #     files = os.listdir(os.path.join(seg_path, group))
    #     files = sorted(files)
    #     for file in files:
    #         name = file.replace('_leftImg8bit', '')
    #         print(name)
    #         img = Image.open(os.path.join(seg_path, group, file))
    #         img = np.asarray(img)
    #         img = img*30
    #         Image.fromarray(img).convert('L').save(f'./DepthTrainData/segmentation/train/{name}')
