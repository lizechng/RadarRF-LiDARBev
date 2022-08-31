import numpy as np

def load_TIRadarHeatmap(heatmap_path):
    '''
    read TI radar heatmap
    :param heatmap_path: str - TI radar heatmap path
    :return: dict(np.array)
    '''
    data = np.fromfile(heatmap_path, dtype='float32')
    print(data.shape)
    data = data.reshape((4*257, 232), order='F')
    data = data.reshape((4, 257, 232))
    res = {
        "heatmap_static": data[0, :, :],
        "heatmap_dynamic": data[1, :, :],
        "x_bins": data[2, :, :],
        "y_bins": data[3, :, :],
    }
    return res

pth = './Dataset/20211027_1_group0021/group0021_frame0000/TIRadar/1635319097.410.heatmap.bin'
res = load_TIRadarHeatmap(pth)
x_bins = res['x_bins']
y_bins = res['y_bins']
static = res['heatmap_static']
dynamic = res['heatmap_dynamic']
print(x_bins.shape)
print(static.shape)
print(dynamic.shape)

import matplotlib.pyplot as plt

print(np.arctan2(y_bins[20, :], x_bins[20, :]) / (np.pi) * 180)
# print(y_bins[-1, :], y_bins[-1, :].shape)
plt.imshow(dynamic)
plt.show()
print(np.max(y_bins))
n = 1
map = np.zeros((151*n, 76*n))
print(map.shape)
for i in range(x_bins.shape[0]):
    for j in range(x_bins.shape[1]):
       map[(int(np.ceil(x_bins[i,j])) + 75)*n, int(np.ceil(y_bins[i,j]))*n] = dynamic[i,j]

data = map.T[::-1,:]
print(data.shape)
# data_split = data[55:, 50:100]
data_split = data
print(np.max(data_split), np.min(data_split))
d = data_split / np.max(data_split)
# plt.imshow(d, interpolation='gaussian')
import cv2
# d = cv2.GaussianBlur(d, (3, 3), 1)
plt.imshow(d)
plt.show()
dd = np.where(d, 1, 0)
plt.imshow(dd)
plt.show()
# import seaborn as sns
# sns.heatmap(d)
# plt.show()
