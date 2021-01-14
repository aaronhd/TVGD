import numpy as np

import matplotlib.pyplot as plt

from skimage.draw import polygon
from skimage.feature import peak_local_max

# img1 = np.zeros((7, 7))
# img1[3, 4] = 1
# img1[3, 2] = 1.5
# print(img1)
#
# print(peak_local_max(img1, min_distance=1, num_peaks=3))
#
# fields = [
#     'img_id',
#     'rgb',
#     'depth_inpainted',
#     'bounding_boxes',
#     'grasp_points_img',
#     'angle_img',
#     'grasp_width'
# ]
#
# # Empty datatset.
# dataset = {
#     'test': dict([(f, []) for f in fields]),
#     'train': dict([(f, []) for f in fields])
# }
# print(dataset)
# print(dataset['test'])
#
# for tt_name in dataset:
#     for ds_name in dataset[tt_name]:
# #         print(ds_name)
#
#
# from skimage.draw import polygon
# img = np.zeros((10, 10), dtype=np.uint8)
# r = np.array([1, 2, 8, 1])
# c = np.array([1, 7, 4, 1])
# rr, cc = polygon(r, c)
# print(rr)
# print(len(rr))
# print(cc)
# img[rr, cc] = 3
#
# print(img)


# a = np.array(
#             [1292, 1294 , 1816, 1613])
# a = a.astype(np.float32)
#
# print(a)
# b = a/1000
# # print(b)
# theta = 10.0
# A = np.array([[0.0592, 0.9842, 0.1667], [0.4415, -0.1755, 0.88], [0.8953, 0.0214, -0.4449]])
# B = np.array([[1, 0, 0],
#              [0, np.cos(np.deg2rad(theta)), -np.sin(np.deg2rad(theta))],
#              [0, np.sin(np.deg2rad(theta)), np.cos(np.deg2rad(theta))]])
# # print(B)
# print(A)
# ans = np.dot(B, A)
# print(ans)
# print(np.cos(np.deg2rad(20)))
# print(np.rad2deg(20))
# print(np.deg2rad(65))
# print(np.eye(3))





def RYXZ(alpha, beta, gamma):
    a11 = np.sin(np.deg2rad(alpha))*np.sin(np.deg2rad(beta))*np.sin(np.deg2rad(gamma)) + np.cos(np.deg2rad(alpha))*np.cos(np.deg2rad(gamma))
    a12 = np.sin(np.deg2rad(alpha))*np.sin(np.deg2rad(beta))*np.cos(np.deg2rad(gamma)) - np.cos(np.deg2rad(alpha))*np.sin(np.deg2rad(gamma))
    a13 = np.sin(np.deg2rad(alpha))*np.cos(np.deg2rad(beta))

    a21 = np.cos(np.deg2rad(beta))*np.sin(np.deg2rad(gamma))
    a22 = np.cos(np.deg2rad(beta))*np.cos(np.deg2rad(gamma))
    a23 = -np.sin(np.deg2rad(beta))

    a31 = np.cos(np.deg2rad(alpha))*np.sin(np.deg2rad(beta))*np.sin(np.deg2rad(gamma)) - np.sin(np.deg2rad(alpha))*np.cos(np.deg2rad(gamma))
    a32 = np.cos(np.deg2rad(alpha))*np.sin(np.deg2rad(beta))*np.cos(np.deg2rad(gamma)) + np.cos(np.deg2rad(alpha))*np.sin(np.deg2rad(gamma))
    a33 = np.cos(np.deg2rad(alpha))*np.cos(np.deg2rad(beta))

    return np.array([[a11, a12, a13], [a21, a22, a23], [a31, a32, a33]])


def RXYZ(alpha, beta, gamma):
    a11 = np.cos(np.deg2rad(beta))*np.cos(np.deg2rad(gamma))
    a12 = -np.cos(np.deg2rad(beta))*np.sin(np.deg2rad(gamma))
    a13 = np.sin(np.deg2rad(beta))

    a21 = np.sin(np.deg2rad(alpha))*np.sin(np.deg2rad(beta))*np.cos(np.deg2rad(gamma)) + np.cos(np.deg2rad(alpha))*np.sin(np.deg2rad(gamma))
    a22 = -np.sin(np.deg2rad(alpha))*np.sin(np.deg2rad(beta))*np.sin(np.deg2rad(gamma)) + np.cos(np.deg2rad(alpha))*np.cos(np.deg2rad(gamma))
    a23 = -np.sin(np.deg2rad(alpha))*np.cos(np.deg2rad(beta))

    a31 = -np.cos(np.deg2rad(alpha))*np.sin(np.deg2rad(beta))*np.cos(np.deg2rad(gamma)) + np.sin(np.deg2rad(alpha))*np.sin(np.deg2rad(gamma))
    a32 = np.cos(np.deg2rad(alpha))*np.sin(np.deg2rad(beta))*np.sin(np.deg2rad(gamma)) + np.sin(np.deg2rad(alpha))*np.cos(np.deg2rad(gamma))
    a33 = np.cos(np.deg2rad(alpha))*np.cos(np.deg2rad(beta))
    return np.array([[a11, a12, a13], [a21, a22, a23], [a31, a32, a33]])

def RXYZ_fix(alpha, beta, gamma):
    a11 = np.cos(np.deg2rad(alpha))*np.cos(np.deg2rad(beta))
    a12 = np.cos(np.deg2rad(alpha))*np.sin(np.deg2rad(beta))*np.sin(np.deg2rad(gamma)) - np.sin(np.deg2rad(alpha))*np.cos(np.deg2rad(gamma))
    a13 = np.cos(np.deg2rad(alpha))*np.sin(np.deg2rad(beta))*np.cos(np.deg2rad(gamma)) + np.sin(np.deg2rad(alpha))*np.sin(np.deg2rad(gamma))

    a21 = np.sin(np.deg2rad(alpha))*np.cos(np.deg2rad(beta))
    a22 = np.sin(np.deg2rad(alpha))*np.sin(np.deg2rad(beta))*np.sin(np.deg2rad(gamma)) + np.cos(np.deg2rad(alpha))*np.cos(np.deg2rad(gamma))
    a23 = np.sin(np.deg2rad(alpha))*np.sin(np.deg2rad(beta))*np.cos(np.deg2rad(gamma)) + np.cos(np.deg2rad(alpha))*np.sin(np.deg2rad(gamma))


    a31 = -np.sin(np.deg2rad(beta))
    a32 = np.cos(np.deg2rad(beta))*np.sin(np.deg2rad(gamma))
    a33 = np.cos(np.deg2rad(beta))*np.cos(np.deg2rad(gamma))
    return np.array([[a11, a12, a13], [a21, a22, a23], [a31, a32, a33]])


#
# print(RXYZ_fix(-49.45, 0, 180))

print(np.rad2deg(0.666))

# z20 = np.array([[np.cos(20)]])
