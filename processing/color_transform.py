"""
Copyleft 2021
This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation version 3.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.

author: Benhur Ortiz-Jaramillo
"""

import numpy as np
import cv2


def bgr_to_l_alpha_beta(img):
    RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    A = np.array([[0.3811, 0.5783, 0.0402],
                  [0.1967, 0.7244, 0.0782],
                  [0.0241, 0.1288, 0.8444]])
    L = np.log(A[0, 0] * RGB[:, :, 0] + A[0, 1] * RGB[:, :, 1] + A[0, 2] * RGB[:, :, 2] + 1)
    M = np.log(A[1, 0] * RGB[:, :, 0] + A[1, 1] * RGB[:, :, 1] + A[1, 2] * RGB[:, :, 2] + 1)
    S = np.log(A[2, 0] * RGB[:, :, 0] + A[2, 1] * RGB[:, :, 1] + A[2, 2] * RGB[:, :, 2] + 1)

    A = np.dot(np.array([[1. / np.sqrt(3.), 0., 0.],
                         [0., 1. / np.sqrt(6.), 0.],
                         [0., 0., 1. / np.sqrt(2.)]]),
               np.array([[1.,  1.,  1.],
                         [1.,  1., -2.],
                         [1., -1.,  0.]]))

    l = A[0, 0] * L + A[0, 1] * M + A[0, 2] * S
    alpha = A[1, 0] * L + A[1, 1] * M + A[1, 2] * S
    beta = A[2, 0] * L + A[2, 1] * M + A[2, 2] * S

    return np.dstack((l, alpha, beta))


def linear_color_transform(img, tr_type="rgb_to_xyz"):
    in_img = img.astype("float32")
    if tr_type == "rgb_to_xyz":
        tr_mat = np.array(
            [[42.62846,  38.29084,  13.67019], [21.64618,  72.06528,   5.83799], [1.77295,  12.93408,  92.75945]]
            )
    elif tr_type == "xyz_to_rgb":
        tr_mat = np.array(
            [[42.62846,  38.29084,  13.67019], [21.64618,  72.06528,   5.83799], [1.77295,  12.93408,  92.75945]]
            )
        tr_mat = np.linalg.inv(tr_mat)
    elif tr_type == "xyz_to_o1o2o3":
        tr_mat = np.array([[0.2787,  0.7218, -0.1066], [-0.4488,  0.2898, -0.0772], [0.0860, -0.5900,  0.5011]])
    elif tr_type == "o1o2o3_to_xyz":
        tr_mat = np.array([[0.2787,  0.7218, -0.1066], [-0.4488,  0.2898, -0.0772], [0.0860, -0.5900,  0.5011]])
        tr_mat = np.linalg.inv(tr_mat)

    out1 = tr_mat[0, 0] * in_img[:, :, 0] + tr_mat[0, 1] * in_img[:, :, 1] + tr_mat[0, 2] * in_img[:, :, 2]
    out2 = tr_mat[1, 0] * in_img[:, :, 0] + tr_mat[1, 1] * in_img[:, :, 1] + tr_mat[1, 2] * in_img[:, :, 2]
    out3 = tr_mat[2, 0] * in_img[:, :, 0] + tr_mat[2, 1] * in_img[:, :, 1] + tr_mat[2, 2] * in_img[:, :, 2]

    return np.dstack((out1, out2, out3))
