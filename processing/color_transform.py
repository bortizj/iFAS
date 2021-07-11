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