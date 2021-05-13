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

author: Benhur Ortiz Jaramillo
"""

import numpy as np
import cv2


# B. Goossens, et.al.
# Wavelet domain image denoising for non-stationary noise and signal-dependent noise
# IEEE PROCEEDINGS of the International conference on image processing
def estimate_noise(cd_wavelet):
    """Computes means in blocks of size 8"""
    # initializing constants 
    b = 8
    alpha = 0.1
    corr_factor = 0.716
    window = np.ones((b + 1, b + 1), dtype=np.float32)
    window[:, 0] = 0.0
    window[0, :] = 0.0
    window = (window / (b * b)).astype("float32")
    avg_blk = cv2.filter2D(np.sqrt(np.abs(cd_wavelet)), ddepth=-1, kernel=window, borderType=cv2.BORDER_REFLECT_101)
    block = avg_blk[int(b / 2):int(-(b / 2 - 1)):b, int(b / 2):int(-(b / 2 - 1)):b]
    block = np.sort(block, axis=None)
    avg_block = 1 / corr_factor * np.mean(block[0:np.int(np.floor(alpha * (len(block) - 1)))])
    return np.sqrt(avg_block)


# Based on guide
# https://docs.opencv.org/master/d4/d13/tutorial_py_filtering.html
def variance_of_laplacian(img):
	return cv2.Laplacian(img, -1).var()


# Threshold based on the median and deviation of differences
def median_threshold(img, std_fact=4.4478):
    med = np.median(img)
    stdev = np.median(np.abs(img - med))
    th = med + std_fact * stdev
    return th
