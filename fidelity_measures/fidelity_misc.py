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
import pywt
import cv2

import image_processing.img_misc as misc


def psnr(ref_img, tst_img):
    mse = np.mean(np.power(ref_img - tst_img, 2))
    if mse != 0:
        psnr = 10 * np.log10((255. ** 2) / mse)
    else:
        psnr = np.inf
    return psnr


# Z. Wang, et.al.
# Image quality assessment: from error visibility to structural similarity
# IEEE Transactions on Image processing 2004
def ssim(ref_img, tst_img, *argv):
    # Checcking if there are more arguments or setting to default
    if len(argv) < 1:
        k, wsize = [0.01, 0.03], 11
    elif len(argv) < 2:
        k, wsize = argv[0], 11
    elif len(argv) < 3:
        k, wsize = argv[0], argv[1]

    ref_img_gray = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY).astype('float32')
    tst_img_gray = cv2.cvtColor(tst_img, cv2.COLOR_BGR2GRAY).astype('float32')

    c1 = (k[0] * 255) ** 2
    c2 = (k[1] * 255) ** 2

    ref_mu = cv2.GaussianBlur(ref_img_gray, (wsize, wsize), 1.5)
    tst_mu = cv2.GaussianBlur(tst_img_gray, (wsize, wsize), 1.5)

    ref_mu_sq = (ref_mu * ref_mu).astype('float32')
    tst_mu_sq = (tst_mu * tst_mu).astype('float32')
    ref_tst_mu = (ref_mu * tst_mu).astype('float32')

    ref_sigma_sq = cv2.GaussianBlur((ref_img_gray * ref_img_gray).astype('float32'), (wsize, wsize), 1.5) - ref_mu_sq
    tst_sigma_sq = cv2.GaussianBlur((tst_img_gray * tst_img_gray).astype('float32'), (wsize, wsize), 1.5) - tst_mu_sq
    ref_tst_sigma = cv2.GaussianBlur((ref_img_gray * tst_img_gray).astype('float32'), (wsize, wsize), 1.5) - ref_tst_mu

    if c1 > 0 and c2 > 0:
        ssim_map = (
            ((2 * ref_tst_mu + c1) * (2 * ref_tst_sigma + c2)) / 
            ((ref_mu_sq + tst_mu_sq + c1) * (ref_sigma_sq + tst_sigma_sq + c2))
            )
    else:
        numerator1 = 2 * ref_tst_mu + c1
        numerator2 = 2 * ref_tst_sigma + c2
        denominator1 = ref_mu_sq + tst_mu_sq + c1
        denominator2 = ref_sigma_sq + tst_sigma_sq + c2
        ssim_map = np.ones_like(ref_mu)
        index = denominator1 * denominator2 > 0
        ssim_map[index] = (numerator1[index] * numerator2[index]) / (denominator1[index] * denominator2[index])
        index = (denominator1 != 0) & (denominator2 == 0)
        ssim_map[index] = numerator1[index] / denominator1[index]

    return np.nanmean(ssim_map)


# B. Goossens, et.al.
# Wavelet domain image denoising for non-stationary noise and signal-dependent noise
# IEEE PROCEEDINGS of the International conference on image processing
def noise_difference(ref_img, tst_img):
    ref_img_gray = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY).astype('float32')
    tst_img_gray = cv2.cvtColor(tst_img, cv2.COLOR_BGR2GRAY).astype('float32')
    _, (_, _, ref_cd) = pywt.dwt2(ref_img_gray, 'db1', 'sym')
    _, (_, _, tst_cd) = pywt.dwt2(tst_img_gray, 'db1', 'sym')
    noise_ref = misc.estimate_noise(ref_cd)
    noise_pro = misc.estimate_noise(tst_cd)

    return noise_ref - noise_pro


# Based on guide
# https://docs.opencv.org/master/d4/d13/tutorial_py_filtering.html
def blur_difference(ref_img, tst_img):
    ref_img_gray = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY).astype('float32')
    tst_img_gray = cv2.cvtColor(tst_img, cv2.COLOR_BGR2GRAY).astype('float32')
    ref_var = misc.variance_of_laplacian(ref_img_gray)
    tst_var = misc.variance_of_laplacian(tst_img_gray)
    
    return ref_var - tst_var


# C. Lee, et.al.
# EPSNR for objective image quality measurements
# Proceedings of the First International Conference on Computer Imaging Theory and Applications
def epsnr(ref_img, tst_img):
    ref_img_gray = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY).astype('float32')
    tst_img_gray = cv2.cvtColor(tst_img, cv2.COLOR_BGR2GRAY).astype('float32')

    ref_px = cv2.Sobel(ref_img_gray, -1, 1, 0, ksize=3)
    ref_py = cv2.Sobel(ref_img_gray, -1, 0, 1, ksize=3)
    ref_mag = np.sqrt(ref_px * ref_px + ref_py * ref_py)

    thr = misc.median_threshold(ref_mag)
    mask = ref_mag >= thr

    dif = ref_img_gray - tst_img_gray
    dif_sq = (dif * dif)
    emse = np.mean(dif_sq[mask])

    if emse != 0:
        epsnr = 10 * np.log10((255. ** 2) / emse)
    else:
        epsnr = np.inf

    return epsnr
