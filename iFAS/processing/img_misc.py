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

from scipy import signal
import numpy as np
from iFAS.gui import ifas_misc
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
    window[::, 0] = 0.0
    window[0, ::] = 0.0
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


# Brightness weight for jncd_delta_e
def rho_jncd_delta_e(mu_y):
    rho_mu_y = np.zeros_like(mu_y)
    rho_mu_y[mu_y <= 6] = 0.06
    rho_mu_y[np.logical_and(mu_y > 6, mu_y <= 100)] = 0.04
    rho_mu_y[np.logical_and(mu_y > 100, mu_y <= 140)] = 0.01
    rho_mu_y[mu_y > 140] = 0.03

    return rho_mu_y

# Colorfullness based on Gao formula
def colorfulness(img_bgr):
    K = 2
    alpha = img_bgr[::, ::, 2] - img_bgr[::, ::, 1]
    beta = 0.5 * (img_bgr[::, ::, 2] + img_bgr[::, ::, 1]) - img_bgr[::, ::, 1]
    mu_alpha = np.mean(alpha)
    mu_beta = np.mean(beta)
    sigma_alpha_sq = np.var(alpha)
    sigma_beta_sq = np.var(beta)
    c = 0.02 * np.log(sigma_alpha_sq / (np.power(np.abs(mu_alpha), 0.2) + K) + K) * \
        np.log(sigma_beta_sq / (np.power(np.abs(mu_beta), 0.2) + K) + K)

    return c


# Log Gabor filter
def log_gabor(rows, cols, omega_0=0.0210, sigma_f=1.34):
    u1, u2 = np.meshgrid(
        (np.arange(cols) - (np.fix(cols / 2))) / (cols - np.mod(cols, 2)),
        (np.arange(rows) - (np.fix(rows / 2))) / (rows - np.mod(rows, 2))
        )
    mask = np.ones((rows, cols))

    for ii in range(rows):
        for jj in range(cols):
            if (u1[ii, jj]**2 + u2[ii, jj]**2) > 0.25:
                mask[ii, jj] = 0

    u1 = u1 * mask
    u2 = u2 * mask
    u1 = np.fft.ifftshift(u1)
    u2 = np.fft.ifftshift(u2)

    radius = np.sqrt(np.power(u1, 2) + np.power(u2, 2))
    radius[0, 0] = 1
    idx = np.where(radius==0)
    radius[idx] = 1

    lg = np.exp((-np.power(np.log(radius / omega_0),2))/(2*(sigma_f ** 2)))
    lg[idx] = 0
    lg[0, 0] = 0

    return lg


# SDSP algorithm for salient region detection from a given image
def salient_regions(img, sigma_f=1.34, omega_0=0.0210, sigma_d=145, sigma_c=0.001):
    m, n, _ = img.shape
    resized = cv2.resize(img, (256, 256), interpolation=cv2.INTER_LINEAR)
    lab = cv2.cvtColor((resized / 255).astype("float32"), cv2.COLOR_BGR2LAB).astype("float32")

    lch = lab[::, ::, 0]
    ach = lab[::, ::, 1]
    bch = lab[::, ::, 2]

    lfft = np.fft.fft2(lch)
    afft = np.fft.fft2(ach)
    bfft = np.fft.fft2(bch)

    rows, cols, _ = resized.shape
    lg = log_gabor(rows, cols, omega_0, sigma_f)

    l = np.real(np.fft.ifft2(lfft * lg))
    a = np.real(np.fft.ifft2(afft * lg))
    b = np.real(np.fft.ifft2(bfft * lg))

    SFMap = np.sqrt(np.power(l, 2) + np.power(a, 2) + np.power(b, 2))

    idx = np.zeros((rows, cols, 2))
    idx[:,:,0] = np.tile(np.arange(rows), (cols, 1)).T
    idx[:,:,1] = np.tile(np.arange(cols), (rows, 1))

    cy = rows / 2
    cx = cols / 2

    cxy = np.zeros((rows, cols, 2))
    cxy[:,:, 0] = np.ones((rows, cols)) * cy
    cxy[:,:, 1] = np.ones((rows, cols)) * cx

    SD_map = np.exp(-np.sum(np.power(idx - cxy, 2), 2) / (sigma_d ** 2))

    maxA = np.max(ach)
    minA = np.min(ach)
    a_norm = (ach - minA) / (maxA - minA)

    maxB = np.max(bch)
    minB = np.min(bch)
    b_norm = (bch - minB) / (maxB - minB)

    ab_mag = np.power(a_norm, 2) + np.power(b_norm, 2)

    SCMap = 1 - np.exp(-ab_mag / (sigma_c ** 2))
    VSMap = SFMap * SD_map * SCMap

    VSMap = cv2.resize(VSMap, (n, m), interpolation=cv2.INTER_LINEAR)

    return VSMap / np.max(VSMap)


# Average of an 8x8 block for chroma spread and extrem 
def mean8x8block(cb, cr):
    window = np.ones((9, 9))
    window[::, 0] = 0
    window[0, ::] = 0
    window = window / (8 * 8)

    MuCb = cv2.filter2D(cb, ddepth=-1, kernel=window, borderType=cv2.BORDER_REFLECT_101)
    mCb = MuCb[0::8, 0::8]

    MuCr = cv2.filter2D(cr, ddepth=-1, kernel=window, borderType=cv2.BORDER_REFLECT_101)
    mCr = 1.5 * MuCr[0::8, 0::8]

    return mCb, mCr, MuCb, MuCr


# Color histogram of a given image
def color_histogram(img, minc=np.array([0, 0, 0]), maxc=np.array([255, 255, 255]), nbins=8, retbins=False):
    img_vec = np.hstack((img[::, ::, 0].reshape(-1, 1), img[::, ::, 1].reshape(-1, 1), img[::, ::, 2].reshape(-1, 1)))

    xx = np.linspace(minc[0], maxc[0], nbins + 1)
    yy = np.linspace(minc[1], maxc[1], nbins + 1)
    zz = np.linspace(minc[2], maxc[2], nbins + 1)

    hist, edges = np.histogramdd(img_vec, (xx, yy, zz))

    if retbins:
        return hist, edges
    else:
        return hist


# PSNR returning 1000 when mse = 0
def clip_psnr(s1):
    if s1 == 0:
        return 1000.
    else:
        return 10 * np.log10(255. * 255. / s1)


# Variance times the input size
def vari(x):
    return np.var(x) * x.size


# Mean squared error of discrete cossine transform coefficients 
def dct_block_mse(A_ref, A_pro):
    MaskCof = np.array([
        [0.390625, 0.826446, 1.000000, 0.390625, 0.173611, 0.062500, 0.038447, 0.026874],
        [0.694444, 0.694444, 0.510204, 0.277008, 0.147929, 0.029727, 0.027778, 0.033058],
        [0.510204, 0.591716, 0.390625, 0.173611, 0.062500, 0.030779, 0.021004, 0.031888],
        [0.510204, 0.346021, 0.206612, 0.118906, 0.038447, 0.013212, 0.015625, 0.026015],
        [0.308642, 0.206612, 0.073046, 0.031888, 0.021626, 0.008417, 0.009426, 0.016866],
        [0.173611, 0.081633, 0.033058, 0.024414, 0.015242, 0.009246, 0.007831, 0.011815],
        [0.041649, 0.024414, 0.016437, 0.013212, 0.009426, 0.006830, 0.006944, 0.009803],
        [0.019290, 0.011815, 0.011080, 0.010412, 0.007972, 0.010000, 0.009426, 0.010203]
        ])
    CSFCof = np.array([
        [1.608443, 2.339554, 2.573509, 1.608443, 1.072295, 0.643377, 0.504610, 0.421887],
        [2.144591, 2.144591, 1.838221, 1.354478, 0.989811, 0.443708, 0.428918, 0.467911],
        [1.838221, 1.979622, 1.608443, 1.072295, 0.643377, 0.451493, 0.372972, 0.459555],
        [1.838221, 1.513829, 1.169777, 0.887417, 0.504610, 0.295806, 0.321689, 0.415082],
        [1.429727, 1.169777, 0.695543, 0.459555, 0.378457, 0.236102, 0.249855, 0.334222],
        [1.072295, 0.735288, 0.467911, 0.402111, 0.317717, 0.247453, 0.227744, 0.279729],
        [0.525206, 0.402111, 0.329937, 0.295806, 0.249855, 0.212687, 0.214459, 0.254803],
        [0.357432, 0.279729, 0.270896, 0.262603, 0.229778, 0.257351, 0.249855, 0.259950]
        ])

    blk_size = 8
    M, N = A_ref.shape
    S1blk = np.zeros((M, N))
    S2blk = np.zeros((M, N))

    for ii in range(0, M, blk_size):
        for jj in range(0, N, blk_size):
            if ii + blk_size <= M and jj + blk_size <= N:
                values_ref = A_ref[ii:ii + blk_size, jj:jj + blk_size].astype("float32")
                values_ref = cv2.dct(values_ref)
                BMasket_ref = np.power(values_ref, 2) * MaskCof
                m_ref = np.sum(BMasket_ref) - BMasket_ref[0, 0]
                pop_ref = vari(values_ref)

                if pop_ref != 0:
                    pop_ref = (
                        vari(values_ref[0:3, 0:3]) + vari(values_ref[0:3, 4:7]) + vari(values_ref[4:7, 4:7]) + 
                        vari(values_ref[4:7, 0:3])
                        ) / pop_ref

                m_ref = np.sqrt(m_ref * pop_ref) / 32
                values_pro = A_pro[ii:ii + blk_size, jj:jj + blk_size]
                values_pro = cv2.dct(values_pro)
                BMasket_pro = np.power(values_pro, 2) * MaskCof
                m_pro = np.sum(BMasket_pro) - BMasket_pro[0, 0]
                pop_pro = vari(values_pro)

                if pop_pro != 0:
                    pop_pro = (
                        vari(values_pro[0:3, 0:3]) + vari(values_pro[0:3, 4:7]) + vari(values_pro[4:7, 4:7]) + 
                        vari(values_pro[4:7, 0:3])
                        ) / pop_pro

                m_pro = np.sqrt(m_pro * pop_pro) / 32
                if m_pro > m_ref:
                    m_ref = m_pro

                Dif_dct = np.abs(values_ref - values_pro)
                S1 = np.power(Dif_dct * CSFCof, 2)

                for kk in range(0, blk_size):
                    for ll in range(0, blk_size):
                        if kk != 0 or ll != 0:
                            if Dif_dct[kk, ll] < m_ref / MaskCof[kk, ll]:
                                Dif_dct[kk, ll] = 0
                            else:
                                Dif_dct[kk, ll] -= m_ref / MaskCof[kk, ll]

                S2 = np.power(Dif_dct * CSFCof, 2)
                S1blk[ii:ii + blk_size, jj:jj + blk_size] = np.mean(S1)
                S2blk[ii:ii + blk_size, jj:jj + blk_size] = np.mean(S2)

    s1 = np.nanmean(S1blk)
    s2 = np.nanmean(S2blk)

    return s1, s2, S1blk, S2blk


# Separable filers
def separable_filters(samp_per_deg, dimension=1):
    min_samp_per_deg = 224.
    if ((samp_per_deg < min_samp_per_deg) and dimension != 2):
        uprate = np.ceil(min_samp_per_deg / samp_per_deg)
        samp_per_deg = samp_per_deg * uprate
    else:
        uprate = 1.

    x1 = np.array([0.05, 1.00327, 0.225, 0.114416, 7.0, -0.117686])
    x2 = np.array([0.0685, 0.616725, 0.826, 0.383275])
    x3 = np.array([0.0920, 0.567885, 0.6451, 0.432115])

    x1[[1, 3, 5]] = x1[[1, 3, 5]] * samp_per_deg
    x2[[1, 3]] = x2[[1, 3]] * samp_per_deg
    x3[[1, 3]] = x3[[1, 3]] * samp_per_deg
    width = np.ceil(samp_per_deg / 2) * 2 - 1

    k1 = np.vstack((
        np.vstack((ifas_misc.gauss(x1[0], width) * np.sqrt(np.abs(x1[1])) * np.sign(x1[1]), 
                   ifas_misc.gauss(x1[2], width) * np.sqrt(np.abs(x1[3])) * np.sign(x1[3]))),
        ifas_misc.gauss(x1[4], width) * np.sqrt(np.abs(x1[5])) * np.sign(x1[5])))
    k2 = np.vstack((
        ifas_misc.gauss(x2[0], width) * np.sqrt(np.abs(x2[1])) * np.sign(x2[1]),
        ifas_misc.gauss(x2[2], width) * np.sqrt(np.abs(x2[3])) * np.sign(x2[3])
        ))
    k3 = np.vstack((
        ifas_misc.gauss(x3[0], width) * np.sqrt(np.abs(x3[1])) * np.sign(x3[1]),
        ifas_misc.gauss(x3[2], width) * np.sqrt(np.abs(x3[3])) * np.sign(x3[3])
        ))

    if ((dimension != 2) and uprate > 1):
        upcol = np.hstack((np.arange(1, uprate + 1), np.arange(uprate - 1, 0, -1))) / uprate
        s = upcol.size
        upcol = resize(upcol, np.array([1, s + width - 1]))
        up1 = signal.convolve2d(k1, upcol, mode='same')
        up2 = signal.convolve2d(k2, upcol, mode='same')
        up3 = signal.convolve2d(k3, upcol, mode='same')
        s = up1.shape[1]
        mid = np.ceil(s / 2.)
        downs = np.hstack((np.arange(1, mid, uprate), np.arange(mid + uprate, up1.shape[1], uprate)))
        k1 = up1[:, np.int_(downs)]
        k2 = up2[:, np.int_(downs)]
        k3 = up3[:, np.int_(downs)]

    return k1, k2, k3


# Resize the vector
def resize(orig, newSize, align=np.array([0, 0]), padding=0):
    sizem1n1 = orig.shape
    if isinstance(newSize, int):
        newSize = np.array([newSize, newSize])
    if isinstance(align, int):
        align = np.array([align, align])
    n1 = sizem1n1[0]
    if len(sizem1n1) < 2:
        m1 = 1
    else:
        m1 = sizem1n1[0]
        n1 = sizem1n1[1]
    orig = orig.reshape((m1, n1))
    m2 = int(newSize[0])
    n2 = int(newSize[1])
    m = np.minimum(m1, m2)
    n = np.minimum(n1, n2)
    result = np.ones((m2, n2)) * padding
    start1 = [np.floor((m1 - m) / 2. * (1. + align[0])), np.floor((n1 - n) / 2. * (1. + align[1]))]
    start2 = [np.floor((m2 - m) / 2. * (1. + align[0])), np.floor((n2 - n) / 2. * (1. + align[1]))]
    result[np.arange(int(start2[0]), int(start2[0] + m))[:,None], np.arange(int(start2[1]), int(start2[1] + n))] = \
        orig[np.arange(int(start1[0]), int(start1[0] + m))[:,None], np.arange(int(start1[1]), int(start1[1] + n))]

    return result


# image padding for the convolution
def pad4conv(im, kernelsize, dim=None):
    if not isinstance(kernelsize, list):
        kernelsize = [kernelsize, kernelsize]
    if dim is None:
        dim = 3
    imsize = im.shape
    m = imsize[0]
    n = imsize[1]
    if (kernelsize[0] >= m):
        h = np.floor(m / 2.)
    else:
        h = np.floor(kernelsize[0] / 2)
    if (kernelsize[1] >= n):
        w = np.floor(n / 2.)
    else:
        w = np.floor(kernelsize[1] / 2)
    if h != 0 and dim != 2:
        im = np.vstack((im, np.flipud(im[range(int(m) - int(h), int(m)), :])))
        im = np.vstack((np.flipud(im[range(0, int(h)), :]), im))
    if w != 0 and dim != 1:
        im = np.hstack((im, np.fliplr(im[:, range(int(n) - int(w), int(n))])))
        im = np.hstack((np.fliplr(im[:, range(0, int(w))]), im))
    return im


# Separable image convolution
def separable_conv(im, xkernels, ykernels=None):
    if ykernels is None:
        ykernels = xkernels
    imsize = im.shape
    w1 = pad4conv(im, xkernels.shape[1], 2)
    result = np.zeros(imsize)
    for jj in range(xkernels.shape[0]):
        p = signal.convolve2d(w1, xkernels[jj, :].reshape((1, xkernels.shape[1])), mode='full')
        p = resize(p, imsize)
        w2 = pad4conv(p, ykernels.shape[1], 1)
        p = signal.convolve2d(w2, ykernels[jj, :].reshape((xkernels.shape[1], 1)), mode='full')
        p = resize(p, imsize)
        result = result + p

    return result
