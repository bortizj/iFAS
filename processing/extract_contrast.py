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
import pywt
import cv2


def sme(img, blk_size=np.array([3, 3])):
    """
    Estimates the simple measure of contrast in small patches
    based on
    K. Panetta, C. Gao, and S. Agaian. No reference color image contrast and quality measures. IEEE
    Transactions on Consumer Electronics, 59:643 – 651, 2013.

    img: input image, blk_size: size of the block of analysis
    returns c_img: contrast image, sme: global contrast
    """
    if len(img.shape) > 2:
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype("float32")
    else:
        img_gray = img.astype('float32')

    n = 0
    sme = 0.
    M, N = img_gray.shape
    c_img = np.zeros_like(img_gray)
    for ii in range(0, M - blk_size[0], blk_size[0]):
        for jj in range(0, N - blk_size[1], blk_size[1]):
            values = img_gray[ii:ii + blk_size[0], jj:jj + blk_size[1]]
            Imax = np.maximum(1., np.max(values))
            Imin = np.maximum(1., np.min(values))
            c = np.log(Imax / Imin)
            c_img[ii:ii + blk_size[0], jj:jj + blk_size[1]] = c
            sme += c
            n += 1
    sme = (20. * sme / (blk_size[0] * blk_size[1])) / n

    return sme, c_img


def wme(img, blk_size=np.array([3, 3])):
    """
    Estimates the Weber measure of contrast in small patches
    based on
    S.S. Agaian, K.P. Lentz, and A.M. Grigoryan. A new measure of image enhancement. In Proc. of the
    International Conference on Signal Processing and Communication, pages 19 – 22, 2000.

    img: input image, blk_size: size of the block of analysis
    returns c_img: contrast image, sme: global contrast
    """
    if len(img.shape) > 2:
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype("float32")
    else:
        img_gray = img.astype('float32')

    n = 0
    wme = 0.
    M, N = img_gray.shape
    c_img = np.zeros_like(img_gray)
    for ii in range(0, M - blk_size[0], blk_size[0]):
        for jj in range(0, N - blk_size[1], blk_size[1]):
            values = img_gray[ii:ii + blk_size[0], jj:jj + blk_size[1]]
            Imax = np.maximum(0., np.max(values))
            Imin = np.maximum(1., np.min(values))
            c = np.log((np.abs(Imax - Imin) / (Imin)) + 1)
            c_img[ii:ii + blk_size[0], jj:jj + blk_size[1]] = c
            wme += c
            n += 1
    wme = (20. * wme / (blk_size[0] * blk_size[1])) / n

    return wme, c_img


def mme(img, blk_size=np.array([3, 3])):
    """
    Estimates the Michelson measure of contrast in small patches
    based on
    S.S. Agaian, B. Silver, and K.A. Panetta. Transform coefficient histogram-based image enhancement
    algorithms using contrast entropy. IEEE Transactions on Image Processing, 16:741 – 758, 2007.

    img: input image, blk_size: size of the block of analysis
    returns c_img: contrast image, sme: global contrast
    """
    if len(img.shape) > 2:
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype("float32")
    else:
        img_gray = img.astype('float32')

    n = 0
    mme = 0.
    M, N = img_gray.shape
    c_img = np.zeros_like(img_gray)
    for ii in range(0, M - blk_size[0], blk_size[0]):
        for jj in range(0, N - blk_size[1], blk_size[1]):
            values = img_gray[ii:ii + blk_size[0], jj:jj + blk_size[1]]
            Imax = np.maximum(1., np.max(values))
            Imin = np.min(values)
            c = np.log(((Imax - Imin) / (Imax + Imin)) + 1)
            mme += c
            n += 1
            c_img[ii:ii + blk_size[0], jj:jj + blk_size[1]] = c
    mme = (20. * mme / (blk_size[0] * blk_size[1])) / n

    return mme, c_img


def rme(img, blk_size=np.array([3, 3])):
    """
    Estimates the Root mean squared measure of enhancement in small patches
    based on
    K. Panetta, C. Gao, and S. Agaian. No reference color image contrast and quality measures. IEEE
    Transactions on Consumer Electronics, 59:643 – 651, 2013.

    img: input image, blk_size: size of the block of analysis
    returns c_img: contrast image, sme: global contrast
    """
    if len(img.shape) > 2:
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype("float32")
    else:
        img_gray = img.astype('float32')

    n = 0
    rme = 0.
    M, N = img_gray.shape
    c_img = np.zeros_like(img_gray)
    for ii in range(0, M - blk_size[0], blk_size[0]):
        for jj in range(0, N - blk_size[1], blk_size[1]):
            values = img_gray[ii:ii + blk_size[0], jj:jj + blk_size[1]]
            values_center = values[np.int(np.floor(blk_size[0] / 2.)), np.int(np.floor(blk_size[0] / 2.))]
            mean_values = np.mean(values)
            Imin = np.maximum(1., np.abs(values_center - mean_values))
            Imax = np.maximum(1., np.abs(values_center + mean_values))
            if Imax == 1:
                c = 0
            else:
                c = np.abs(np.log(Imin) / np.log(Imax))
                rme += np.sqrt(c)
                n += 1
            c_img[ii:ii + blk_size[0], jj:jj + blk_size[1]] = np.sqrt(c)
    rme = (rme / (blk_size[0] * blk_size[1])) / n

    return rme, c_img


def contrast_peli(img, wname='db1'):
    """
    Estimates the Peli's measured of contrast using the wavelet transform
    based on
    E. Provenzi and V. Caselles. A wavelet perspective on variational perceptually-inspired color enhancement.
    International Journal of Computer Vision, 106:153 – 171, 2014.

    img: input image, wname: wavelet name
    returns c_img: contrast image, sme: global contrast
    """
    alpha = 0.1
    if len(img.shape) > 2:
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype("float32")
    else:
        img_gray = img.astype('float32')

    cA = img_gray
    CA = img_gray
    c_img = np.zeros(img_gray.shape)
    levels = pywt.dwt_max_level(np.int_(np.min(img_gray.shape)), pywt.Wavelet(wname).dec_len)
    c = 0.

    for kk in range(levels-1):
        (cA, (cH, cV, cD)) = pywt.dwt2(cA, wname, 'sym')
        (CA, (CH, CV, CD)) = pywt.swt2(CA, wname, level=1)[0]
        app = alpha * np.mean(cA) + (1 - alpha) * cA
        App = alpha * np.mean(CA) + (1 - alpha) * CA
        th = np.max(cH) / 10.
        cH[np.where(cH<th)] = 0.
        tv = np.max(cV) / 10.
        cV[np.where(cV < tv)] = 0.
        td = np.max(cD) / 10.
        cD[np.where(cD < td)] = 0.

        # Contrast value with decimated wavelet
        contrastH = np.abs(cH) / np.abs(app)
        contrastV = np.abs(cV) / np.abs(app)
        contrastD = np.abs(cD) / np.abs(app)
        m0 = np.mean(contrastH[np.where(contrastH > 0.)])
        m1 = np.mean(contrastV[np.where(contrastV > 0.)])
        m2 = np.mean(contrastD[np.where(contrastD > 0.)])

        if np.isnan(m0):
            m0 = 0.
        if np.isnan(m1):
            m1 = 0.
        if np.isnan(m2):
            m2 = 0.
        c += (m0 + m1 + m2) / 3.

        # Contrast map computed using non decimated wavelet
        Th = np.max(CH) / 10.
        CH[np.where(CH < Th)] = 0.
        Tv = np.max(CV) / 10.
        CV[np.where(CV < Tv)] = 0.
        Td = np.max(CD) / 10.
        CD[np.where(CD < Td)] = 0.
        ContrastH = np.abs(CH) / np.abs(App)
        ContrastV = np.abs(CV) / np.abs(App)
        ContrastD = np.abs(CD) / np.abs(App)
        c_img += (ContrastH + ContrastV + ContrastD) / 3.

    c = c / levels
    C = c_img / levels

    return c, c_img
