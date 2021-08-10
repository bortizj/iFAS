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

# This module contains the translation from Matlab to Python from the iCam module 
# https://www.rit.edu/cos/colorscience/re_iCAM06.php

# image distance in iCAM module
def idl_dist(m, n=None):
    if n is None:
        n = m
    x = np.arange(n)
    x = np.power(np.minimum(x, (n - x)), 2)
    a = np.zeros((m, n))
    for ii in range(0, int(m / 2)):
        y = np.sqrt(x + ii ** 2)
        a[ii, :] = y
        if ii != 0:
            a[m - ii, :] = y

    return a


# Implementation for fast bilateral filter in iCAM module
def fast_bilateral_filter(img):
    if np.min(img.shape) < 1024:
        z = 2
    else:
        z = 4
    img[np.where(img < 0.0001)] = 0.0001
    logimg = np.log10(img)
    base_layer = piecewise_bilateral_filter(logimg, z)
    base_layer = np.minimum(base_layer, np.max(logimg))
    detail_layer = logimg - base_layer
    detail_layer[np.where(detail_layer > 12.)] = 0.
    base_layer = np.power(10, base_layer)
    detail_layer = np.power(10, detail_layer)

    return base_layer, detail_layer


# Implementation for piecewise bilateral filter in iCAM module
def piecewise_bilateral_filter(imageIn, z):
    imSize = imageIn.shape
    xDim = imSize[1]
    yDim = imSize[0]
    sigma_s = 2. * xDim / z / 100.
    sigma_r = 0.35
    maxI = np.max(imageIn)
    minI = np.min(imageIn)
    nSeg = (maxI - minI) / sigma_r
    inSeg = np.int(round(nSeg))
    distMap = idl_dist(yDim, xDim)
    kernel = np.exp(-1. * np.power(distMap / sigma_s, 2))
    kernel = kernel / kernel[0, 0]
    fs = np.maximum(np.real(np.fft.fft(kernel)), 0)
    fs = fs / fs[0, 0]
    Ip = imageIn[0::z, 0::z]
    fsp = fs[0::z, 0::z]
    imageOut = np.zeros(imSize)

    for jj in range(0, inSeg):
        value_i = minI + jj * (maxI - minI) / inSeg
        jGp = np.exp((-1. / 2.) * np.power((Ip - value_i) / sigma_r, 2))
        jKp = np.maximum(np.real(np.fft.ifft(np.fft.fft(jGp) * fsp)), 0.0000000001)
        jHp = jGp * Ip
        sjHp = np.real(np.fft.ifft(np.fft.fft(jHp) * fsp))
        jJp = sjHp / jKp
        m, n = jJp.shape
        jJ = cv2.resize(jJp, (int(z * n), int(z * m)), interpolation=cv2.INTER_LINEAR)

        jJ = jJ[0:yDim, 0:xDim]
        intW = np.maximum(np.ones(imSize) - np.abs(imageIn - value_i) * (inSeg) / (maxI - minI), 0)
        imageOut = imageOut + jJ * intW

    return imageOut



def RGB2iCAM(RGB):
    max_L = 20000.
    p = 0.7
    gamma_value = 1.
    M = np.array([
        [0.412424, 0.212656, 0.0193324], [0.357579, 0.715158, 0.119193], [0.180464, 0.0721856, 0.950444]
        ])
    X = M[0, 0] * RGB[::, ::, 0] + M[0, 1] * RGB[::, ::, 1] + M[0, 2] * RGB[::, ::, 2]
    Y = M[1, 0] * RGB[::, ::, 0] + M[1, 1] * RGB[::, ::, 1] + M[1, 2] * RGB[::, ::, 2]
    Z = M[2, 0] * RGB[::, ::, 0] + M[2, 1] * RGB[::, ::, 1] + M[2, 2] * RGB[::, ::, 2]
    XYZimg = np.dstack((X, Y, Z))
    XYZimg = XYZimg / np.max(XYZimg[::, ::, 1]) * max_L
    XYZimg[np.where(XYZimg < 0.00000001)] = 0.00000001

    base_imgX, detail_imgX = fast_bilateral_filter(XYZimg[::, ::, 0])
    base_imgY, detail_imgY = fast_bilateral_filter(XYZimg[::, ::, 1])
    base_imgZ, detail_imgZ = fast_bilateral_filter(XYZimg[::, ::, 2])

    base_img = np.dstack((base_imgX,base_imgY,base_imgZ))
    detail_img = np.dstack((detail_imgX, detail_imgY, detail_imgZ))

    white = iCAM06_blur(XYZimg, 2)
    XYZ_adapt = iCAM06_CAT(base_img, white)
    white = iCAM06_blur(XYZimg, 3)
    XYZ_tc = iCAM06_TC(XYZ_adapt, white, p)
    XYZ_d = XYZ_tc * iCAM06_LocalContrast(detail_img, base_img)

    return iCAM06_IPT(XYZ_d, base_img, gamma_value)


# Extra functions necessary to compute color spaces, e.g., intermediate color spaces
def iCAM06_blur(img, d):
    sy,sx,sz = img.shape
    m = np.minimum(sy,sx)
    if m<64:
        z = 1
    elif m<256:
        z = 2
    elif m<512:
        z = 4
    elif m<1024:
        z = 8
    elif m<2056:
        z = 16
    else:
        z = 32

    img = img[0::z,0::z,:]
    imSize = img.shape
    xDim = imSize[1]
    yDim = imSize[0]

    Y = np.zeros((2*yDim, 2*xDim,3))
    Y[int(round(yDim/2)):int(round(yDim/2)+yDim),int(round(xDim/2)):int(round(xDim/2)+xDim), ::] = img
    Y[int(round(yDim/2)):int(round(yDim/2)+yDim),0:int(round(xDim/2)), ::] = img[::, 0:int(round(xDim/2)), ::]
    Y[int(round(yDim/2)):int(round(yDim/2)+yDim),int(round(xDim/2)+xDim):int(2*xDim), ::] = (
        img[::, int(round(xDim/2)):int(xDim), ::]
        )
    Y[0:int(round(yDim/2)),int(round(xDim/2)):int(round(xDim/2)+xDim), ::] = img[0:int(round(yDim/2)), ::, ::]
    Y[int(round(yDim/2)+yDim):int(2*yDim),int(round(xDim/2)):int(round(xDim/2)+xDim), ::] = (
        img[int(round(yDim/2)):int(yDim), ::, ::]
        )
    Y[0:int(round(yDim/2)),0:int(round(xDim/2)), ::] = img[0:int(round(yDim/2)),0:int(round(xDim/2)), ::]
    Y[0:int(round(yDim/2)),int(round(xDim/2)+xDim):int(2*xDim), ::] = (
        img[0:int(round(yDim/2)),int(round(xDim/2)):int(xDim), ::]
        )
    Y[int(round(yDim/2)+yDim):int(2*yDim), int(round(xDim/2)+xDim):int(2*xDim), ::] = (
        img[int(round(yDim/2)):int(yDim),int(round(xDim/2)):int(xDim), ::]
        )
    Y[int(round(yDim/2)+yDim):int(2*yDim), 0:int(round(xDim/2)), ::] = (
        img[int(round(yDim/2)):int(yDim),0:int(round(xDim/2)), ::]
        )

    distMap = idl_dist(Y.shape[0], Y.shape[1])
    Dim = np.maximum(xDim, yDim)
    kernel = np.exp(-1*np.power(distMap/(Dim/d),2))
    filter = np.maximum(np.real(np.fft.fft(kernel)),0)
    filter = filter / filter[0,0]

    whiteX = np.maximum(np.real(np.fft.ifft(np.fft.fft(Y[:,:,0])*filter)),0)
    whiteY = np.maximum(np.real(np.fft.ifft(np.fft.fft(Y[:,:,1])*filter)),0)
    whiteZ = np.maximum(np.real(np.fft.ifft(np.fft.fft(Y[:,:,2])*filter)),0)

    white = np.dstack((
        whiteX[int(round(yDim/2)):int(round(yDim/2)+yDim),int(round(xDim/2)):int(round(xDim/2)+xDim)], 
        whiteY[int(round(yDim/2)):int(round(yDim/2)+yDim),int(round(xDim/2)):int(round(xDim/2)+xDim)],
        whiteZ[int(round(yDim/2)):int(round(yDim/2)+yDim),int(round(xDim/2)):int(round(xDim/2)+xDim)]
        ))

    m, n, __ = white.shape
    white = cv2.resize(white, (int(z * n), int(z * m)), interpolation=cv2.INTER_NEAREST)

    return white[0:sy, 0:sx, ::]


def iCAM06_CAT(XYZimg, white):
    M = np.array([[0.7328, 0.4296, -0.1624], [-0.7036, 1.6974, 0.0061], [0.0030, 0.0136, 0.9834]])
    Mi = np.linalg.inv(M.T)
    R = M[0, 0] * XYZimg[::, ::, 0] + M[0, 1] * XYZimg[::, ::, 1] + M[0, 2] * XYZimg[::, ::, 2]
    G = M[1, 0] * XYZimg[::, ::, 0] + M[1, 1] * XYZimg[::, ::, 1] + M[1, 2] * XYZimg[::, ::, 2]
    B = M[2, 0] * XYZimg[::, ::, 0] + M[2, 1] * XYZimg[::, ::, 1] + M[2, 2] * XYZimg[::, ::, 2]
    RGB_img = np.dstack((R,G,B))
    R = M[0, 0] * white[::, ::, 0] + M[0, 1] * white[::, ::, 1] + M[0, 2] * white[::, ::, 2]
    G = M[1, 0] * white[::, ::, 0] + M[1, 1] * white[::, ::, 1] + M[1, 2] * white[::, ::, 2]
    B = M[2, 0] * white[::, ::, 0] + M[2, 1] * white[::, ::, 1] + M[2, 2] * white[::, ::, 2]
    RGB_white = np.dstack((R,G,B))
    xyz_d65 = np.array([ 95.05,  100.0, 108.88])
    R = M[0, 0] * xyz_d65[0] + M[0, 1] * xyz_d65[1] + M[0, 2] * xyz_d65[2]
    G = M[1, 0] * xyz_d65[0] + M[1, 1] * xyz_d65[1] + M[1, 2] * xyz_d65[2]
    B = M[2, 0] * xyz_d65[0] + M[2, 1] * xyz_d65[1] + M[2, 2] * xyz_d65[2]
    La = 0.2 * white[::, ::, 1]
    F = 1
    D = 0.3 * F * (1 - (1 / 3.6) * np.exp(-1 * (La - 42) / 92))
    RGB_white = RGB_white + 0.0000001
    Rc = (D * R/RGB_white[::, ::,0] + 1 - D) * RGB_img[::, ::, 0]
    Gc = (D * G/RGB_white[::, ::,1] + 1 - D) * RGB_img[::, ::, 1]
    Bc = (D * B/RGB_white[::, ::,2] + 1 - D) * RGB_img[::, ::, 2]
    adaptImage = np.dstack((Rc, Gc, Bc))
    X = Mi[0, 0] * adaptImage[::, ::, 0] + Mi[0, 1] * adaptImage[::, ::, 1] + Mi[0, 2] * adaptImage[::, ::, 2]
    Y = Mi[1, 0] * adaptImage[::, ::, 0] + Mi[1, 1] * adaptImage[::, ::, 1] + Mi[1, 2] * adaptImage[::, ::, 2]
    Z = Mi[2, 0] * adaptImage[::, ::, 0] + Mi[2, 1] * adaptImage[::, ::, 1] + Mi[2, 2] * adaptImage[::, ::, 2]

    return np.dstack((X, Y, Z))


def iCAM06_TC(XYZ_adapt, white_img, p):
    M = np.array([ [0.38971, 0.68898, -0.07868],[-0.22981, 1.18340,  0.04641],[ 0.00000, 0.00000,  1.00000]])
    Mi = np.linalg.inv(M.T)
    R = M[0, 0] * XYZ_adapt[::, ::, 0] + M[0, 1] * XYZ_adapt[::, ::, 1] + M[0, 2] * XYZ_adapt[::, ::, 2]
    G = M[1, 0] * XYZ_adapt[::, ::, 0] + M[1, 1] * XYZ_adapt[::, ::, 1] + M[1, 2] * XYZ_adapt[::, ::, 2]
    B = M[2, 0] * XYZ_adapt[::, ::, 0] + M[2, 1] * XYZ_adapt[::, ::, 1] + M[2, 2] * XYZ_adapt[::, ::, 2]
    RGB_img = np.dstack((R, G, B))
    La = 0.2 * white_img[::, ::, 1]
    k = 1. / (5. * La + 1)
    FL = 0.2 * np.power(k, 4) * (5 * La) + 0.1 * np.power(1 - np.power(k, 4), 2) * np.power(5 * La, 1 / 3.)
    FL = np.dstack((FL, FL, FL))
    white_3img = np.dstack((white_img[::, ::, 1], white_img[::, ::, 1], white_img[::, ::, 1]))
    sign_RGB = np.sign(RGB_img)
    RGB_c = (
        sign_RGB * (
            (400 * np.power(FL * np.abs(RGB_img) / white_3img,p)) / 
            (27.13 + np.power(FL * np.abs(RGB_img) / white_3img,p))
            ) + .1
        )
    Las = 2.26 * La
    j = 0.00001 / (5 * Las / 2.26+0.00001)
    FLS = (
        3800 * np.power(j, 2) * (5 * Las / 2.26) + 0.2 * np.power(1 - np.power(j, 2), 4) * np.power(5 * Las /2.26, 1/6.)
        )
    Sw = np.max(5 * La)
    S = np.abs(XYZ_adapt[::, ::, 1])
    Bs = 0.5 / (1 + .3 * np.power((5 * Las / 2.26) * (S / Sw), 3)) + 0.5 / (1 + 5 * (5 * Las /2.26))
    As = 3.05 * Bs * (((400 * np.power(FLS * (S / Sw), p))/  (27.13 + np.power(FLS * (S / Sw), p)) )) + .03
    As = np.dstack((As, As, As))
    RGB_c = RGB_c + As
    R = Mi[0, 0] * RGB_c[::, ::, 0] + Mi[0, 1] * RGB_c[::, ::, 1] + Mi[0, 2] * RGB_c[::, ::, 2]
    G = Mi[1, 0] * RGB_c[::, ::, 0] + Mi[1, 1] * RGB_c[::, ::, 1] + Mi[1, 2] * RGB_c[::, ::, 2]
    B = Mi[2, 0] * RGB_c[::, ::, 0] + Mi[2, 1] * RGB_c[::, ::, 1] + Mi[2, 2] * RGB_c[::, ::, 2]

    return np.dstack((R, G, B))


def iCAM06_LocalContrast(detail, base_img):
    La = 0.2 * base_img[::, ::, 1]
    k = 1. / (5 * La + 1)
    FL = 0.2 * np.power(k, 4) * (5 * La) + 0.1 * np.power(1 - np.power(k, 4), 2) * np.power(5 * La, 1/3.)
    FL = np.dstack((FL, FL, FL))

    return np.power(detail, np.power((FL + 0.8), .25))


def iCAM06_IPT(XYZ_img, base_img, gamma):
    xyz2lms = np.array([[.4002, .7077, -.0807],[-.2280, 1.1500, .0612],[.0, .0, .9184]]).T
    iptMat = np.array([[ 0.4000, 0.4000, 0.2000],[ 4.4550,-4.8510, 0.3960],[ 0.8056, 0.3572,-1.1628] ]).T
    L = xyz2lms[0, 0] * XYZ_img[::, ::, 0] + xyz2lms[0, 1] * XYZ_img[::, ::, 1] + xyz2lms[0, 2] * XYZ_img[::, ::, 2]
    M = xyz2lms[1, 0] * XYZ_img[::, ::, 0] + xyz2lms[1, 1] * XYZ_img[::, ::, 1] + xyz2lms[1, 2] * XYZ_img[::, ::, 2]
    S = xyz2lms[2, 0] * XYZ_img[::, ::, 0] + xyz2lms[2, 1] * XYZ_img[::, ::, 1] + xyz2lms[2, 2] * XYZ_img[::, ::, 2]
    lms_img = np.dstack((L, M, S))
    lms_img = np.power(np.abs(lms_img),.43)
    i = iptMat[0, 0] * lms_img[::, ::, 0] + iptMat[0, 1] * lms_img[::, ::, 1] + iptMat[0, 2] * lms_img[::, ::, 2]
    p = iptMat[1, 0] * lms_img[::, ::, 0] + iptMat[1, 1] * lms_img[::, ::, 1] + iptMat[1, 2] * lms_img[::, ::, 2]
    t = iptMat[2, 0] * lms_img[::, ::, 0] + iptMat[2, 1] * lms_img[::, ::, 1] + iptMat[2, 2] * lms_img[::, ::, 2]
    ipt_img = np.dstack((i, p, t))
    c = np.sqrt(np.power(ipt_img[::, ::, 1],2)+np.power(ipt_img[::, ::, 2], 2))
    La = 0.2*base_img[::, ::, 1]
    k = 1 / (5 * La + 1)
    FL = 0.2 * np.power(k, 4) * (5 * La) + 0.1 * np.power(1 - np.power(k, 4), 2) * np.power(5 * La, 1/3)
    ipt_img[::, ::, 1] = (
        ipt_img[::, ::, 1] * 
        (np.power(FL + 1, .15) * ((1.29 * np.power(c, 2) - 0.27 * c + 0.42) / (np.power(c, 2) - 0.31 * c + 0.42)))
        )
    ipt_img[::, ::, 2] = (
        ipt_img[::, ::, 2] * 
        (np.power(FL + 1, .15) * ((1.29 * np.power(c, 2) - 0.27 * c + 0.42) / (np.power(c, 2) - 0.31 * c + 0.42)))
        )
    max_i = np.max(ipt_img[::, ::, 0])
    ipt_img[::, ::, 0] = ipt_img[::, ::, 0] / max_i
    ipt_img[::, ::, 0] = np.power(ipt_img[::, ::, 0], (gamma))
    ipt_img[::, ::, 0] = ipt_img[::, ::, 0] * max_i

    return ipt_img
