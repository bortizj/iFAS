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

from processing import img_misc
from gui import ifas_misc
import numpy as np
from scipy import io
from scipy import interpolate
import cv2
from pathlib import Path


FILE_PATH = Path(__file__).parent.absolute()


# Color transfrom BGR to L alpha beta
def bgr_to_l_alpha_beta(img):
    RGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    A = np.array([[0.3811, 0.5783, 0.0402], [0.1967, 0.7244, 0.0782], [0.0241, 0.1288, 0.8444]])
    L = np.log(A[0, 0] * RGB[:, :, 0] + A[0, 1] * RGB[:, :, 1] + A[0, 2] * RGB[:, :, 2] + 1)
    M = np.log(A[1, 0] * RGB[:, :, 0] + A[1, 1] * RGB[:, :, 1] + A[1, 2] * RGB[:, :, 2] + 1)
    S = np.log(A[2, 0] * RGB[:, :, 0] + A[2, 1] * RGB[:, :, 1] + A[2, 2] * RGB[:, :, 2] + 1)

    A = np.dot(
        np.array([[1. / np.sqrt(3.), 0., 0.], [0., 1. / np.sqrt(6.), 0.], [0., 0., 1. / np.sqrt(2.)]]),
        np.array([[1.,  1.,  1.], [1.,  1., -2.], [1., -1.,  0.]])
        )

    l = A[0, 0] * L + A[0, 1] * M + A[0, 2] * S
    alpha = A[1, 0] * L + A[1, 1] * M + A[1, 2] * S
    beta = A[2, 0] * L + A[2, 1] * M + A[2, 2] * S

    return np.dstack((l, alpha, beta))


# Some linear color transforms not available in opencv
def linear_color_transform(img, tr_type="rgb_to_xyz"):
    in_img = img.astype("float32")
    if isinstance(tr_type, str):
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
        else:
            tr_mat = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    else:
        tr_mat = tr_type

    out1 = tr_mat[0, 0] * in_img[:, :, 0] + tr_mat[0, 1] * in_img[:, :, 1] + tr_mat[0, 2] * in_img[:, :, 2]
    out2 = tr_mat[1, 0] * in_img[:, :, 0] + tr_mat[1, 1] * in_img[:, :, 1] + tr_mat[1, 2] * in_img[:, :, 2]
    out3 = tr_mat[2, 0] * in_img[:, :, 0] + tr_mat[2, 1] * in_img[:, :, 1] + tr_mat[2, 2] * in_img[:, :, 2]

    return np.dstack((out1, out2, out3))


# XYZ to xyY color space conversion 
def XYZ_to_xyY(XYZ):
    denom = np.sum(XYZ, 2)
    denom[denom == 0.] = 1.
    x = XYZ[:, :, 0] / denom
    y = XYZ[:, :, 1] / denom
    return np.dstack((x, y, XYZ[:, :, 1]))


# BGR to OSA UCS color space
def bgr2Ljg(BGR):
    RGB = cv2.cvtColor(BGR, cv2.COLOR_BGR2RGB)
    M_XYZToRGB = np.array([[0.799, 0.4194, -0.1648], [-0.4493, 1.3265, 0.0927], [-0.1149, 0.3394, 0.7170]])
    M_RGBToXYZ = np.linalg.inv(M_XYZToRGB)
    X = M_RGBToXYZ[0, 0] * RGB[::, ::, 0] + M_RGBToXYZ[0, 1] * RGB[::, ::, 1] + M_RGBToXYZ[0, 2] * RGB[::, ::, 2]
    Y = M_RGBToXYZ[1, 0] * RGB[::, ::, 0] + M_RGBToXYZ[1, 1] * RGB[::, ::, 1] + M_RGBToXYZ[1, 2] * RGB[::, ::, 2]
    Z = M_RGBToXYZ[2, 0] * RGB[::, ::, 0] + M_RGBToXYZ[2, 1] * RGB[::, ::, 1] + M_RGBToXYZ[2, 2] * RGB[::, ::, 2]
    XYZ = np.dstack((X, Y, Z))

    RGB3 = np.power(RGB, 1./3.)
    xyY = XYZ_to_xyY(XYZ)
    x= xyY[:, :, 0]
    y= xyY[:, :, 1]
    Y= xyY[:, :, 2]
    Y0 = Y * (4.4934 * np.power(x, 2) + 4.3034 * np.power(y, 2) - 4.276 * (x * y) - 1.3744 * x - 2.5643 * y + 1.8103)
    scriptL = np.zeros(Y0.shape)
    index = np.where(Y0 > 30)

    if index[0].any():
        scriptL[index] = 5.9 * (
            (np.sign(Y0[index]) * np.power(np.abs(Y0[index]), 1. / 3.)) - (2. / 3.) + 
            0.042 * (np.power(np.abs(Y0[index] - 30), 1./3.))
            )
    index = np.where(Y0 <= 30)

    if index[0].any():
        scriptL[index] = 5.9 * (
            (np.sign(Y0[index]) * np.power(np.abs(Y0[index]), 1. / 3.)) - (2. / 3.) - 
            0.042 * (np.power(np.abs(Y0[index] - 30), 1./3.))
            )

    C = scriptL / (5.9 * (np.sign(Y0) * np.power(np.abs(Y0), 1. / 3.)) - (2. / 3.))
    L = (scriptL - 14.4) / np.sqrt(2.)
    j = C * (1.7 * RGB3[:, :, 0] + 8 * RGB3[:, :, 1] - 9.7 * RGB3[:, :, 2])
    g = C * (-13.7 * RGB3[:, :, 0] + 17.7 * RGB3[:, :, 1] - 4 * RGB3[:, :, 2])

    return np.real(np.dstack((L, j, g)))


# SRGB to XYZ color transform
def SRGB_to_XYZ(img_SRGB):
    R = inv_gamma(img_SRGB[::, ::, 0] / 255.)
    G = inv_gamma(img_SRGB[::, ::, 1] / 255.)
    B = inv_gamma(img_SRGB[::, ::, 2] / 255.)
    T = np.linalg.inv(np.array([[3.2406, -1.5372, -0.4986], [-0.9689, 1.8758, 0.0415], [0.0557, -0.2040, 1.057]]))
    X = T[0,0] * R + T[0,1] * G + T[0,2] * B
    Y = T[1,0] * R + T[1,1] * G + T[1,2] * B
    Z = T[2,0] * R + T[2,1] * G + T[2,2] * B

    return np.dstack((X, Y, Z))


# Gamma correction inversion 
def inv_gamma(Rp):
    R = np.zeros(Rp.shape)
    ii = np.where(Rp <= 0.0404482362771076)
    R[ii] = Rp[ii] / 12.92
    ii = np.where(Rp > 0.0404482362771076)
    R[ii] = np.real(np.power((Rp[ii] + 0.055) / 1.055, 2.4))

    return R


# Simple spatial cielab
def scielab_simple(samp_per_deg, image):
    T = np.array(
        [[278.7336, 721.8031, -106.5520], [-448.7736, 289.8056, 77.1569], [85.9513, -589.9859, 501.1089]]
        ) / 1000.
    opp = linear_color_transform(image, T)
    [k1, k2, k3] = img_misc.separable_filters(samp_per_deg, 3)
    p1 = img_misc.separable_conv(opp[::, ::, 0], k1, np.abs(k1))
    p2 = img_misc.separable_conv(opp[::, ::, 1], k2, np.abs(k2))
    p3 = img_misc.separable_conv(opp[::, ::, 2], k3, np.abs(k3))
    opp = np.dstack((p1, p2, p3))
    xyz = linear_color_transform(opp, np.linalg.inv(T))

    return xyz


def XYZ_to_LAB2000HL(XYZ):
    LAB = XYZ_to_LAB(XYZ)
    return LAB_2_LAB2000HL(LAB)


def XYZ_to_LAB(XYZ):
    WhitePoint = np.array([0.950456, 1, 1.088754])
    X = XYZ[:, :, 0] / WhitePoint[0]
    Y = XYZ[:, :, 1] / WhitePoint[1]
    Z = XYZ[:, :, 2] / WhitePoint[2]
    fX = ifas_misc.lum_fun(X)
    fY = ifas_misc.lum_fun(Y)
    fZ = ifas_misc.lum_fun(Z)
    L = 116. * fY - 16.
    a = 500. * (fX - fY)
    b = 200. * (fY - fZ)

    return np.dstack((L, a, b))


def LAB_2_LAB2000HL(LAB):
    L = LAB[:, :, 0]
    a = LAB[:, :, 1]
    b = LAB[:, :, 2]
    L[L < 0] = 0
    L[L > 100] = 100
    a[a < -128] = -128
    a[a > 128] = 128
    b[b < -128] = -128
    b[b > 128] = 128
    mat_contents = io.loadmat(str(FILE_PATH.joinpath('LAB2000HL.mat')))
    RegularGrid = mat_contents['RegularGrid']
    Lgrid = mat_contents['L']
    fL = interpolate.interp1d(np.arange(0, 100 + 0.001, 0.001), Lgrid)
    L2000HL = fL(L).reshape(L.shape)
    x = np.arange(-128, 129)
    y = np.arange(-128, 129)
    fa = interpolate.RectBivariateSpline(x, y, RegularGrid[::, ::, 0])
    a2000HL = fa.ev(a, b)
    fb = interpolate.RectBivariateSpline(x, y, RegularGrid[::, ::, 1])
    b2000HL = fb.ev(a, b)

    return np.dstack((L2000HL, a2000HL, b2000HL))
