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


from processing import img_misc, color_transform
from fidelity_measures import fidelity_misc
import numpy as np
import cv2


def delta_e(img_ref, img_tst):
    """
    # Delta E color difference -> http://zschuessler.github.io/DeltaE/learn/
    """
    lab_ref = cv2.cvtColor((img_ref / 255).astype("float32"), cv2.COLOR_BGR2LAB)
    lab_tst = cv2.cvtColor((img_tst / 255).astype("float32"), cv2.COLOR_BGR2LAB)
    img_de = np.sqrt(np.sum(np.power(lab_ref - lab_tst, 2), axis=2))
    de = np.mean(img_de)

    return de


def diff_hue_saturation(img_ref, img_tst):
    """
    Y. Ming, L. Huijuan, G. Yingchun, and Z. Dongming. A method for reduced-reference color image
    quality assessment. In Proc. of the International Congress on Image and Signal Processing, pages 1 –
    5, 2009.
    """
    w1 = 0.3
    w2 = 0.1
    hsv_ref = cv2.cvtColor(img_ref, cv2.COLOR_BGR2HSV)
    hsv_tst = cv2.cvtColor(img_tst, cv2.COLOR_BGR2HSV)

    delta_h = np.abs(np.mean(hsv_ref[::, ::, 0]) - np.mean(hsv_tst[::, ::, 0]))
    delta_s = np.abs(np.mean(hsv_ref[::, ::, 1]) - np.mean(hsv_tst[::, ::, 1]))
    delta_hs = w1 * delta_h + w2 * delta_s
    

    return delta_hs


def adaptive_image_difference(img_ref, img_tst, sizewin=5):
    """
    U. Rajashekar, Z. Wang, and E.P. Simoncelli. Quantifying color image distortions based on adaptive
    spatio-chromatic signal decompositions. In Proc. of the IEEE International Conference on Image
    Processing, pages 2213 – 2216, 2009.
    """
    m, n, __ = img_ref.shape
    img_ascd = np.zeros((m, n))
    half_blk = int(np.floor(sizewin / 2))
    samples = 3 * sizewin * sizewin
    A = np.zeros((samples, 6))
    rgb_vec_tst = np.zeros((samples, 1))
    WA = 0.1 * np.eye(6)
    WA[5, 5] = 0.5

    for ii in range(half_blk, m - half_blk, sizewin):
        for jj in range(half_blk, n - half_blk, sizewin):
            temp_ref = img_ref[ii - half_blk:ii + half_blk + 1, jj - half_blk:jj + half_blk + 1, :]
            temp_tst = img_tst[ii - half_blk:ii + half_blk + 1, jj - half_blk:jj + half_blk + 1, :]

            a1 = temp_ref[::, ::, 0]
            A[0:samples:3, 0] = a1.ravel()
            a2 = temp_ref[::, ::, 1]
            A[1:samples:3, 1] = a2.ravel()
            a3 = temp_ref[:,:, 2]
            A[1:samples:3, 2] = a3.ravel()

            rgb_vec_ref = A[::, 0] + A[::, 1] + A[::, 2]
            rgb_vec_tst[0:samples:3, 0] = temp_tst[::, ::, 0].ravel()
            rgb_vec_tst[1:samples:3, 0] = temp_tst[::, ::, 1].ravel()
            rgb_vec_tst[2:samples:3, 0] = temp_tst[::, ::, 2].ravel()

            l = np.sum(temp_ref, 2) / 3
            A[0:samples:3, 3] = l.ravel()
            A[1:samples:3, 3] = l.ravel()
            A[2:samples:3, 3] = l.ravel()
            A[:, 4] = rgb_vec_ref - A[:, 3]
            A[0:samples:3, 5] = l.ravel() * (a3.ravel() - a2.ravel())
            A[1:samples:3, 5] = l.ravel() * (a1.ravel() - a3.ravel())
            A[2:samples:3, 5] = l.ravel() * (a2.ravel() - a1.ravel())

            An = np.tile(np.sqrt(np.sum(np.power(A, 2), 0)), (A.shape[0], 1))
            idx = np.where(A == 0)
            An[idx] = 1.
            A = A / An
            A[idx] = 0.

            delta_rgb = rgb_vec_ref - rgb_vec_tst
            delta_ca = np.linalg.solve((np.power(WA, 2) + np.dot(A.T, A)),(np.dot(A.T, delta_rgb)))
            img_ascd[ii, jj] = \
                np.sum(np.power(np.dot(WA, delta_ca), 2)) + np.sum(np.power(delta_rgb - np.dot(A, delta_ca), 2))

    delta_ascd = np.mean(img_ascd[half_blk:m - half_blk,half_blk:n - half_blk])

    return delta_ascd


def delta_e2000(img_ref, img_tst):
    """
    # Delta E 2000 color difference -> http://zschuessler.github.io/DeltaE/learn/
    """
    kl, kc, kh = 1, 1, 1
    lab_ref = cv2.cvtColor((img_ref / 255).astype("float32"), cv2.COLOR_BGR2LAB)
    lab_tst = cv2.cvtColor((img_tst / 255).astype("float32"), cv2.COLOR_BGR2LAB)

    l_ref = lab_ref[::, ::, 0]
    a_ref = lab_ref[::, ::, 1]
    b_ref = lab_ref[::, ::, 2]
    c_ref = np.sqrt(np.power(a_ref, 2) + np.power(b_ref, 2))

    l_tst = lab_tst[:, :, 0]
    a_tst = lab_tst[:, :, 1]
    b_tst = lab_tst[:, :, 2]
    c_tst = np.sqrt(np.power(a_tst, 2) + np.power(b_tst, 2))

    c_avg = (c_ref + c_tst) / 2.
    g = 0.5 * (1 - np.sqrt(np.power(c_avg, 7) / (np.power(c_avg, 7) + 25. ** 7)))

    ap_ref = (1 + g) * a_ref
    ap_tst = (1 + g) * a_tst

    cp_ref = np.sqrt(np.power(ap_ref, 2) + np.power(b_ref, 2))
    cp_tst = np.sqrt(np.power(ap_tst, 2) + np.power(b_tst, 2))

    cp_prod = (cp_ref * cp_tst)
    zcidx = np.where(cp_prod == 0)

    hp_ref = np.arctan2(b_ref, ap_ref)
    hp_ref = hp_ref + 2 * np.pi * (hp_ref < 0)
    hp_ref[(np.abs(ap_ref) + np.abs(b_ref)) == 0] = 0

    hp_tst = np.arctan2(b_tst, ap_tst)
    hp_tst = hp_tst + 2 * np.pi * (hp_tst < 0)
    hp_tst[(np.abs(ap_tst) + np.abs(b_tst)) == 0] = 0

    dif_l = (l_ref - l_tst)
    dif_cp = (cp_ref - cp_tst)
    dif_hp = (hp_ref - hp_tst)
    dif_hp = dif_hp - 2 * np.pi * (dif_hp > np.pi)
    dif_hp = dif_hp + 2 * np.pi * (dif_hp < (-np.pi))
    dif_hp[zcidx] = 0
    dif_h = 2 * np.sqrt(cp_prod) * np.sin(dif_hp / 2)

    lp = (l_ref + l_tst) / 2
    cp = (cp_ref + cp_tst) / 2
    hp = (hp_ref + hp_tst) / 2
    hp = hp - (np.abs(hp_ref - hp_tst) > np.pi) * np.pi
    hp = hp + (hp < 0) * 2 * np.pi
    hp[zcidx] = hp_ref[zcidx] + hp_tst[zcidx]

    lpm502 = np.power((lp - 50), 2)
    sl = 1 + 0.015 * lpm502 / np.sqrt(20 + lpm502)
    sc = 1 + 0.045 * cp
    T = 1 - 0.17 * np.cos(hp - np.pi / 6) + 0.24 * np.cos(2 * hp) + 0.32 * np.cos(3 * hp + np.pi / 30) - \
        0.20 * np.cos(4 * hp - 63 * np.pi / 180)
    sh = 1 + 0.015 * cp * T

    delthetarad = (30 * np.pi / 180) * np.exp(- np.power((180 / np.pi * hp - 275) / 25, 2))
    rc = 2 * np.sqrt(np.power(cp, 7) / (np.power(cp, 7) + 25 ** 7))
    rt = - np.sin(2 * delthetarad) * rc
    klSl = kl * sl
    kcSc = kc * sc
    khSh = kh * sh
    img_de00 = np.sqrt(
        np.power((dif_l / klSl), 2) + np.power((dif_cp / kcSc), 2) + np.power((dif_h / khSh), 2) + 
        rt * (dif_cp / kcSc) * (dif_h / khSh)
        )
    de00 = np.mean(img_de00)

    return de00


def color_mahalanobis(img_ref, img_tst):
    """
    F.H. Imai, N. Tsumura, and Y. Miyake. Perceptual color difference metric for complex images based
    on mahalanobis distance. Journal of Electronic Imaging, 10:385 – 393, 2001.
    """
    lab_ref = cv2.cvtColor((img_ref / 255).astype("float32"), cv2.COLOR_BGR2LAB)
    lab_tst = cv2.cvtColor((img_tst / 255).astype("float32"), cv2.COLOR_BGR2LAB)

    l_ref = lab_ref[::, ::, 0]
    a_ref = lab_ref[::, ::, 1]
    b_ref = lab_ref[::, ::, 2]
    c_ref = np.sqrt(np.power(a_ref, 2) + np.power(b_ref, 2))

    l_tst = lab_tst[:, :, 0]
    a_tst = lab_tst[:, :, 1]
    b_tst = lab_tst[:, :, 2]
    c_tst = np.sqrt(np.power(a_tst, 2) + np.power(b_tst, 2))

    c_avg = (c_ref + c_tst) / 2.
    g = 0.5 * (1 - np.sqrt(np.power(c_avg, 7) / (np.power(c_avg, 7) + 25. ** 7)))

    ap_ref = (1 + g) * a_ref
    ap_tst = (1 + g) * a_tst

    cp_ref = np.sqrt(np.power(ap_ref, 2) + np.power(b_ref, 2))
    cp_tst = np.sqrt(np.power(ap_tst, 2) + np.power(b_tst, 2))

    hp_ref = np.arctan2(b_ref, ap_ref)
    hp_ref = hp_ref + 2 * np.pi * (hp_ref < 0)
    hp_ref[(np.abs(ap_ref) + np.abs(b_ref)) == 0] = 0

    hp_tst = np.arctan2(b_tst, ap_tst)
    hp_tst = hp_tst + 2 * np.pi * (hp_tst < 0)
    hp_tst[(np.abs(ap_tst) + np.abs(b_tst)) == 0] = 0

    num_1 = l_ref.size - 1
    mul = np.mean(l_ref[:])
    muc = np.mean(cp_ref[:])
    muh = np.mean(hp_ref[:])
    sigmaLL = np.sum((l_ref[:] - mul) * (l_ref[:] - mul)) / num_1
    sigmaLC = np.sum((l_ref[:] - mul) * (cp_ref[:] - muc)) / num_1
    sigmaLh = np.sum((l_ref[:] - mul) * (hp_ref[:] - muh)) / num_1
    sigmaCC = np.sum((cp_ref[:] - muc) * (cp_ref[:] - muc)) / num_1
    sigmaCh = np.sum((cp_ref[:] - muc) * (hp_ref[:] - muh)) / num_1
    sigmahh = np.sum((hp_ref[:] - muh) * (hp_ref[:] - muh)) / num_1

    A = np.array([[sigmaLL, sigmaLC, sigmaLh], [sigmaLC, sigmaCC, sigmaCh], [sigmaLh, sigmaCh, sigmahh]])
    A = np.linalg.inv(A)

    deltal = l_ref - l_tst
    deltac = cp_ref - cp_tst
    deltah = hp_ref - hp_tst
    CDM = A[0, 0] * np.power(deltal, 2) + A[1, 1] * np.power(deltac, 2) + A[2, 2] * np.power(deltah, 2) + \
        2 * A[0, 1] * (deltal * deltac) + 2 * A[0, 2] * (deltal * deltah) + 2 * A[1, 2] * (deltac * deltah)
    CDM = np.sqrt(CDM)
    cdm = np.mean(CDM[:])

    return cdm


def jncd_delta_e(img_ref, img_tst):
    """
    C.H. Chou and K.C. Liu. A fidelity metric for assessing visual quality of color images. In Proc. of the
    International Conference on Computer Communications and Networks, pages 1154 – 1159, 2007.
    """
    w = 49
    JNCDLab = 2.3
    window = np.ones((w, w)) / (w * w)
    lab_ref = cv2.cvtColor((img_ref / 255).astype("float32"), cv2.COLOR_BGR2LAB)
    lab_tst = cv2.cvtColor((img_tst / 255).astype("float32"), cv2.COLOR_BGR2LAB)

    temp = cv2.filter2D(lab_ref[::, ::, 0], -1, window, borderType=cv2.BORDER_REPLICATE)
    sigma_l_sq = \
        cv2.filter2D(lab_ref[::, ::, 0] * lab_ref[::, ::, 0], -1, window, borderType=cv2.BORDER_REPLICATE) - temp * temp

    temp = cv2.filter2D(lab_ref[::, ::, 1], -1, window, borderType=cv2.BORDER_REPLICATE)
    sigma_a_sq = \
        cv2.filter2D(lab_ref[::, ::, 1] * lab_ref[::, ::, 1], -1, window, borderType=cv2.BORDER_REPLICATE) - temp * temp

    temp = cv2.filter2D(lab_ref[::, ::, 2], -1, window, borderType=cv2.BORDER_REPLICATE)
    sigma_b_sq = \
        cv2.filter2D(lab_ref[::, ::, 2] * lab_ref[::, ::, 2], -1, window, borderType=cv2.BORDER_REPLICATE) - temp * temp

    v_lab = (sigma_l_sq + sigma_a_sq + sigma_b_sq) / 3
    alpha = (v_lab / 150) + 1
    alpha[alpha > 4.3] = 4.3
    Sc = 1 + 0.045 * np.sqrt(lab_ref[::, ::, 1] * lab_ref[::, ::, 1] + lab_ref[::, ::, 2] * lab_ref[::, ::, 2])
    y = 0.299 * img_ref[::, ::, 0] + 0.587 * img_ref[::, ::, 1] + 0.114 * img_ref[::, ::, 2]

    gx, gy = np.gradient(y)
    gm = np.sqrt(gx * gx + gy * gy)
    mu_y = cv2.filter2D(y, -1, window, borderType=cv2.BORDER_REPLICATE)
    rho_mu_y = img_misc.rho_jncd_delta_e(mu_y)

    beta = rho_mu_y * gm + 1
    vjncd = JNCDLab * alpha * beta * Sc

    delta = np.abs(lab_ref - lab_tst)
    dPE = np.zeros_like(vjncd)

    for ii in range(0, 3):
        dPE += (delta[::, ::, ii] > vjncd) * np.power(np.abs(delta[::, ::, ii] - vjncd), 2)
    dp_e = np.sqrt(np.mean(dPE / 3))

    return dp_e


def colorfulness_dif(img_ref, img_tst):
    """
    C. Gao, K. Panetta, and S. Agaian. No reference color image quality measures. In Proc. of the IEEE
    International Conference on Cybernetics, pages 243 – 248, 2013.
    """
    c_ref = img_misc.colorfulness(img_ref)
    c_pro = img_misc.colorfulness(img_tst)

    return np.abs(c_ref - c_pro)


def color_ssim(img_ref, img_tst):
    """
    C. Gao, K. Panetta, and S. Agaian. No reference color image quality measures. In Proc. of the IEEE
    International Conference on Cybernetics, pages 243 – 248, 2013.
    """
    wl = 3.05
    walpha = 1.1
    wbeta = 0.85
    l_alpha_beta_ref = color_transform.bgr_to_l_alpha_beta(img_ref)
    l_alpha_beta_tst = color_transform.bgr_to_l_alpha_beta(img_tst)

    mssim_l = fidelity_misc.ssim(l_alpha_beta_ref[::, ::, 0], l_alpha_beta_tst[::, ::, 0])
    mssim_alpha = fidelity_misc.ssim(l_alpha_beta_ref[::, ::, 1], l_alpha_beta_tst[::, ::, 1])
    mssim_beta = fidelity_misc.ssim(l_alpha_beta_ref[::, ::, 2], l_alpha_beta_tst[::, ::, 2])

    cssim = np.sqrt(wl * mssim_l * mssim_l + walpha * mssim_alpha * mssim_alpha + wbeta * mssim_beta * mssim_beta)

    return cssim
