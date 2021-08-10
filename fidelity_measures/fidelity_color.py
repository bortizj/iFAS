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


from processing import img_misc, color_transform, icam, compute_texture
from fidelity_measures import fidelity_misc
from gui import ifas_misc
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
            a3 = temp_ref[::, ::, 2]
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


def delta_e2000(img_ref, img_tst, return_mat=False):
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

    l_tst = lab_tst[::, ::, 0]
    a_tst = lab_tst[::, ::, 1]
    b_tst = lab_tst[::, ::, 2]
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

    if return_mat:
        return img_de00, hp_tst
    else:
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

    l_tst = lab_tst[::, ::, 0]
    a_tst = lab_tst[::, ::, 1]
    b_tst = lab_tst[::, ::, 2]
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
    img_m = A[0, 0] * np.power(deltal, 2) + A[1, 1] * np.power(deltac, 2) + A[2, 2] * np.power(deltah, 2) + \
        2 * A[0, 1] * (deltal * deltac) + 2 * A[0, 2] * (deltal * deltah) + 2 * A[1, 2] * (deltac * deltah)
    img_m = np.sqrt(img_m)
    cdm = np.mean(img_m)

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


def local_de2000(img_ref, img_tst):
    """
    S. Ouni, E. Zagrouba, M. Chambah, M. Herbin, A new spatial colour metricfor perceptual comparison, in: 
    Proc. of the International Conference on615Computing and e-Systems, 2008, pp. 413 – 428.
    """
    img_de00, __ = delta_e2000(img_ref, img_tst, return_mat=True)
    w = np.array([[0.5, 1., 0.5], [1., 0., 1.], [0.5, 1., 0.5]]).astype("float32")
    img_de00_w = cv2.filter2D(img_de00.astype("float32"), -1, w, borderType=cv2.BORDER_REPLICATE)
    wde = np.mean((img_de00 + img_de00_w) / 7)

    return wde


def spatial_delta_e2000(img_ref, img_tst, return_imgs=False):
    """
    X. Zhang and B.A. Wandell. A spatial extension of CIELAB for digital color-image reproduction.
    Journal of the Society for Information Display, 5:61 – 63, 1997.
    """
    wi = np.array([[0.921, 0.105, -0.108], [0.531, 0.330, 0], [0.488, 0.371, 0]]).astype("float32")
    si = np.array([[0.0283, 0.133, 4.336], [0.0392, 0.494, 0], [0.0536, 0.386, 0]]).astype("float32")
    xx, yy = np.meshgrid(np.arange(-11, 12), np.arange(-11, 12))

    h1 = wi[0, 0] * ifas_misc.gaussian(xx, yy, si[0, 0]) +  wi[0, 1] * ifas_misc.gaussian(xx, yy, si[0, 1]) + \
        wi[0, 2] * ifas_misc.gaussian(xx, yy, si[0, 2])
    h1 = (h1 / np.sum(h1)).astype("float32")

    h2 = wi[1, 0] * ifas_misc.gaussian(xx, yy, si[1, 0]) + wi[1, 1] * ifas_misc.gaussian(xx, yy, si[1, 1])
    h2 = (h2 / np.sum(h2)).astype("float32")

    h3 = wi[2, 0] * ifas_misc.gaussian(xx, yy, si[2, 0]) + wi[2, 1] * ifas_misc.gaussian(xx, yy, si[2, 1])
    h3 = (h3 / np.sum(h3)).astype("float32")

    xyz_ref = color_transform.linear_color_transform(
        cv2.cvtColor(img_ref, cv2.COLOR_BGR2RGB) / 255., tr_type="rgb_to_xyz"
        )
    xyz_tst = color_transform.linear_color_transform(
        cv2.cvtColor(img_tst, cv2.COLOR_BGR2RGB) / 255., tr_type="rgb_to_xyz"
        )

    o123_ref = color_transform.linear_color_transform(xyz_ref, tr_type="xyz_to_o1o2o3").astype("float32")
    o123_tst = color_transform.linear_color_transform(xyz_tst, tr_type="xyz_to_o1o2o3").astype("float32")

    o1_ref = cv2.filter2D(o123_ref[::, ::, 0], -1, h1, borderType=cv2.BORDER_REPLICATE)
    o2_ref = cv2.filter2D(o123_ref[::, ::, 1], -1, h2, borderType=cv2.BORDER_REPLICATE)
    o3_ref = cv2.filter2D(o123_ref[::, ::, 2], -1, h3, borderType=cv2.BORDER_REPLICATE)
    o1_tst = cv2.filter2D(o123_tst[::, ::, 0], -1, h1, borderType=cv2.BORDER_REPLICATE)
    o2_tst = cv2.filter2D(o123_tst[::, ::, 1], -1, h2, borderType=cv2.BORDER_REPLICATE)
    o3_tst = cv2.filter2D(o123_tst[::, ::, 2], -1, h3, borderType=cv2.BORDER_REPLICATE)

    xyz_ref = color_transform.linear_color_transform(
        np.dstack((o1_ref, o2_ref, o3_ref)), tr_type="o1o2o3_to_xyz"
        ).astype("float32")
    xyz_tst = color_transform.linear_color_transform(
        np.dstack((o1_tst, o2_tst, o3_tst)), tr_type="o1o2o3_to_xyz"
        ).astype("float32")
    rgb_ref = color_transform.linear_color_transform(xyz_ref, tr_type="xyz_to_rgb").astype("float32")
    rgb_tst = color_transform.linear_color_transform(xyz_tst, tr_type="xyz_to_rgb").astype("float32")
    rgb_ref = np.clip(255 * rgb_ref, 0, 255).astype("uint8")
    rgb_tst = np.clip(255 * rgb_tst, 0, 255).astype("uint8")

    img_de00, __ = delta_e2000(
        cv2.cvtColor(rgb_ref, cv2.COLOR_BGR2RGB), cv2.cvtColor(rgb_tst, cv2.COLOR_BGR2RGB), return_mat=True
        )
    sde00 = np.mean(img_de00)

    if return_imgs:
        return rgb_ref, rgb_tst
    else:
        return sde00


def wdelta_e(img_ref, img_tst):
    """
    G. Hong, M. Luo, A new algorithm for calculating perceived colourdiffer-ence of images, 
    Imaging Science Journal 54 (2006) 1 – 15.
    """
    img_de00, hp_tst = delta_e2000(img_ref, img_tst, return_mat=True)

    h_edges = np.linspace(0, 360, 181)
    hist_c, __ = np.histogram(180 * np.reshape(hp_tst, hp_tst.size) / np.pi, h_edges)
    ind = np.digitize(180 * hp_tst / np.pi, h_edges)
    hist_c = hist_c / np.sum(hist_c)
    index_sort = np.argsort(hist_c)
    hists_sort = hist_c[index_sort]
    hists_sort = np.cumsum(hists_sort)

    k_25 = np.where(hists_sort > 0.25)[0][0]
    k_50 = np.where(hists_sort > 0.50)[0][0]
    k_75 = np.where(hists_sort > 0.75)[0][0]

    hist_c[index_sort[0:k_25]] = hist_c[index_sort[0:k_25]]/4
    hist_c[index_sort[k_25:k_50]] = hist_c[index_sort[k_25:k_50]]/2
    hist_c[index_sort[k_50:k_75]] = hist_c[index_sort[k_50:k_75]]
    hist_c[index_sort[k_75:]] = 2.25*hist_c[index_sort[k_75:]]

    h_dis = np.zeros_like(hist_c)
    img_de00_w = np.zeros_like(img_de00)
    for ii in range(0, h_dis.size):
        if np.sum(ind == ii) != 0:
            h_dis[ii] = np.mean(img_de00[ind == ii])
            img_de00_w[ind == ii] = img_de00[ind == ii] * hist_c[ii]

    w_delta_e = np.sum(np.power(h_dis, 2) * hist_c) / 4

    return w_delta_e


def shame_cielab(img_ref, img_tst):
    """
    M. Pedersen, J. Hardeberg, A new spatial filtering based imagedifferencemetric based on hue angle weighting, 
    Journal of Imaging Science andTech-nology 56 (2012) 50501 1 – 12.
    """
    rgb_ref, rgb_tst = spatial_delta_e2000(img_ref, img_tst, return_imgs=True)
    shame_de = wdelta_e(cv2.cvtColor(rgb_ref, cv2.COLOR_RGB2BGR), cv2.cvtColor(rgb_tst, cv2.COLOR_RGB2BGR))

    return shame_de


def visual_saliency_index(img_ref, img_tst):
    """
    L. Zhang, Y. Shen, and H. Li. Vsi: A visual saliency-induced index for perceptual image quality 
    assessment. IEEE Transactions on Image Processing, 23:4270 – 4281, 2014.
    """
    constForVS = 1.27
    constForGM = 386
    constForChrom = 130
    alpha = 0.40
    lambda_ = 0.020
    sigmaF = 1.34
    omega0 = 0.0210
    sigmaD = 145
    sigmaC = 0.001

    sm1 = img_misc.salient_regions(img_ref, sigma_f=sigmaF, omega_0=omega0, sigma_d=sigmaD, sigma_c=sigmaC)
    sm2 = img_misc.salient_regions(img_tst, sigma_f=sigmaF, omega_0=omega0, sigma_d=sigmaD, sigma_c=sigmaC)

    rows, cols, _ = img_ref.shape
    L1 = 0.06 * img_ref[::, ::, 0] + 0.63 * img_ref[::, ::, 1] + 0.27 * img_ref[::, ::, 2]
    L2 = 0.06 * img_tst[::, ::, 0] + 0.63 * img_tst[::, ::, 1] + 0.27 * img_tst[::, ::, 2]
    M1 = 0.30 * img_ref[::, ::, 0] + 0.04 * img_ref[::, ::, 1] - 0.35 * img_ref[::, ::, 2]
    M2 = 0.30 * img_tst[::, ::, 0] + 0.04 * img_tst[::, ::, 1] - 0.35 * img_tst[::, ::, 2]
    N1 = 0.34 * img_ref[::, ::, 0] - 0.60 * img_ref[::, ::, 1] + 0.17 * img_ref[::, ::, 2]
    N2 = 0.34 * img_tst[::, ::, 0] - 0.60 * img_tst[::, ::, 1] + 0.17 * img_tst[::, ::, 2]

    min_dim = np.minimum(rows, cols)
    w = int(np.maximum(1, np.round(min_dim / 256)))
    aveKernel = np.ones((w, w)) / (w * w)

    aveM1 = cv2.filter2D(M1, ddepth=-1, kernel=aveKernel, borderType=cv2.BORDER_REFLECT_101)
    aveM2 = cv2.filter2D(M2, ddepth=-1, kernel=aveKernel, borderType=cv2.BORDER_REFLECT_101)

    M1 = aveM1[0:rows:w, 0:cols:w]
    M2 = aveM2[0:rows:w, 0:cols:w]

    aveN1 = cv2.filter2D(N1, ddepth=-1, kernel=aveKernel, borderType=cv2.BORDER_REFLECT_101)
    aveN2 = cv2.filter2D(N2, ddepth=-1, kernel=aveKernel, borderType=cv2.BORDER_REFLECT_101)

    N1 = aveN1[0:rows:w, 0:cols:w]
    N2 = aveN2[0:rows:w, 0:cols:w]

    aveL1 = cv2.filter2D(L1, ddepth=-1, kernel=aveKernel, borderType=cv2.BORDER_REFLECT_101)
    aveL2 = cv2.filter2D(L2, ddepth=-1, kernel=aveKernel, borderType=cv2.BORDER_REFLECT_101)

    L1 = aveL1[0:rows:w, 0:cols:w]
    L2 = aveL2[0:rows:w, 0:cols:w]

    aveSM1 = cv2.filter2D(sm1, ddepth=-1, kernel=aveKernel, borderType=cv2.BORDER_REFLECT_101)
    aveSM2 = cv2.filter2D(sm2, ddepth=-1, kernel=aveKernel, borderType=cv2.BORDER_REFLECT_101)

    sm1 = aveSM1[0:rows:w, 0:cols:w]
    sm2 = aveSM2[0:rows:w, 0:cols:w]

    dx = np.array([[3, 0, -3], [10, 0, -10], [3, 0, -3]])/16.
    dy = np.array([[3, 10, 3], [0, 0, 0], [-3, -10, -3]])/16.

    IxL1 = cv2.filter2D(L1, ddepth=-1, kernel=dx, borderType=cv2.BORDER_REFLECT_101)
    IyL1 = cv2.filter2D(L1, ddepth=-1, kernel=dy, borderType=cv2.BORDER_REFLECT_101)
    gradientMap1 = np.sqrt(np.power(IxL1, 2) + np.power(IyL1, 2))

    IxL2 = cv2.filter2D(L2, ddepth=-1, kernel=dx, borderType=cv2.BORDER_REFLECT_101)
    IyL2 = cv2.filter2D(L2, ddepth=-1, kernel=dy, borderType=cv2.BORDER_REFLECT_101)
    gradientMap2 = np.sqrt(np.power(IxL2, 2) + np.power(IyL2, 2))

    VSSimMatrix = (2 * sm1 * sm2 + constForVS) / (np.power(sm1, 2) + np.power(sm2, 2) + constForVS)
    gradientSimMatrix = (
        (2 * gradientMap1 * gradientMap2 + constForGM) / 
        (np.power(gradientMap1, 2) + np.power(gradientMap2, 2) + constForGM)
        )

    weight = np.maximum(sm1, sm2)

    ISimMatrix = (2 * M1 * M2 + constForChrom) / (np.power(M1, 2) + np.power(M2, 2) + constForChrom)
    QSimMatrix = (2 * N1 * N2 + constForChrom) / (np.power(N1, 2) + np.power(N2, 2) + constForChrom)

    prod = ISimMatrix * QSimMatrix
    prod[np.where(prod < 0)] = 0
    SimMatrixC = np.power(gradientSimMatrix, alpha) * VSSimMatrix * np.real(np.power(prod, lambda_)) * weight
    sim = np.nansum(SimMatrixC) / np.nansum(weight)
    SimMatrixC[np.where(np.isnan(SimMatrixC))] = 0

    return sim


def chroma_spread_extreme(img_ref, img_tst):
    """
    M.H. Pinson and S. Wolf. A new standardized method for objectively measuring video quality. IEEE 
    transactions on broadcasting, 50:312 – 322, 2004.
    """
    ycrcb_ref = cv2.cvtColor(img_ref, cv2.COLOR_BGR2YCrCb)
    ycrcb_tst = cv2.cvtColor(img_tst, cv2.COLOR_BGR2YCrCb)

    mCb_ref, mCr_ref, MuCb_ref, MuCr_ref = img_misc.mean8x8block(ycrcb_ref[::, ::, 2], ycrcb_ref[::, ::, 1])
    mCb_pro, mCr_pro, MuCb_pro, MuCr_pro = img_misc.mean8x8block(ycrcb_tst[::, ::, 2], ycrcb_tst[::, ::, 1])

    eCbCr = np.sqrt(np.power(mCb_ref - mCb_pro, 2) + np.power(mCr_ref - mCr_pro, 2))
    ECbCr = np.sqrt(np.power(MuCb_ref - MuCb_pro, 2) + np.power(MuCr_ref - MuCr_pro, 2))
    chroma_spread = np.std(eCbCr)

    p = np.sort(np.array(eCbCr).ravel())[::-1]
    chroma_extreme = np.mean(p[0:np.int_(p.size * 0.01)]) - p[np.int_(p.size * 0.01) - 1]

    return 0.0192 * chroma_spread + 0.0076 * chroma_extreme


def colorhist_diff(img_ref, img_tst):
    """
    S.M. Lee, J.H. Xin, and S. Westland. Evaluation of image similarity by histogram intersection. Color 
    Research and Application, 30:265 – 274, 2005.
    """
    lab_ref = cv2.cvtColor((img_ref / 255).astype("float32"), cv2.COLOR_BGR2LAB)
    lab_tst = cv2.cvtColor((img_tst / 255).astype("float32"), cv2.COLOR_BGR2LAB)

    H_ref = img_misc.color_histogram(lab_ref, minc=np.array([0, -128, -128]), maxc=np.array([100, 127, 127]), nbins=8)
    H_pro = img_misc.color_histogram(lab_tst, minc=np.array([0, -128, -128]), maxc=np.array([100, 127, 127]), nbins=8)

    if np.sum(H_ref) != 0:
        H_ref = H_ref / np.sum(H_ref)
    if np.sum(H_pro) != 0:
        H_pro = H_pro / np.sum(H_pro)

    di = np.sum(np.minimum(H_ref, H_pro))

    return di


def cpsnrha(img_ref, img_tst, wsize=11):
    """
    N. Ponomarenko, O. Ieremeiev, V. Lukin, K. Egiazarian, and M. Carli. Modified image visual quality 
    metrics for contrast change and mean shift accounting. In Proc. of the International Conference The 
    Experience of Designing and Application of CAD Systems in Microelectronics, pages 305 – 311, 2011.
    """
    ycrcb_ref = cv2.cvtColor(img_ref, cv2.COLOR_BGR2YCrCb)
    ycrcb_tst = cv2.cvtColor(img_tst, cv2.COLOR_BGR2YCrCb)

    muy_ref = np.mean(ycrcb_ref[::, ::, 0])
    mu_Cb_ref = np.mean(ycrcb_ref[::, ::, 2])
    mu_Cr_ref = np.mean(ycrcb_ref[::, ::, 1])
    muy_tst = np.mean(ycrcb_tst[::, ::, 0])
    mu_Cb_tst = np.mean(ycrcb_tst[::, ::, 2])
    mu_Cr_tst = np.mean(ycrcb_tst[::, ::, 1])

    mu_delta_y = muy_ref - muy_tst
    mu_delta_Cb = mu_Cb_ref - mu_Cb_tst
    mu_delta_Cr = mu_Cr_ref - mu_Cr_tst

    cy_tst = ycrcb_tst[::, ::, 0] + mu_delta_y
    c_Cb_tst = ycrcb_tst[::, ::, 2] + mu_delta_Cb
    c_Cr_tst = ycrcb_tst[::, ::, 1] + mu_delta_Cr

    mu_CY_tst = np.mean(cy_tst)
    mu_CCb_tst = np.mean(c_Cb_tst)
    mu_CCr_tst = np.mean(c_Cr_tst)

    PY = (
        np.sum((ycrcb_ref[::, ::, 0] - muy_ref) * (cy_tst - mu_CY_tst)) / 
        np.sum(np.power((cy_tst - mu_CY_tst), 2))
        )
    PCb = (
        np.sum((ycrcb_ref[::, ::, 2] - mu_Cb_ref) * (c_Cb_tst - mu_CCb_tst)) / 
        np.sum(np.power((c_Cb_tst - mu_CCb_tst), 2))
        )
    PCr = (
        np.sum((ycrcb_ref[::, ::, 1] - mu_Cr_ref) * (c_Cr_tst - mu_CCr_tst)) / 
        np.sum(np.power((c_Cr_tst - mu_CCr_tst), 2))
        )
    DY = PY * cy_tst
    DCb = PCb * c_Cb_tst
    DCr = PCr * c_Cr_tst

    __, p_hvs_m_YCHMA, __, __ = img_misc.dct_block_mse(ycrcb_ref[::, ::, 0], cy_tst)
    __, p_hvs_m_CbCHMA, __, __ = img_misc.dct_block_mse(ycrcb_ref[::, ::, 2], c_Cb_tst)
    __, p_hvs_m_CrCHMA, __, __ = img_misc.dct_block_mse(ycrcb_ref[::, ::, 1], c_Cr_tst)
    __, p_hvs_m_YDHMA, __, __ = img_misc.dct_block_mse(ycrcb_ref[::, ::, 0], DY)
    __, p_hvs_m_CbDHMA, __, __ = img_misc.dct_block_mse(ycrcb_ref[::, ::, 2], DCb)
    __, p_hvs_m_CrDHMA, __, __ = img_misc.dct_block_mse(ycrcb_ref[::, ::, 1], DCr)

    if PY < 1:
        c = 0.002
    else:
        c = 0.25

    if p_hvs_m_YCHMA > p_hvs_m_YDHMA:
        p_hvs_m_YCHMA = p_hvs_m_YDHMA + (p_hvs_m_YCHMA - p_hvs_m_YDHMA) * c
    if p_hvs_m_CbCHMA > p_hvs_m_CbDHMA:
        p_hvs_m_CbCHMA = p_hvs_m_CbDHMA + (p_hvs_m_CbCHMA - p_hvs_m_CbDHMA) * c
    if p_hvs_m_CrCHMA > p_hvs_m_CrDHMA:
        p_hvs_m_CrCHMA = p_hvs_m_CrDHMA + (p_hvs_m_CrCHMA - p_hvs_m_CrDHMA) * c

    p_hvs_m_YCHMA += mu_delta_y * mu_delta_y * 0.04
    p_hvs_m_CbCHMA += mu_delta_Cb * mu_delta_Cb * 0.04
    p_hvs_m_CrCHMA += mu_delta_Cr * mu_delta_Cr * 0.04
    p_hvs_m_YCHMA = img_misc.clip_psnr(p_hvs_m_YCHMA)
    p_hvs_m_CbCHMA = img_misc.clip_psnr(p_hvs_m_CbCHMA)
    p_hvs_m_CrCHMA = img_misc.clip_psnr(p_hvs_m_CrCHMA)

    cpsnrhma = (p_hvs_m_YCHMA + p_hvs_m_CbCHMA * 0.5 + p_hvs_m_CrCHMA * 0.5) / (1 + 2 * 0.5)

    return cpsnrhma


def ssim_ipt(img_ref, img_tst):
    """
    N. Bonnier, F. Schmitt, H. Brettel, and S. Berche. Evaluation of spatial gamut mapping algorithms. 
    In Proc. of the Color and Imaging Conference, pages 56 – 61, 2006.
    """
    img_ref = cv2.cvtColor(img_ref, cv2.COLOR_BGR2RGB)
    img_tst = cv2.cvtColor(img_tst, cv2.COLOR_BGR2RGB)
    iCAM_ref = icam.RGB2iCAM(img_ref)  # ipt_image
    iCAM_tst = icam.RGB2iCAM(img_tst)
    iCAM_ref[np.where(np.isnan(iCAM_ref))] = 0
    iCAM_tst[np.where(np.isnan(iCAM_tst))] = 0
    mssim_i = fidelity_misc.ssim(iCAM_ref[::, ::, 0], iCAM_tst[::, ::, 0])
    mssim_p = fidelity_misc.ssim(iCAM_ref[::, ::, 1], iCAM_tst[::, ::, 1])
    mssim_t = fidelity_misc.ssim(iCAM_ref[::, ::, 2], iCAM_tst[::, ::, 2])

    return mssim_i * mssim_p * mssim_t


def cid_appearance(img_ref, img_tst):
    """
    G.M. Johnson. Using color appearance in image quality metrics. In Proc. of the International Workshop 
    on Video Processing and Quality Metrics for Consumer Electronics, pages 1 – 4, 2006.
    """
    iCAM_ref = icam.RGB2iCAM(img_ref)#ipt_image
    iCAM_tst = icam.RGB2iCAM(img_tst)
    CID = np.nansum(np.power(iCAM_ref - iCAM_tst, 2), 2)

    return np.nanmean(CID)


def circular_hue(img_ref, img_tst):
    """
    D. Lee and E.S. Rogers. Towards a novel perceptual color difference metric using circular processing 
    of hue components. In Proc. of the IEEE International Conference on Acoustics, Speech and Signal 
    Processing, pages 166 – 170, 2014.
    """
    wi = np.array([[0.921, 0.105, -0.108], [0.531, 0.330, 0], [0.488, 0.371, 0]])
    si = np.array([[0.0283, 0.133, 4.336], [0.0392, 0.494, 0], [0.0536, 0.386, 0]])
    xx, yy = np.meshgrid(np.arange(-11, 12), np.arange(-11, 12))

    h1 = (
        wi[0, 0] * ifas_misc.gaussian(xx, yy, si[0, 0]) + wi[0, 1] * ifas_misc.gaussian(xx, yy, si[0, 1]) + 
        wi[0, 2] * ifas_misc.gaussian(xx, yy, si[0, 2])
        )
    h1 = h1 / np.sum(h1)
    h2 = wi[1, 0] * ifas_misc.gaussian(xx, yy, si[1, 0]) + wi[1, 1] * ifas_misc.gaussian(xx, yy, si[1, 1])
    h2 = h2 / np.sum(h2)
    h3 = wi[2, 0] * ifas_misc.gaussian(xx, yy, si[2, 0]) + wi[2, 1] * ifas_misc.gaussian(xx, yy, si[2, 1])
    h3 = h3 / np.sum(h3)

    XYZ_ref = color_transform.linear_color_transform(cv2.cvtColor(img_ref, cv2.COLOR_BGR2RGB), tr_type="rgb_to_xyz")
    XYZ_tst = color_transform.linear_color_transform(cv2.cvtColor(img_tst, cv2.COLOR_BGR2RGB), tr_type="rgb_to_xyz")
    O123_ref = color_transform.linear_color_transform(XYZ_ref, tr_type="xyz_to_o1o2o3")
    O123_tst = color_transform.linear_color_transform(XYZ_tst, tr_type="xyz_to_o1o2o3")

    O1_ref = cv2.filter2D(O123_ref[::, ::, 0].astype("float32"), ddepth=-1, kernel=h1, borderType=cv2.BORDER_REFLECT_101)
    O2_ref = cv2.filter2D(O123_ref[::, ::, 1].astype("float32"), ddepth=-1, kernel=h2, borderType=cv2.BORDER_REFLECT_101)
    O3_ref = cv2.filter2D(O123_ref[::, ::, 2].astype("float32"), ddepth=-1, kernel=h3, borderType=cv2.BORDER_REFLECT_101)
    O1_tst = cv2.filter2D(O123_tst[::, ::, 0].astype("float32"), ddepth=-1, kernel=h1, borderType=cv2.BORDER_REFLECT_101)
    O2_tst = cv2.filter2D(O123_tst[::, ::, 1].astype("float32"), ddepth=-1, kernel=h2, borderType=cv2.BORDER_REFLECT_101)
    O3_tst = cv2.filter2D(O123_tst[::, ::, 2].astype("float32"), ddepth=-1, kernel=h3, borderType=cv2.BORDER_REFLECT_101)

    XYZ_ref = color_transform.linear_color_transform(np.dstack((O1_ref, O2_ref, O3_ref)), tr_type="o1o2o3_to_xyz")
    XYZ_tst = color_transform.linear_color_transform(np.dstack((O1_tst, O2_tst, O3_tst)), tr_type="o1o2o3_to_xyz")
    rgb_ref_back = color_transform.linear_color_transform(XYZ_ref, tr_type="xyz_to_rgb")
    rgb_tst_back = color_transform.linear_color_transform(XYZ_tst, tr_type="xyz_to_rgb")
    rgb_ref_back = np.clip(255 * rgb_ref_back, 0, 255).astype("uint8")
    rgb_tst_back = np.clip(255 * rgb_tst_back, 0, 255).astype("uint8")

    lab_ref = cv2.cvtColor((rgb_ref_back / 255).astype("float32"), cv2.COLOR_RGB2LAB)
    lab_tst = cv2.cvtColor((rgb_tst_back / 255).astype("float32"), cv2.COLOR_RGB2LAB)
    H_ref = np.arctan2(lab_ref[:,:, 2], lab_ref[:,:, 1])
    H_tst = np.arctan2(lab_tst[:,:, 2], lab_tst[:,:, 1])

    sizewin = 11
    window = np.ones((sizewin, sizewin))
    Kh = (360 * 0.01) ** 2
    Kc = (180 * 0.01) ** 2

    H_ref_mean = np.arctan2(
        cv2.filter2D(np.cos(H_ref).astype("float32"), ddepth=-1, kernel=window, borderType=cv2.BORDER_REFLECT_101),
        cv2.filter2D(np.sin(H_ref).astype("float32"), ddepth=-1, kernel=window, borderType=cv2.BORDER_REFLECT_101) 
        )
    H_tst_mean = np.arctan2(
        cv2.filter2D(np.cos(H_tst).astype("float32"), ddepth=-1, kernel=window, borderType=cv2.BORDER_REFLECT_101),
        cv2.filter2D(np.sin(H_tst).astype("float32"), ddepth=-1, kernel=window, borderType=cv2.BORDER_REFLECT_101) 
        )
    DH = (2 * H_ref_mean * H_tst_mean + Kh) / (np.power(H_ref_mean, 2) + np.power(H_tst_mean, 2) + Kh)
    dH = np.mean(DH)

    C_ref = np.sqrt(np.power(lab_ref[::, ::, 1], 2) + np.power(lab_ref[::, ::, 2], 2))
    C_ref_mean = cv2.filter2D(C_ref.astype("float32"), ddepth=-1, kernel=window, borderType=cv2.BORDER_REFLECT_101)
    C_tst = np.sqrt(np.power(lab_tst[::, ::, 1], 2) + np.power(lab_tst[::, ::, 2], 2))
    C_tst_mean = cv2.filter2D(C_tst.astype("float32"), ddepth=-1, kernel=window, borderType=cv2.BORDER_REFLECT_101)
    DC = (2 * C_ref_mean * C_tst_mean + Kc) / (np.power(C_ref_mean, 2) + np.power(C_tst_mean, 2) + Kc)
    dC = np.mean(DC)
    dL = fidelity_misc.ssim(lab_ref[::, ::, 0], lab_tst[::, ::, 0])

    dE = 1 - dH * dC * dL

    return dE


def texture_patch_cd(img_ref, img_tst, th=10, r=1., min_num_pixels=4, sq=False):
    """
    B. Ortiz-Jaramillo, A. Kumcu, L. Platisa, and W. Philips. Evaluation of color differences in natural 
    scene color images. Signal Processing: Image Communication, 71:128 – 137, 2019.
    """
    SE = np.ones((np.int_(2 * r + 1), np.int_(2 * r + 1)))
    ycrcb_ref = cv2.cvtColor(img_ref, cv2.COLOR_BGR2YCrCb)
    ycrcb_tst = cv2.cvtColor(img_tst, cv2.COLOR_BGR2YCrCb)
    DL = fidelity_misc.ssim(ycrcb_ref[::, ::, 0], ycrcb_tst[::, ::, 0], ismap=True)
    dl = 0.

    ycrcb_ref = cv2.cvtColor(img_ref, cv2.COLOR_BGR2YCrCb).astype("float32")
    ycrcb_tst = cv2.cvtColor(img_tst, cv2.COLOR_BGR2YCrCb).astype("float32")
    ECbCr = np.sqrt(
        np.power(ycrcb_ref[::, ::, 1] - ycrcb_tst[::, ::, 1], 2) + 
        np.power(ycrcb_ref[::, ::, 2] - ycrcb_tst[::, ::, 2], 2)
        )

    __, lbp_img = compute_texture.lbp(img_ref, th=th, r=r)
    list_of_patterns = np.unique(lbp_img)
    cd = 0.
    wpt = 0.
    CD = np.zeros(lbp_img.shape)
    Np = lbp_img.size

    for pp in list_of_patterns:
        homo_patches = lbp_img == pp
        homo_patches = cv2.morphologyEx(homo_patches.astype("uint8"), cv2.MORPH_CLOSE, SE.astype("uint8"))
        num_patches, patch_index, stats, centroids = cv2.connectedComponentsWithStats(homo_patches, 8, cv2.CV_32S)
        for ii in range(1, num_patches):
            idx = np.where(patch_index == ii)
            if idx[0].size > min_num_pixels:
                if sq:
                    chroma_spread = np.std(np.power(ECbCr[idx], 2))
                    p = np.sort(np.array(np.power(ECbCr[idx], 2)).ravel())[::-1]
                    chroma_extreme = (
                        np.mean(p[0:np.int_(np.ceil(p.size * 0.01))]) - p[np.int_(np.ceil(p.size * 0.01)) - 1]
                        )
                else:
                    chroma_spread = np.std(ECbCr[idx])
                    p = np.sort(np.array(ECbCr[idx]).ravel())[::-1]
                    chroma_extreme = (
                        np.mean(p[0:np.int_(np.ceil(p.size * 0.01))]) - p[np.int_(np.ceil(p.size * 0.01)) - 1]
                        )
                temp = 0.0192 * chroma_spread + 0.0076 * chroma_extreme
                templ = np.mean((1. - DL[idx]) / 2.)
                CD[idx] = 0.7 * temp + 0.3 * templ

                wp = (1. * idx[0].size) / Np
                cd += wp * temp
                dl += wp * templ
                wpt += 1  # wp

    return 0.7 * cd + 0.3 * dl


def osa_ucs_de(img_ref, img_tst):
    """
    R. Huertas, M. Melgosa, and C. Oleari. Performance of a color-difference formula based on OSA-UCS space using 
    small-medium color differences. Journal of the Optical Society of America A, 23:2077 – 2084, 2006.
    """
    Ljg_ref = color_transform.bgr2Ljg(img_ref)
    Ljg_pro = color_transform.bgr2Ljg(img_tst)
    C_ref = np.sqrt(np.power(Ljg_ref[::, ::, 1], 2) + np.power(Ljg_ref[::, ::, 2], 2))
    h_ref = np.arctan2(Ljg_ref[::, ::, 1], -Ljg_ref[::, ::, 2])
    C_pro = np.sqrt(np.power(Ljg_pro[::, ::, 1], 2) + np.power(Ljg_pro[::, ::, 2], 2))
    h_pro = np.arctan2(Ljg_pro[::, ::, 1], -Ljg_pro[::, ::, 2])
    S_L = 2.499 + 0.07 * (Ljg_ref[::, ::, 0] + Ljg_pro[::, ::, 0]) / 2.
    S_C = 1.235 + 0.58 * (C_ref + C_pro) / 2
    S_H = 1.392 + 0.17 * (h_ref + h_pro) / 2
    dL = (Ljg_ref[::, ::, 0] - Ljg_pro[::, ::, 0]) / S_L
    dC = (C_ref - C_pro) / S_C
    dh = (h_ref - h_pro) / S_H
    DE = 10 * np.sqrt(np.power(dL, 2) + np.power(dC, 2) + np.power(dh, 2))
    dE = np.nanmean(DE)
    return dE


def osa_ucs_sde(img_ref, img_tst):
    """
    G. Simone, C. Oleari, and I. Farup. An alternative color difference formula for computing image 
    difference. In Proc. of the Gjøvik Color Imaging Symposium, pages 8 – 11, 2009.
    """
    wi = np.array([[0.921, 0.105, -0.108], [0.531, 0.330, 0], [0.488, 0.371, 0]])
    si = np.array([[0.0283, 0.133, 4.336], [0.0392, 0.494, 0], [0.0536, 0.386, 0]])
    xx, yy = np.meshgrid(np.arange(-11, 12), np.arange(-11, 12))

    h1 = (
        wi[0, 0] * ifas_misc.gaussian(xx, yy, si[0, 0]) + wi[0, 1] * ifas_misc.gaussian(xx, yy, si[0, 1]) + 
        wi[0, 2] * ifas_misc.gaussian(xx, yy, si[0, 2])
        )
    h1 = h1 / np.sum(h1)
    h2 = wi[1, 0] * ifas_misc.gaussian(xx, yy, si[1, 0]) + wi[1, 1] * ifas_misc.gaussian(xx, yy, si[1, 1])
    h2 = h2 / np.sum(h2)
    h3 = wi[2, 0] * ifas_misc.gaussian(xx, yy, si[2, 0]) + wi[2, 1] * ifas_misc.gaussian(xx, yy, si[2, 1])
    h3 = h3 / np.sum(h3)

    XYZ_ref = color_transform.linear_color_transform(cv2.cvtColor(img_ref, cv2.COLOR_BGR2RGB), tr_type="rgb_to_xyz")
    XYZ_tst = color_transform.linear_color_transform(cv2.cvtColor(img_tst, cv2.COLOR_BGR2RGB), tr_type="rgb_to_xyz")
    O123_ref = color_transform.linear_color_transform(XYZ_ref, tr_type="xyz_to_o1o2o3")
    O123_tst = color_transform.linear_color_transform(XYZ_tst, tr_type="xyz_to_o1o2o3")

    O1_ref = cv2.filter2D(O123_ref[::, ::, 0].astype("float32"), ddepth=-1, kernel=h1, borderType=cv2.BORDER_REFLECT_101)
    O2_ref = cv2.filter2D(O123_ref[::, ::, 1].astype("float32"), ddepth=-1, kernel=h2, borderType=cv2.BORDER_REFLECT_101)
    O3_ref = cv2.filter2D(O123_ref[::, ::, 2].astype("float32"), ddepth=-1, kernel=h3, borderType=cv2.BORDER_REFLECT_101)
    O1_tst = cv2.filter2D(O123_tst[::, ::, 0].astype("float32"), ddepth=-1, kernel=h1, borderType=cv2.BORDER_REFLECT_101)
    O2_tst = cv2.filter2D(O123_tst[::, ::, 1].astype("float32"), ddepth=-1, kernel=h2, borderType=cv2.BORDER_REFLECT_101)
    O3_tst = cv2.filter2D(O123_tst[::, ::, 2].astype("float32"), ddepth=-1, kernel=h3, borderType=cv2.BORDER_REFLECT_101)

    XYZ_ref = color_transform.linear_color_transform(np.dstack((O1_ref, O2_ref, O3_ref)), tr_type="o1o2o3_to_xyz")
    XYZ_tst = color_transform.linear_color_transform(np.dstack((O1_tst, O2_tst, O3_tst)), tr_type="o1o2o3_to_xyz")
    rgb_ref_back = color_transform.linear_color_transform(XYZ_ref, tr_type="xyz_to_rgb")
    rgb_tst_back = color_transform.linear_color_transform(XYZ_tst, tr_type="xyz_to_rgb")
    rgb_ref_back = np.clip(255 * rgb_ref_back, 0, 255).astype("uint8")
    rgb_tst_back = np.clip(255 * rgb_tst_back, 0, 255).astype("uint8")

    return osa_ucs_de(rgb_ref_back, rgb_tst_back)


def color_image_diff(img_ref, img_tst):
    """
    I. Lissner, J. Preiss, P. Urban, M.-S. Lichtenauer, and P. Zolliker. Image-difference prediction: From 
    grayscale to color. IEEE Transactions on Image Processing, 22:435 – 446, 2013.
    """
    img1 = cv2.cvtColor(img_ref, cv2.COLOR_BGR2RGB)
    img2 = cv2.cvtColor(img_tst, cv2.COLOR_BGR2RGB)
    cycles_per_degree = 20
    idf_consts = np.array([0.002, 0.1, 0.1, 0.002, 0.008])

    img1_XYZ = color_transform.SRGB_to_XYZ(img1)
    img1_filt = color_transform.scielab_simple(2 * cycles_per_degree, img1_XYZ)
    img1_LAB2000HL = color_transform.XYZ_to_LAB2000HL(img1_filt)
    img2_XYZ = color_transform.SRGB_to_XYZ(img2)
    img2_filt = color_transform.scielab_simple(2 * cycles_per_degree, img2_XYZ)
    img2_LAB2000HL = color_transform.XYZ_to_LAB2000HL(img2_filt)

    Window = ifas_misc.matlab_style_gauss2D(shape=(11,11), sigma=1.5)

    img1 = img1_LAB2000HL
    img2 = img2_LAB2000HL
    L1 = img1[::, ::, 0]
    A1 = img1[::, ::, 1]
    B1 = img1[::, ::, 2]
    Chr1_sq = np.power(A1, 2) + np.power(B1, 2)

    L2 = img2[::, ::, 0]
    A2 = img2[::, ::, 1]
    B2 = img2[::, ::, 2]
    Chr2_sq = np.power(A2, 2) + np.power(B2, 2)

    muL1 = cv2.filter2D(L1.astype("float32"), ddepth=-1, kernel=Window, borderType=cv2.BORDER_REFLECT_101)
    muC1 = cv2.filter2D(np.sqrt(Chr1_sq).astype("float32"), ddepth=-1, kernel=Window, borderType=cv2.BORDER_REFLECT_101)
    muL2 = cv2.filter2D(L2.astype("float32"), ddepth=-1, kernel=Window, borderType=cv2.BORDER_REFLECT_101)
    muC2 = cv2.filter2D(np.sqrt(Chr2_sq).astype("float32"), ddepth=-1, kernel=Window, borderType=cv2.BORDER_REFLECT_101)

    sL1_sq = cv2.filter2D(
        np.power(L1, 2).astype("float32"), ddepth=-1, kernel=Window, borderType=cv2.BORDER_REFLECT_101
        ) - np.power(muL1, 2)
    sL1_sq[sL1_sq < 0] = 0
    sL1 = np.sqrt(sL1_sq)
    sL2_sq = cv2.filter2D(
        np.power(L2, 2).astype("float32"), ddepth=-1, kernel=Window, borderType=cv2.BORDER_REFLECT_101
        ) - np.power(muL2, 2)
    sL2_sq[sL2_sq < 0] = 0
    sL2 = np.sqrt(sL2_sq)

    dL_sq = np.power(muL1 - muL2,2)
    dC_sq = np.power(muC1 - muC2,2)

    Tem = np.sqrt(np.power(A1 - A2,2) + np.power(B1 - B2,2) - np.power(np.sqrt(Chr1_sq) - np.sqrt(Chr2_sq),2))
    Tem_filt = cv2.filter2D(Tem.astype("float32"), ddepth=-1, kernel=Window, borderType=cv2.BORDER_REFLECT_101)
    dH_sq = np.power(Tem_filt, 2)
    sL12 = cv2.filter2D(
        (L1 * L2).astype("float32"), ddepth=-1, kernel=Window, borderType=cv2.BORDER_REFLECT_101
        ) - muL1 * muL2

    Maps_invL = 1 / (idf_consts[0] * dL_sq + 1)
    Maps_invLc = (idf_consts[1] + 2 * sL1 * sL2) / (idf_consts[1] + sL1_sq + sL2_sq)
    Maps_invLs = (idf_consts[2] + sL12) / (idf_consts[2] + sL1 * sL2)
    Maps_invC = 1 / (idf_consts[3] * dC_sq + 1)
    Maps_invH = 1 / (idf_consts[4] * dH_sq + 1)

    IDF1 = np.nanmean(Maps_invL)
    IDF2 = np.nanmean(Maps_invLc)
    IDF3 = np.nanmean(Maps_invLs)
    IDF4 = np.nanmean(Maps_invC)
    IDF5 = np.nanmean(Maps_invH)

    prediction = np.real(1 - IDF1 * IDF2 * IDF3 * IDF4 * IDF5)
    Prediction = np.real(1 - Maps_invL * Maps_invLc * Maps_invLs * Maps_invC * Maps_invH)

    return prediction
