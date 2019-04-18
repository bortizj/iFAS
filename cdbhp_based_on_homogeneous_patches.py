import numpy as np
from skimage import color
import cython_functions
from scipy import stats
from scipy import ndimage
import my_utilities as MU
import compute_texture as CT
import matplotlib.pyplot as plt


def cd00_hist_diff_patches(Ref_image, Pro_image, th=0.005, r=1, nbins=8, local_maxmin=True):
    SE = np.ones((np.int_(2 * r + 1), np.int_(2 * r + 1)))
    Lab_ref = color.rgb2lab(Ref_image)
    L_ref = Lab_ref[:, :, 0]
    a_ref = Lab_ref[:, :, 1]
    b_ref = Lab_ref[:, :, 2]
    Lab_pro = color.rgb2lab(Pro_image)
    L_pro = Lab_pro[:, :, 0]
    a_pro = Lab_pro[:, :, 1]
    b_pro = Lab_pro[:, :, 2]
    LBP, _ = CT.lbp(Ref_image, th=th, r=r)
    list_of_patterns = np.unique(LBP)
    cd = 0.
    wpt = 0.
    CD = np.zeros(L_ref.shape)
    Np = L_ref.size
    min_num_pixels = 25
    for pp in list_of_patterns:
        homo_patches = LBP == pp
        homo_patches = ndimage.morphology.binary_closing(homo_patches, SE)
        patch_index, num_patches = ndimage.measurements.label(homo_patches)
        for ii in range(1, num_patches):
            idx = np.where(patch_index == ii)
            if idx[0].size > min_num_pixels:
                wp = 1 - np.exp(-idx[0].size * (-np.log(0.01) / Np))
                L_ref_v = L_ref[idx]
                a_ref_v = a_ref[idx]
                b_ref_v = b_ref[idx]
                if local_maxmin:
                    minc = np.array([np.minimum(np.min(L_ref), np.min(L_pro)), \
                                     np.minimum(np.min(a_ref), np.min(a_pro)), \
                                     np.minimum(np.min(b_ref), np.min(b_pro))])
                    maxc = np.array([np.maximum(np.max(L_ref), np.max(L_pro)), \
                                     np.maximum(np.max(a_ref), np.max(a_pro)), \
                                     np.maximum(np.max(b_ref), np.max(b_pro))])
                else:
                    minc = np.array([0, -128, -128])
                    maxc = np.array([100, 127, 127])
                xx = np.linspace(minc[0], maxc[0], nbins + 1)
                yy = np.linspace(minc[1], maxc[1], nbins + 1)
                zz = np.linspace(minc[2], maxc[2], nbins + 1)
                Lab_v_ref = np.vstack((L_ref_v, np.vstack((a_ref_v, b_ref_v)))).T
                H_ref, _ = np.histogramdd(Lab_v_ref, (xx, yy, zz))
                H_ref = 1. * H_ref / np.sum(H_ref)
                L_pro_v = L_pro[idx]
                a_pro_v = a_pro[idx]
                b_pro_v = b_pro[idx]
                Lab_v_pro = np.vstack((L_pro_v, np.vstack((a_pro_v, b_pro_v)))).T
                H_pro, _ = np.histogramdd(Lab_v_pro, (xx, yy, zz))
                H_pro = 1. * H_pro / np.sum(H_pro)
                temp = np.sum(np.minimum(H_ref, H_pro))
                CD[idx] = temp
                if np.isnan(temp):
                    pass
                cd += temp  # wp * temp
                wpt += 1  # wp
    return CD, cd / wpt


def cd01_colorfulness_diff_patches(Ref_image, Pro_image, th=0.01, r=1.5):
    SE = np.ones((np.int_(2 * r + 1), np.int_(2 * r + 1)))
    Ref_alpha = 1. * Ref_image[:, :, 0] - Ref_image[:, :, 1]
    Ref_beta = 0.5 * (1. * Ref_image[:, :, 0] + Ref_image[:, :, 1]) - Ref_image[:, :, 2]
    Pro_alpha = 1. * Pro_image[:, :, 0] - Pro_image[:, :, 1]
    Pro_beta = 0.5 * (1. * Pro_image[:, :, 0] + Pro_image[:, :, 1]) - Pro_image[:, :, 2]
    LBP, _ = CT.lbp(Ref_image, th=th, r=r)
    list_of_patterns = np.unique(LBP)
    cd = 0.
    wpt = 0.
    CD = np.zeros(L_ref.shape)
    Np = L_ref.size
    min_num_pixels = 25
    for pp in list_of_patterns:
        homo_patches = LBP == pp
        homo_patches = ndimage.morphology.binary_closing(homo_patches, SE)
        patch_index, num_patches = ndimage.measurements.label(homo_patches)
        for ii in range(1, num_patches):
            idx = np.where(patch_index == ii)
            if idx[0].size > min_num_pixels:
                c_text_ref = CAM.colorfulness_Gao(Ref_alpha[idx], Ref_beta[idx])
                c_text_pro = CAM.colorfulness_Gao(Pro_alpha[idx], Pro_beta[idx])
                temp = np.abs(c_text_ref-c_text_pro)
                CD[idx] = temp
                if np.isnan(temp):
                    pass
                cd += temp  # wp * temp
                wpt += 1  # wp
    return CD, cd / wpt


def cd02_ciede2000_patches(Ref_image, Pro_image, th=0.01, r=1.5):
    SE = np.ones((np.int_(2 * r + 1), np.int_(2 * r + 1)))
    Lab_ref = color.rgb2lab(Ref_image)
    L_ref = Lab_ref[:, :, 0]
    a_ref = Lab_ref[:, :, 1]
    b_ref = Lab_ref[:, :, 2]
    Lab_pro = color.rgb2lab(Pro_image)
    L_pro = Lab_pro[:, :, 0]
    a_pro = Lab_pro[:, :, 1]
    b_pro = Lab_pro[:, :, 2]
    LBP, _ = CT.lbp(Ref_image, th=th, r=r)
    list_of_patterns = np.unique(LBP)
    cd = 0.
    wpt = 0.
    CD = np.zeros(L_ref.shape)
    Np = L_ref.size
    min_num_pixels = 25
    for pp in list_of_patterns:
        homo_patches = LBP == pp
        homo_patches = ndimage.morphology.binary_closing(homo_patches, SE)
        patch_index, num_patches = ndimage.measurements.label(homo_patches)
        for ii in range(1, num_patches):
            idx = np.where(patch_index == ii)
            if idx[0].size > min_num_pixels:
                wp = 1 - np.exp(-idx[0].size * (-np.log(0.01) / Np))
                L_ref_v = L_ref[idx]
                a_ref_v = a_ref[idx]
                b_ref_v = b_ref[idx]
                L_pro_v = L_pro[idx]
                a_pro_v = a_pro[idx]
                b_pro_v = b_pro[idx]
                temp = CAM.deltaE2000(L_ref_v, a_ref_v, b_ref_v, L_pro_v, a_pro_v, b_pro_v)
                CD[idx] = temp
                if np.isnan(temp):
                    pass
                cd += temp  # wp * temp
                wpt += 1  # wp
    return CD, cd / wpt


def cd03_mahalanobis_patches(Ref_image, Pro_image, th=0.01, r=1.5):
    SE = np.ones((np.int_(2 * r + 1), np.int_(2 * r + 1)))
    Lab_ref = color.rgb2lab(Ref_image)
    L_ref = Lab_ref[:, :, 0]
    a_ref = Lab_ref[:, :, 1]
    b_ref = Lab_ref[:, :, 2]
    Lab_pro = color.rgb2lab(Pro_image)
    L_pro = Lab_pro[:, :, 0]
    a_pro = Lab_pro[:, :, 1]
    b_pro = Lab_pro[:, :, 2]
    LBP, _ = CT.lbp(Ref_image, th=th, r=r)
    list_of_patterns = np.unique(LBP)
    cd = 0.
    wpt = 0.
    CD = np.zeros(L_ref.shape)
    Np = L_ref.size
    min_num_pixels = 25
    for pp in list_of_patterns:
        homo_patches = LBP == pp
        homo_patches = ndimage.morphology.binary_closing(homo_patches, SE)
        patch_index, num_patches = ndimage.measurements.label(homo_patches)
        for ii in range(1, num_patches):
            idx = np.where(patch_index == ii)
            if idx[0].size > min_num_pixels:
                wp = 1 - np.exp(-idx[0].size * (-np.log(0.01) / Np))
                L_ref_v = L_ref[idx]
                a_ref_v = a_ref[idx]
                b_ref_v = b_ref[idx]
                L_pro_v = L_pro[idx]
                a_pro_v = a_pro[idx]
                b_pro_v = b_pro[idx]
                temp = CAM.deltaEmahalanobis(L_ref_v, a_ref_v, b_ref_v, L_pro_v, a_pro_v, b_pro_v)
                CD[idx] = temp
                if np.isnan(temp):
                    pass
                cd += temp  # wp * temp
                wpt += 1  # wp
    return CD, cd / wpt