import numpy as np
from skimage import color
import cython_functions
from scipy import stats
from scipy import ndimage
import my_utilities as MU
import compute_texture as CT
import matplotlib.pyplot as plt
import color_spaces as CAM



def conf00_colorfulness_diff_patches(Ref_image, Pro_image, th=0.005, r=1., min_num_pixels = 4, sq = False):
    SE = np.ones((np.int_(2 * r + 1), np.int_(2 * r + 1)))
    Ref_alpha = 1. * Ref_image[:, :, 0] - Ref_image[:, :, 1]
    Ref_beta = 0.5 * (1. * Ref_image[:, :, 0] + Ref_image[:, :, 1]) - Ref_image[:, :, 2]
    Pro_alpha = 1. * Pro_image[:, :, 0] - Pro_image[:, :, 1]
    Pro_beta = 0.5 * (1. * Pro_image[:, :, 0] + Pro_image[:, :, 1]) - Pro_image[:, :, 2]
    LBP, _ = CT.lbp(Ref_image, th=th, r=r)
    list_of_patterns = np.unique(LBP)
    cd = 0.
    wpt = 0.
    CD = np.zeros(Ref_alpha.shape)
    Np = Ref_alpha.size
    for pp in list_of_patterns:
        homo_patches = LBP == pp
        homo_patches = ndimage.morphology.binary_closing(homo_patches, SE)
        patch_index, num_patches = ndimage.measurements.label(homo_patches)
        for ii in range(1, num_patches):
            idx = np.where(patch_index == ii)
            if idx[0].size > min_num_pixels:
                c_text_ref = CAM.colorfulness_Gao(Ref_alpha[idx], Ref_beta[idx])
                c_text_pro = CAM.colorfulness_Gao(Pro_alpha[idx], Pro_beta[idx])
                if sq:
                    temp = np.power(c_text_ref-c_text_pro,2)
                else:
                    temp = np.abs(c_text_ref - c_text_pro)
                CD[idx] = temp
                if np.isnan(temp):
                    pass
                wp = (1. * idx[0].size) / Np
                cd += wp * temp
                wpt += 1  # wp
    return CD, cd


def conf01_colorfulness_diff_patches(Ref_image, Pro_image, th=0.005, r=1.5, min_num_pixels = 4, sq = False):
    SE = np.ones((np.int_(2 * r + 1), np.int_(2 * r + 1)))
    Ref_alpha = 1. * Ref_image[:, :, 0] - Ref_image[:, :, 1]
    Ref_beta = 0.5 * (1. * Ref_image[:, :, 0] + Ref_image[:, :, 1]) - Ref_image[:, :, 2]
    Pro_alpha = 1. * Pro_image[:, :, 0] - Pro_image[:, :, 1]
    Pro_beta = 0.5 * (1. * Pro_image[:, :, 0] + Pro_image[:, :, 1]) - Pro_image[:, :, 2]
    LBP, _ = CT.lbp(Ref_image, th=th, r=r)
    list_of_patterns = np.unique(LBP)
    cd = 0.
    wpt = 0.
    CD = np.zeros(Ref_alpha.shape)
    Np = Ref_alpha.size
    for pp in list_of_patterns:
        homo_patches = LBP == pp
        homo_patches = ndimage.morphology.binary_closing(homo_patches, SE)
        patch_index, num_patches = ndimage.measurements.label(homo_patches)
        for ii in range(1, num_patches):
            idx = np.where(patch_index == ii)
            if idx[0].size > min_num_pixels:
                c_text_ref = CAM.colorfulness_Gao(Ref_alpha[idx], Ref_beta[idx])
                c_text_pro = CAM.colorfulness_Gao(Pro_alpha[idx], Pro_beta[idx])
                if sq:
                    temp = np.power(c_text_ref-c_text_pro,2)
                else:
                    temp = np.abs(c_text_ref - c_text_pro)
                CD[idx] = temp
                if np.isnan(temp):
                    pass
                wp = (1. * idx[0].size) / Np
                cd += wp * temp
                wpt += 1  # wp
    return CD, cd


def conf02_colorfulness_diff_patches(Ref_image, Pro_image, th=0.005, r=2., min_num_pixels = 4, sq = False):
    SE = np.ones((np.int_(2 * r + 1), np.int_(2 * r + 1)))
    Ref_alpha = 1. * Ref_image[:, :, 0] - Ref_image[:, :, 1]
    Ref_beta = 0.5 * (1. * Ref_image[:, :, 0] + Ref_image[:, :, 1]) - Ref_image[:, :, 2]
    Pro_alpha = 1. * Pro_image[:, :, 0] - Pro_image[:, :, 1]
    Pro_beta = 0.5 * (1. * Pro_image[:, :, 0] + Pro_image[:, :, 1]) - Pro_image[:, :, 2]
    LBP, _ = CT.lbp(Ref_image, th=th, r=r)
    list_of_patterns = np.unique(LBP)
    cd = 0.
    wpt = 0.
    CD = np.zeros(Ref_alpha.shape)
    Np = Ref_alpha.size
    for pp in list_of_patterns:
        homo_patches = LBP == pp
        homo_patches = ndimage.morphology.binary_closing(homo_patches, SE)
        patch_index, num_patches = ndimage.measurements.label(homo_patches)
        for ii in range(1, num_patches):
            idx = np.where(patch_index == ii)
            if idx[0].size > min_num_pixels:
                c_text_ref = CAM.colorfulness_Gao(Ref_alpha[idx], Ref_beta[idx])
                c_text_pro = CAM.colorfulness_Gao(Pro_alpha[idx], Pro_beta[idx])
                if sq:
                    temp = np.power(c_text_ref-c_text_pro,2)
                else:
                    temp = np.abs(c_text_ref - c_text_pro)
                CD[idx] = temp
                if np.isnan(temp):
                    pass
                wp = (1. * idx[0].size) / Np
                cd += wp * temp
                wpt += 1  # wp
    return CD, cd


def conf03_colorfulness_diff_patches(Ref_image, Pro_image, th=0.005, r=3., min_num_pixels = 4, sq = False):
    SE = np.ones((np.int_(2 * r + 1), np.int_(2 * r + 1)))
    Ref_alpha = 1. * Ref_image[:, :, 0] - Ref_image[:, :, 1]
    Ref_beta = 0.5 * (1. * Ref_image[:, :, 0] + Ref_image[:, :, 1]) - Ref_image[:, :, 2]
    Pro_alpha = 1. * Pro_image[:, :, 0] - Pro_image[:, :, 1]
    Pro_beta = 0.5 * (1. * Pro_image[:, :, 0] + Pro_image[:, :, 1]) - Pro_image[:, :, 2]
    LBP, _ = CT.lbp(Ref_image, th=th, r=r)
    list_of_patterns = np.unique(LBP)
    cd = 0.
    wpt = 0.
    CD = np.zeros(Ref_alpha.shape)
    Np = Ref_alpha.size
    for pp in list_of_patterns:
        homo_patches = LBP == pp
        homo_patches = ndimage.morphology.binary_closing(homo_patches, SE)
        patch_index, num_patches = ndimage.measurements.label(homo_patches)
        for ii in range(1, num_patches):
            idx = np.where(patch_index == ii)
            if idx[0].size > min_num_pixels:
                c_text_ref = CAM.colorfulness_Gao(Ref_alpha[idx], Ref_beta[idx])
                c_text_pro = CAM.colorfulness_Gao(Pro_alpha[idx], Pro_beta[idx])
                if sq:
                    temp = np.power(c_text_ref-c_text_pro,2)
                else:
                    temp = np.abs(c_text_ref - c_text_pro)
                CD[idx] = temp
                if np.isnan(temp):
                    pass
                wp = (1. * idx[0].size) / Np
                cd += wp * temp
                wpt += 1  # wp
    return CD, cd




def conf04_colorfulness_diff_patches(Ref_image, Pro_image, th=0.01, r=1., min_num_pixels = 4, sq = False):
    SE = np.ones((np.int_(2 * r + 1), np.int_(2 * r + 1)))
    Ref_alpha = 1. * Ref_image[:, :, 0] - Ref_image[:, :, 1]
    Ref_beta = 0.5 * (1. * Ref_image[:, :, 0] + Ref_image[:, :, 1]) - Ref_image[:, :, 2]
    Pro_alpha = 1. * Pro_image[:, :, 0] - Pro_image[:, :, 1]
    Pro_beta = 0.5 * (1. * Pro_image[:, :, 0] + Pro_image[:, :, 1]) - Pro_image[:, :, 2]
    LBP, _ = CT.lbp(Ref_image, th=th, r=r)
    list_of_patterns = np.unique(LBP)
    cd = 0.
    wpt = 0.
    CD = np.zeros(Ref_alpha.shape)
    Np = Ref_alpha.size
    for pp in list_of_patterns:
        homo_patches = LBP == pp
        homo_patches = ndimage.morphology.binary_closing(homo_patches, SE)
        patch_index, num_patches = ndimage.measurements.label(homo_patches)
        for ii in range(1, num_patches):
            idx = np.where(patch_index == ii)
            if idx[0].size > min_num_pixels:
                c_text_ref = CAM.colorfulness_Gao(Ref_alpha[idx], Ref_beta[idx])
                c_text_pro = CAM.colorfulness_Gao(Pro_alpha[idx], Pro_beta[idx])
                if sq:
                    temp = np.power(c_text_ref-c_text_pro,2)
                else:
                    temp = np.abs(c_text_ref - c_text_pro)
                CD[idx] = temp
                if np.isnan(temp):
                    pass
                wp = (1. * idx[0].size) / Np
                cd += wp * temp
                wpt += 1  # wp
    return CD, cd


def conf05_colorfulness_diff_patches(Ref_image, Pro_image, th=0.01, r=1.5, min_num_pixels = 4, sq = False):
    SE = np.ones((np.int_(2 * r + 1), np.int_(2 * r + 1)))
    Ref_alpha = 1. * Ref_image[:, :, 0] - Ref_image[:, :, 1]
    Ref_beta = 0.5 * (1. * Ref_image[:, :, 0] + Ref_image[:, :, 1]) - Ref_image[:, :, 2]
    Pro_alpha = 1. * Pro_image[:, :, 0] - Pro_image[:, :, 1]
    Pro_beta = 0.5 * (1. * Pro_image[:, :, 0] + Pro_image[:, :, 1]) - Pro_image[:, :, 2]
    LBP, _ = CT.lbp(Ref_image, th=th, r=r)
    list_of_patterns = np.unique(LBP)
    cd = 0.
    wpt = 0.
    CD = np.zeros(Ref_alpha.shape)
    Np = Ref_alpha.size
    for pp in list_of_patterns:
        homo_patches = LBP == pp
        homo_patches = ndimage.morphology.binary_closing(homo_patches, SE)
        patch_index, num_patches = ndimage.measurements.label(homo_patches)
        for ii in range(1, num_patches):
            idx = np.where(patch_index == ii)
            if idx[0].size > min_num_pixels:
                c_text_ref = CAM.colorfulness_Gao(Ref_alpha[idx], Ref_beta[idx])
                c_text_pro = CAM.colorfulness_Gao(Pro_alpha[idx], Pro_beta[idx])
                if sq:
                    temp = np.power(c_text_ref-c_text_pro,2)
                else:
                    temp = np.abs(c_text_ref - c_text_pro)
                CD[idx] = temp
                if np.isnan(temp):
                    pass
                wp = (1. * idx[0].size) / Np
                cd += wp * temp
                wpt += 1  # wp
    return CD, cd


def conf06_colorfulness_diff_patches(Ref_image, Pro_image, th=0.01, r=2., min_num_pixels = 4, sq = False):
    SE = np.ones((np.int_(2 * r + 1), np.int_(2 * r + 1)))
    Ref_alpha = 1. * Ref_image[:, :, 0] - Ref_image[:, :, 1]
    Ref_beta = 0.5 * (1. * Ref_image[:, :, 0] + Ref_image[:, :, 1]) - Ref_image[:, :, 2]
    Pro_alpha = 1. * Pro_image[:, :, 0] - Pro_image[:, :, 1]
    Pro_beta = 0.5 * (1. * Pro_image[:, :, 0] + Pro_image[:, :, 1]) - Pro_image[:, :, 2]
    LBP, _ = CT.lbp(Ref_image, th=th, r=r)
    list_of_patterns = np.unique(LBP)
    cd = 0.
    wpt = 0.
    CD = np.zeros(Ref_alpha.shape)
    Np = Ref_alpha.size
    for pp in list_of_patterns:
        homo_patches = LBP == pp
        homo_patches = ndimage.morphology.binary_closing(homo_patches, SE)
        patch_index, num_patches = ndimage.measurements.label(homo_patches)
        for ii in range(1, num_patches):
            idx = np.where(patch_index == ii)
            if idx[0].size > min_num_pixels:
                c_text_ref = CAM.colorfulness_Gao(Ref_alpha[idx], Ref_beta[idx])
                c_text_pro = CAM.colorfulness_Gao(Pro_alpha[idx], Pro_beta[idx])
                if sq:
                    temp = np.power(c_text_ref-c_text_pro,2)
                else:
                    temp = np.abs(c_text_ref - c_text_pro)
                CD[idx] = temp
                if np.isnan(temp):
                    pass
                wp = (1. * idx[0].size) / Np
                cd += wp * temp
                wpt += 1  # wp
    return CD, cd


def conf07_colorfulness_diff_patches(Ref_image, Pro_image, th=0.01, r=3., min_num_pixels = 4, sq = False):
    SE = np.ones((np.int_(2 * r + 1), np.int_(2 * r + 1)))
    Ref_alpha = 1. * Ref_image[:, :, 0] - Ref_image[:, :, 1]
    Ref_beta = 0.5 * (1. * Ref_image[:, :, 0] + Ref_image[:, :, 1]) - Ref_image[:, :, 2]
    Pro_alpha = 1. * Pro_image[:, :, 0] - Pro_image[:, :, 1]
    Pro_beta = 0.5 * (1. * Pro_image[:, :, 0] + Pro_image[:, :, 1]) - Pro_image[:, :, 2]
    LBP, _ = CT.lbp(Ref_image, th=th, r=r)
    list_of_patterns = np.unique(LBP)
    cd = 0.
    wpt = 0.
    CD = np.zeros(Ref_alpha.shape)
    Np = Ref_alpha.size
    for pp in list_of_patterns:
        homo_patches = LBP == pp
        homo_patches = ndimage.morphology.binary_closing(homo_patches, SE)
        patch_index, num_patches = ndimage.measurements.label(homo_patches)
        for ii in range(1, num_patches):
            idx = np.where(patch_index == ii)
            if idx[0].size > min_num_pixels:
                c_text_ref = CAM.colorfulness_Gao(Ref_alpha[idx], Ref_beta[idx])
                c_text_pro = CAM.colorfulness_Gao(Pro_alpha[idx], Pro_beta[idx])
                if sq:
                    temp = np.power(c_text_ref-c_text_pro,2)
                else:
                    temp = np.abs(c_text_ref - c_text_pro)
                CD[idx] = temp
                if np.isnan(temp):
                    pass
                wp = (1. * idx[0].size) / Np
                cd += wp * temp
                wpt += 1  # wp
    return CD, cd



def conf08_colorfulness_diff_patches(Ref_image, Pro_image, th=0.015, r=1., min_num_pixels = 4, sq = False):
    SE = np.ones((np.int_(2 * r + 1), np.int_(2 * r + 1)))
    Ref_alpha = 1. * Ref_image[:, :, 0] - Ref_image[:, :, 1]
    Ref_beta = 0.5 * (1. * Ref_image[:, :, 0] + Ref_image[:, :, 1]) - Ref_image[:, :, 2]
    Pro_alpha = 1. * Pro_image[:, :, 0] - Pro_image[:, :, 1]
    Pro_beta = 0.5 * (1. * Pro_image[:, :, 0] + Pro_image[:, :, 1]) - Pro_image[:, :, 2]
    LBP, _ = CT.lbp(Ref_image, th=th, r=r)
    list_of_patterns = np.unique(LBP)
    cd = 0.
    wpt = 0.
    CD = np.zeros(Ref_alpha.shape)
    Np = Ref_alpha.size
    for pp in list_of_patterns:
        homo_patches = LBP == pp
        homo_patches = ndimage.morphology.binary_closing(homo_patches, SE)
        patch_index, num_patches = ndimage.measurements.label(homo_patches)
        for ii in range(1, num_patches):
            idx = np.where(patch_index == ii)
            if idx[0].size > min_num_pixels:
                c_text_ref = CAM.colorfulness_Gao(Ref_alpha[idx], Ref_beta[idx])
                c_text_pro = CAM.colorfulness_Gao(Pro_alpha[idx], Pro_beta[idx])
                if sq:
                    temp = np.power(c_text_ref-c_text_pro,2)
                else:
                    temp = np.abs(c_text_ref - c_text_pro)
                CD[idx] = temp
                if np.isnan(temp):
                    pass
                wp = (1. * idx[0].size) / Np
                cd += wp * temp
                wpt += 1  # wp
    return CD, cd


def conf09_colorfulness_diff_patches(Ref_image, Pro_image, th=0.015, r=1.5, min_num_pixels = 4, sq = False):
    SE = np.ones((np.int_(2 * r + 1), np.int_(2 * r + 1)))
    Ref_alpha = 1. * Ref_image[:, :, 0] - Ref_image[:, :, 1]
    Ref_beta = 0.5 * (1. * Ref_image[:, :, 0] + Ref_image[:, :, 1]) - Ref_image[:, :, 2]
    Pro_alpha = 1. * Pro_image[:, :, 0] - Pro_image[:, :, 1]
    Pro_beta = 0.5 * (1. * Pro_image[:, :, 0] + Pro_image[:, :, 1]) - Pro_image[:, :, 2]
    LBP, _ = CT.lbp(Ref_image, th=th, r=r)
    list_of_patterns = np.unique(LBP)
    cd = 0.
    wpt = 0.
    CD = np.zeros(Ref_alpha.shape)
    Np = Ref_alpha.size
    for pp in list_of_patterns:
        homo_patches = LBP == pp
        homo_patches = ndimage.morphology.binary_closing(homo_patches, SE)
        patch_index, num_patches = ndimage.measurements.label(homo_patches)
        for ii in range(1, num_patches):
            idx = np.where(patch_index == ii)
            if idx[0].size > min_num_pixels:
                c_text_ref = CAM.colorfulness_Gao(Ref_alpha[idx], Ref_beta[idx])
                c_text_pro = CAM.colorfulness_Gao(Pro_alpha[idx], Pro_beta[idx])
                if sq:
                    temp = np.power(c_text_ref-c_text_pro,2)
                else:
                    temp = np.abs(c_text_ref - c_text_pro)
                CD[idx] = temp
                if np.isnan(temp):
                    pass
                wp = (1. * idx[0].size) / Np
                cd += wp * temp
                wpt += 1  # wp
    return CD, cd


def conf10_colorfulness_diff_patches(Ref_image, Pro_image, th=0.015, r=2., min_num_pixels = 4, sq = False):
    SE = np.ones((np.int_(2 * r + 1), np.int_(2 * r + 1)))
    Ref_alpha = 1. * Ref_image[:, :, 0] - Ref_image[:, :, 1]
    Ref_beta = 0.5 * (1. * Ref_image[:, :, 0] + Ref_image[:, :, 1]) - Ref_image[:, :, 2]
    Pro_alpha = 1. * Pro_image[:, :, 0] - Pro_image[:, :, 1]
    Pro_beta = 0.5 * (1. * Pro_image[:, :, 0] + Pro_image[:, :, 1]) - Pro_image[:, :, 2]
    LBP, _ = CT.lbp(Ref_image, th=th, r=r)
    list_of_patterns = np.unique(LBP)
    cd = 0.
    wpt = 0.
    CD = np.zeros(Ref_alpha.shape)
    Np = Ref_alpha.size
    for pp in list_of_patterns:
        homo_patches = LBP == pp
        homo_patches = ndimage.morphology.binary_closing(homo_patches, SE)
        patch_index, num_patches = ndimage.measurements.label(homo_patches)
        for ii in range(1, num_patches):
            idx = np.where(patch_index == ii)
            if idx[0].size > min_num_pixels:
                c_text_ref = CAM.colorfulness_Gao(Ref_alpha[idx], Ref_beta[idx])
                c_text_pro = CAM.colorfulness_Gao(Pro_alpha[idx], Pro_beta[idx])
                if sq:
                    temp = np.power(c_text_ref-c_text_pro,2)
                else:
                    temp = np.abs(c_text_ref - c_text_pro)
                CD[idx] = temp
                if np.isnan(temp):
                    pass
                wp = (1. * idx[0].size) / Np
                cd += wp * temp
                wpt += 1  # wp
    return CD, cd


def conf11_colorfulness_diff_patches(Ref_image, Pro_image, th=0.015, r=3., min_num_pixels = 4, sq = False):
    SE = np.ones((np.int_(2 * r + 1), np.int_(2 * r + 1)))
    Ref_alpha = 1. * Ref_image[:, :, 0] - Ref_image[:, :, 1]
    Ref_beta = 0.5 * (1. * Ref_image[:, :, 0] + Ref_image[:, :, 1]) - Ref_image[:, :, 2]
    Pro_alpha = 1. * Pro_image[:, :, 0] - Pro_image[:, :, 1]
    Pro_beta = 0.5 * (1. * Pro_image[:, :, 0] + Pro_image[:, :, 1]) - Pro_image[:, :, 2]
    LBP, _ = CT.lbp(Ref_image, th=th, r=r)
    list_of_patterns = np.unique(LBP)
    cd = 0.
    wpt = 0.
    CD = np.zeros(Ref_alpha.shape)
    Np = Ref_alpha.size
    for pp in list_of_patterns:
        homo_patches = LBP == pp
        homo_patches = ndimage.morphology.binary_closing(homo_patches, SE)
        patch_index, num_patches = ndimage.measurements.label(homo_patches)
        for ii in range(1, num_patches):
            idx = np.where(patch_index == ii)
            if idx[0].size > min_num_pixels:
                c_text_ref = CAM.colorfulness_Gao(Ref_alpha[idx], Ref_beta[idx])
                c_text_pro = CAM.colorfulness_Gao(Pro_alpha[idx], Pro_beta[idx])
                if sq:
                    temp = np.power(c_text_ref-c_text_pro,2)
                else:
                    temp = np.abs(c_text_ref - c_text_pro)
                CD[idx] = temp
                if np.isnan(temp):
                    pass
                wp = (1. * idx[0].size) / Np
                cd += wp * temp
                wpt += 1  # wp
    return CD, cd



def conf12_colorfulness_diff_patches(Ref_image, Pro_image, th=0.02, r=1., min_num_pixels = 4, sq = False):
    SE = np.ones((np.int_(2 * r + 1), np.int_(2 * r + 1)))
    Ref_alpha = 1. * Ref_image[:, :, 0] - Ref_image[:, :, 1]
    Ref_beta = 0.5 * (1. * Ref_image[:, :, 0] + Ref_image[:, :, 1]) - Ref_image[:, :, 2]
    Pro_alpha = 1. * Pro_image[:, :, 0] - Pro_image[:, :, 1]
    Pro_beta = 0.5 * (1. * Pro_image[:, :, 0] + Pro_image[:, :, 1]) - Pro_image[:, :, 2]
    LBP, _ = CT.lbp(Ref_image, th=th, r=r)
    list_of_patterns = np.unique(LBP)
    cd = 0.
    wpt = 0.
    CD = np.zeros(Ref_alpha.shape)
    Np = Ref_alpha.size
    for pp in list_of_patterns:
        homo_patches = LBP == pp
        homo_patches = ndimage.morphology.binary_closing(homo_patches, SE)
        patch_index, num_patches = ndimage.measurements.label(homo_patches)
        for ii in range(1, num_patches):
            idx = np.where(patch_index == ii)
            if idx[0].size > min_num_pixels:
                c_text_ref = CAM.colorfulness_Gao(Ref_alpha[idx], Ref_beta[idx])
                c_text_pro = CAM.colorfulness_Gao(Pro_alpha[idx], Pro_beta[idx])
                if sq:
                    temp = np.power(c_text_ref-c_text_pro,2)
                else:
                    temp = np.abs(c_text_ref - c_text_pro)
                CD[idx] = temp
                if np.isnan(temp):
                    pass
                wp = (1. * idx[0].size) / Np
                cd += wp * temp
                wpt += 1  # wp
    return CD, cd


def conf13_colorfulness_diff_patches(Ref_image, Pro_image, th=0.02, r=1.5, min_num_pixels = 4, sq = False):
    SE = np.ones((np.int_(2 * r + 1), np.int_(2 * r + 1)))
    Ref_alpha = 1. * Ref_image[:, :, 0] - Ref_image[:, :, 1]
    Ref_beta = 0.5 * (1. * Ref_image[:, :, 0] + Ref_image[:, :, 1]) - Ref_image[:, :, 2]
    Pro_alpha = 1. * Pro_image[:, :, 0] - Pro_image[:, :, 1]
    Pro_beta = 0.5 * (1. * Pro_image[:, :, 0] + Pro_image[:, :, 1]) - Pro_image[:, :, 2]
    LBP, _ = CT.lbp(Ref_image, th=th, r=r)
    list_of_patterns = np.unique(LBP)
    cd = 0.
    wpt = 0.
    CD = np.zeros(Ref_alpha.shape)
    Np = Ref_alpha.size
    for pp in list_of_patterns:
        homo_patches = LBP == pp
        homo_patches = ndimage.morphology.binary_closing(homo_patches, SE)
        patch_index, num_patches = ndimage.measurements.label(homo_patches)
        for ii in range(1, num_patches):
            idx = np.where(patch_index == ii)
            if idx[0].size > min_num_pixels:
                c_text_ref = CAM.colorfulness_Gao(Ref_alpha[idx], Ref_beta[idx])
                c_text_pro = CAM.colorfulness_Gao(Pro_alpha[idx], Pro_beta[idx])
                if sq:
                    temp = np.power(c_text_ref-c_text_pro,2)
                else:
                    temp = np.abs(c_text_ref - c_text_pro)
                CD[idx] = temp
                if np.isnan(temp):
                    pass
                wp = (1. * idx[0].size) / Np
                cd += wp * temp
                wpt += 1  # wp
    return CD, cd


def conf14_colorfulness_diff_patches(Ref_image, Pro_image, th=0.02, r=2., min_num_pixels = 4, sq = False):
    SE = np.ones((np.int_(2 * r + 1), np.int_(2 * r + 1)))
    Ref_alpha = 1. * Ref_image[:, :, 0] - Ref_image[:, :, 1]
    Ref_beta = 0.5 * (1. * Ref_image[:, :, 0] + Ref_image[:, :, 1]) - Ref_image[:, :, 2]
    Pro_alpha = 1. * Pro_image[:, :, 0] - Pro_image[:, :, 1]
    Pro_beta = 0.5 * (1. * Pro_image[:, :, 0] + Pro_image[:, :, 1]) - Pro_image[:, :, 2]
    LBP, _ = CT.lbp(Ref_image, th=th, r=r)
    list_of_patterns = np.unique(LBP)
    cd = 0.
    wpt = 0.
    CD = np.zeros(Ref_alpha.shape)
    Np = Ref_alpha.size
    for pp in list_of_patterns:
        homo_patches = LBP == pp
        homo_patches = ndimage.morphology.binary_closing(homo_patches, SE)
        patch_index, num_patches = ndimage.measurements.label(homo_patches)
        for ii in range(1, num_patches):
            idx = np.where(patch_index == ii)
            if idx[0].size > min_num_pixels:
                c_text_ref = CAM.colorfulness_Gao(Ref_alpha[idx], Ref_beta[idx])
                c_text_pro = CAM.colorfulness_Gao(Pro_alpha[idx], Pro_beta[idx])
                if sq:
                    temp = np.power(c_text_ref-c_text_pro,2)
                else:
                    temp = np.abs(c_text_ref - c_text_pro)
                CD[idx] = temp
                if np.isnan(temp):
                    pass
                wp = (1. * idx[0].size) / Np
                cd += wp * temp
                wpt += 1  # wp
    return CD, cd


def conf15_colorfulness_diff_patches(Ref_image, Pro_image, th=0.02, r=3., min_num_pixels = 4, sq = False):
    SE = np.ones((np.int_(2 * r + 1), np.int_(2 * r + 1)))
    Ref_alpha = 1. * Ref_image[:, :, 0] - Ref_image[:, :, 1]
    Ref_beta = 0.5 * (1. * Ref_image[:, :, 0] + Ref_image[:, :, 1]) - Ref_image[:, :, 2]
    Pro_alpha = 1. * Pro_image[:, :, 0] - Pro_image[:, :, 1]
    Pro_beta = 0.5 * (1. * Pro_image[:, :, 0] + Pro_image[:, :, 1]) - Pro_image[:, :, 2]
    LBP, _ = CT.lbp(Ref_image, th=th, r=r)
    list_of_patterns = np.unique(LBP)
    cd = 0.
    wpt = 0.
    CD = np.zeros(Ref_alpha.shape)
    Np = Ref_alpha.size
    for pp in list_of_patterns:
        homo_patches = LBP == pp
        homo_patches = ndimage.morphology.binary_closing(homo_patches, SE)
        patch_index, num_patches = ndimage.measurements.label(homo_patches)
        for ii in range(1, num_patches):
            idx = np.where(patch_index == ii)
            if idx[0].size > min_num_pixels:
                c_text_ref = CAM.colorfulness_Gao(Ref_alpha[idx], Ref_beta[idx])
                c_text_pro = CAM.colorfulness_Gao(Pro_alpha[idx], Pro_beta[idx])
                if sq:
                    temp = np.power(c_text_ref-c_text_pro,2)
                else:
                    temp = np.abs(c_text_ref - c_text_pro)
                CD[idx] = temp
                if np.isnan(temp):
                    pass
                wp = (1. * idx[0].size) / Np
                cd += wp * temp
                wpt += 1  # wp
    return CD, cd



def conf16_colorfulness_diff_patches(Ref_image, Pro_image, th=0.025, r=1., min_num_pixels = 4, sq = False):
    SE = np.ones((np.int_(2 * r + 1), np.int_(2 * r + 1)))
    Ref_alpha = 1. * Ref_image[:, :, 0] - Ref_image[:, :, 1]
    Ref_beta = 0.5 * (1. * Ref_image[:, :, 0] + Ref_image[:, :, 1]) - Ref_image[:, :, 2]
    Pro_alpha = 1. * Pro_image[:, :, 0] - Pro_image[:, :, 1]
    Pro_beta = 0.5 * (1. * Pro_image[:, :, 0] + Pro_image[:, :, 1]) - Pro_image[:, :, 2]
    LBP, _ = CT.lbp(Ref_image, th=th, r=r)
    list_of_patterns = np.unique(LBP)
    cd = 0.
    wpt = 0.
    CD = np.zeros(Ref_alpha.shape)
    Np = Ref_alpha.size
    for pp in list_of_patterns:
        homo_patches = LBP == pp
        homo_patches = ndimage.morphology.binary_closing(homo_patches, SE)
        patch_index, num_patches = ndimage.measurements.label(homo_patches)
        for ii in range(1, num_patches):
            idx = np.where(patch_index == ii)
            if idx[0].size > min_num_pixels:
                c_text_ref = CAM.colorfulness_Gao(Ref_alpha[idx], Ref_beta[idx])
                c_text_pro = CAM.colorfulness_Gao(Pro_alpha[idx], Pro_beta[idx])
                if sq:
                    temp = np.power(c_text_ref-c_text_pro,2)
                else:
                    temp = np.abs(c_text_ref - c_text_pro)
                CD[idx] = temp
                if np.isnan(temp):
                    pass
                wp = (1. * idx[0].size) / Np
                cd += wp * temp
                wpt += 1  # wp
    return CD, cd


def conf17_colorfulness_diff_patches(Ref_image, Pro_image, th=0.025, r=1.5, min_num_pixels = 4, sq = False):
    SE = np.ones((np.int_(2 * r + 1), np.int_(2 * r + 1)))
    Ref_alpha = 1. * Ref_image[:, :, 0] - Ref_image[:, :, 1]
    Ref_beta = 0.5 * (1. * Ref_image[:, :, 0] + Ref_image[:, :, 1]) - Ref_image[:, :, 2]
    Pro_alpha = 1. * Pro_image[:, :, 0] - Pro_image[:, :, 1]
    Pro_beta = 0.5 * (1. * Pro_image[:, :, 0] + Pro_image[:, :, 1]) - Pro_image[:, :, 2]
    LBP, _ = CT.lbp(Ref_image, th=th, r=r)
    list_of_patterns = np.unique(LBP)
    cd = 0.
    wpt = 0.
    CD = np.zeros(Ref_alpha.shape)
    Np = Ref_alpha.size
    for pp in list_of_patterns:
        homo_patches = LBP == pp
        homo_patches = ndimage.morphology.binary_closing(homo_patches, SE)
        patch_index, num_patches = ndimage.measurements.label(homo_patches)
        for ii in range(1, num_patches):
            idx = np.where(patch_index == ii)
            if idx[0].size > min_num_pixels:
                c_text_ref = CAM.colorfulness_Gao(Ref_alpha[idx], Ref_beta[idx])
                c_text_pro = CAM.colorfulness_Gao(Pro_alpha[idx], Pro_beta[idx])
                if sq:
                    temp = np.power(c_text_ref-c_text_pro,2)
                else:
                    temp = np.abs(c_text_ref - c_text_pro)
                CD[idx] = temp
                if np.isnan(temp):
                    pass
                wp = (1. * idx[0].size) / Np
                cd += wp * temp
                wpt += 1  # wp
    return CD, cd


def conf18_colorfulness_diff_patches(Ref_image, Pro_image, th=0.025, r=2., min_num_pixels = 4, sq = False):
    SE = np.ones((np.int_(2 * r + 1), np.int_(2 * r + 1)))
    Ref_alpha = 1. * Ref_image[:, :, 0] - Ref_image[:, :, 1]
    Ref_beta = 0.5 * (1. * Ref_image[:, :, 0] + Ref_image[:, :, 1]) - Ref_image[:, :, 2]
    Pro_alpha = 1. * Pro_image[:, :, 0] - Pro_image[:, :, 1]
    Pro_beta = 0.5 * (1. * Pro_image[:, :, 0] + Pro_image[:, :, 1]) - Pro_image[:, :, 2]
    LBP, _ = CT.lbp(Ref_image, th=th, r=r)
    list_of_patterns = np.unique(LBP)
    cd = 0.
    wpt = 0.
    CD = np.zeros(Ref_alpha.shape)
    Np = Ref_alpha.size
    for pp in list_of_patterns:
        homo_patches = LBP == pp
        homo_patches = ndimage.morphology.binary_closing(homo_patches, SE)
        patch_index, num_patches = ndimage.measurements.label(homo_patches)
        for ii in range(1, num_patches):
            idx = np.where(patch_index == ii)
            if idx[0].size > min_num_pixels:
                c_text_ref = CAM.colorfulness_Gao(Ref_alpha[idx], Ref_beta[idx])
                c_text_pro = CAM.colorfulness_Gao(Pro_alpha[idx], Pro_beta[idx])
                if sq:
                    temp = np.power(c_text_ref-c_text_pro,2)
                else:
                    temp = np.abs(c_text_ref - c_text_pro)
                CD[idx] = temp
                if np.isnan(temp):
                    pass
                wp = (1. * idx[0].size) / Np
                cd += wp * temp
                wpt += 1  # wp
    return CD, cd


def conf19_colorfulness_diff_patches(Ref_image, Pro_image, th=0.025, r=3., min_num_pixels = 4, sq = False):
    SE = np.ones((np.int_(2 * r + 1), np.int_(2 * r + 1)))
    Ref_alpha = 1. * Ref_image[:, :, 0] - Ref_image[:, :, 1]
    Ref_beta = 0.5 * (1. * Ref_image[:, :, 0] + Ref_image[:, :, 1]) - Ref_image[:, :, 2]
    Pro_alpha = 1. * Pro_image[:, :, 0] - Pro_image[:, :, 1]
    Pro_beta = 0.5 * (1. * Pro_image[:, :, 0] + Pro_image[:, :, 1]) - Pro_image[:, :, 2]
    LBP, _ = CT.lbp(Ref_image, th=th, r=r)
    list_of_patterns = np.unique(LBP)
    cd = 0.
    wpt = 0.
    CD = np.zeros(Ref_alpha.shape)
    Np = Ref_alpha.size
    for pp in list_of_patterns:
        homo_patches = LBP == pp
        homo_patches = ndimage.morphology.binary_closing(homo_patches, SE)
        patch_index, num_patches = ndimage.measurements.label(homo_patches)
        for ii in range(1, num_patches):
            idx = np.where(patch_index == ii)
            if idx[0].size > min_num_pixels:
                c_text_ref = CAM.colorfulness_Gao(Ref_alpha[idx], Ref_beta[idx])
                c_text_pro = CAM.colorfulness_Gao(Pro_alpha[idx], Pro_beta[idx])
                if sq:
                    temp = np.power(c_text_ref-c_text_pro,2)
                else:
                    temp = np.abs(c_text_ref - c_text_pro)
                CD[idx] = temp
                if np.isnan(temp):
                    pass
                wp = (1. * idx[0].size) / Np
                cd += wp * temp
                wpt += 1  # wp
    return CD, cd



def conf20_colorfulness_diff_patches(Ref_image, Pro_image, th=0.03, r=1., min_num_pixels = 4, sq = False):
    SE = np.ones((np.int_(2 * r + 1), np.int_(2 * r + 1)))
    Ref_alpha = 1. * Ref_image[:, :, 0] - Ref_image[:, :, 1]
    Ref_beta = 0.5 * (1. * Ref_image[:, :, 0] + Ref_image[:, :, 1]) - Ref_image[:, :, 2]
    Pro_alpha = 1. * Pro_image[:, :, 0] - Pro_image[:, :, 1]
    Pro_beta = 0.5 * (1. * Pro_image[:, :, 0] + Pro_image[:, :, 1]) - Pro_image[:, :, 2]
    LBP, _ = CT.lbp(Ref_image, th=th, r=r)
    list_of_patterns = np.unique(LBP)
    cd = 0.
    wpt = 0.
    CD = np.zeros(Ref_alpha.shape)
    Np = Ref_alpha.size
    for pp in list_of_patterns:
        homo_patches = LBP == pp
        homo_patches = ndimage.morphology.binary_closing(homo_patches, SE)
        patch_index, num_patches = ndimage.measurements.label(homo_patches)
        for ii in range(1, num_patches):
            idx = np.where(patch_index == ii)
            if idx[0].size > min_num_pixels:
                c_text_ref = CAM.colorfulness_Gao(Ref_alpha[idx], Ref_beta[idx])
                c_text_pro = CAM.colorfulness_Gao(Pro_alpha[idx], Pro_beta[idx])
                if sq:
                    temp = np.power(c_text_ref-c_text_pro,2)
                else:
                    temp = np.abs(c_text_ref - c_text_pro)
                CD[idx] = temp
                if np.isnan(temp):
                    pass
                wp = (1. * idx[0].size) / Np
                cd += wp * temp
                wpt += 1  # wp
    return CD, cd


def conf21_colorfulness_diff_patches(Ref_image, Pro_image, th=0.03, r=1.5, min_num_pixels = 4, sq = False):
    SE = np.ones((np.int_(2 * r + 1), np.int_(2 * r + 1)))
    Ref_alpha = 1. * Ref_image[:, :, 0] - Ref_image[:, :, 1]
    Ref_beta = 0.5 * (1. * Ref_image[:, :, 0] + Ref_image[:, :, 1]) - Ref_image[:, :, 2]
    Pro_alpha = 1. * Pro_image[:, :, 0] - Pro_image[:, :, 1]
    Pro_beta = 0.5 * (1. * Pro_image[:, :, 0] + Pro_image[:, :, 1]) - Pro_image[:, :, 2]
    LBP, _ = CT.lbp(Ref_image, th=th, r=r)
    list_of_patterns = np.unique(LBP)
    cd = 0.
    wpt = 0.
    CD = np.zeros(Ref_alpha.shape)
    Np = Ref_alpha.size
    for pp in list_of_patterns:
        homo_patches = LBP == pp
        homo_patches = ndimage.morphology.binary_closing(homo_patches, SE)
        patch_index, num_patches = ndimage.measurements.label(homo_patches)
        for ii in range(1, num_patches):
            idx = np.where(patch_index == ii)
            if idx[0].size > min_num_pixels:
                c_text_ref = CAM.colorfulness_Gao(Ref_alpha[idx], Ref_beta[idx])
                c_text_pro = CAM.colorfulness_Gao(Pro_alpha[idx], Pro_beta[idx])
                if sq:
                    temp = np.power(c_text_ref-c_text_pro,2)
                else:
                    temp = np.abs(c_text_ref - c_text_pro)
                CD[idx] = temp
                if np.isnan(temp):
                    pass
                wp = (1. * idx[0].size) / Np
                cd += wp * temp
                wpt += 1  # wp
    return CD, cd


def conf22_colorfulness_diff_patches(Ref_image, Pro_image, th=0.03, r=2., min_num_pixels = 4, sq = False):
    SE = np.ones((np.int_(2 * r + 1), np.int_(2 * r + 1)))
    Ref_alpha = 1. * Ref_image[:, :, 0] - Ref_image[:, :, 1]
    Ref_beta = 0.5 * (1. * Ref_image[:, :, 0] + Ref_image[:, :, 1]) - Ref_image[:, :, 2]
    Pro_alpha = 1. * Pro_image[:, :, 0] - Pro_image[:, :, 1]
    Pro_beta = 0.5 * (1. * Pro_image[:, :, 0] + Pro_image[:, :, 1]) - Pro_image[:, :, 2]
    LBP, _ = CT.lbp(Ref_image, th=th, r=r)
    list_of_patterns = np.unique(LBP)
    cd = 0.
    wpt = 0.
    CD = np.zeros(Ref_alpha.shape)
    Np = Ref_alpha.size
    for pp in list_of_patterns:
        homo_patches = LBP == pp
        homo_patches = ndimage.morphology.binary_closing(homo_patches, SE)
        patch_index, num_patches = ndimage.measurements.label(homo_patches)
        for ii in range(1, num_patches):
            idx = np.where(patch_index == ii)
            if idx[0].size > min_num_pixels:
                c_text_ref = CAM.colorfulness_Gao(Ref_alpha[idx], Ref_beta[idx])
                c_text_pro = CAM.colorfulness_Gao(Pro_alpha[idx], Pro_beta[idx])
                if sq:
                    temp = np.power(c_text_ref-c_text_pro,2)
                else:
                    temp = np.abs(c_text_ref - c_text_pro)
                CD[idx] = temp
                if np.isnan(temp):
                    pass
                wp = (1. * idx[0].size) / Np
                cd += wp * temp
                wpt += 1  # wp
    return CD, cd


def conf23_colorfulness_diff_patches(Ref_image, Pro_image, th=0.03, r=3., min_num_pixels = 4, sq = False):
    SE = np.ones((np.int_(2 * r + 1), np.int_(2 * r + 1)))
    Ref_alpha = 1. * Ref_image[:, :, 0] - Ref_image[:, :, 1]
    Ref_beta = 0.5 * (1. * Ref_image[:, :, 0] + Ref_image[:, :, 1]) - Ref_image[:, :, 2]
    Pro_alpha = 1. * Pro_image[:, :, 0] - Pro_image[:, :, 1]
    Pro_beta = 0.5 * (1. * Pro_image[:, :, 0] + Pro_image[:, :, 1]) - Pro_image[:, :, 2]
    LBP, _ = CT.lbp(Ref_image, th=th, r=r)
    list_of_patterns = np.unique(LBP)
    cd = 0.
    wpt = 0.
    CD = np.zeros(Ref_alpha.shape)
    Np = Ref_alpha.size
    for pp in list_of_patterns:
        homo_patches = LBP == pp
        homo_patches = ndimage.morphology.binary_closing(homo_patches, SE)
        patch_index, num_patches = ndimage.measurements.label(homo_patches)
        for ii in range(1, num_patches):
            idx = np.where(patch_index == ii)
            if idx[0].size > min_num_pixels:
                c_text_ref = CAM.colorfulness_Gao(Ref_alpha[idx], Ref_beta[idx])
                c_text_pro = CAM.colorfulness_Gao(Pro_alpha[idx], Pro_beta[idx])
                if sq:
                    temp = np.power(c_text_ref-c_text_pro,2)
                else:
                    temp = np.abs(c_text_ref - c_text_pro)
                CD[idx] = temp
                if np.isnan(temp):
                    pass
                wp = (1. * idx[0].size) / Np
                cd += wp * temp
                wpt += 1  # wp
    return CD, cd


def conf24_colorfulness_diff_patches(Ref_image, Pro_image, th=0.035, r=1., min_num_pixels = 4, sq = False):
    SE = np.ones((np.int_(2 * r + 1), np.int_(2 * r + 1)))
    Ref_alpha = 1. * Ref_image[:, :, 0] - Ref_image[:, :, 1]
    Ref_beta = 0.5 * (1. * Ref_image[:, :, 0] + Ref_image[:, :, 1]) - Ref_image[:, :, 2]
    Pro_alpha = 1. * Pro_image[:, :, 0] - Pro_image[:, :, 1]
    Pro_beta = 0.5 * (1. * Pro_image[:, :, 0] + Pro_image[:, :, 1]) - Pro_image[:, :, 2]
    LBP, _ = CT.lbp(Ref_image, th=th, r=r)
    list_of_patterns = np.unique(LBP)
    cd = 0.
    wpt = 0.
    CD = np.zeros(Ref_alpha.shape)
    Np = Ref_alpha.size
    for pp in list_of_patterns:
        homo_patches = LBP == pp
        homo_patches = ndimage.morphology.binary_closing(homo_patches, SE)
        patch_index, num_patches = ndimage.measurements.label(homo_patches)
        for ii in range(1, num_patches):
            idx = np.where(patch_index == ii)
            if idx[0].size > min_num_pixels:
                c_text_ref = CAM.colorfulness_Gao(Ref_alpha[idx], Ref_beta[idx])
                c_text_pro = CAM.colorfulness_Gao(Pro_alpha[idx], Pro_beta[idx])
                if sq:
                    temp = np.power(c_text_ref-c_text_pro,2)
                else:
                    temp = np.abs(c_text_ref - c_text_pro)
                CD[idx] = temp
                if np.isnan(temp):
                    pass
                wp = (1. * idx[0].size) / Np
                cd += wp * temp
                wpt += 1  # wp
    return CD, cd


def conf25_colorfulness_diff_patches(Ref_image, Pro_image, th=0.035, r=1.5, min_num_pixels = 4, sq = False):
    SE = np.ones((np.int_(2 * r + 1), np.int_(2 * r + 1)))
    Ref_alpha = 1. * Ref_image[:, :, 0] - Ref_image[:, :, 1]
    Ref_beta = 0.5 * (1. * Ref_image[:, :, 0] + Ref_image[:, :, 1]) - Ref_image[:, :, 2]
    Pro_alpha = 1. * Pro_image[:, :, 0] - Pro_image[:, :, 1]
    Pro_beta = 0.5 * (1. * Pro_image[:, :, 0] + Pro_image[:, :, 1]) - Pro_image[:, :, 2]
    LBP, _ = CT.lbp(Ref_image, th=th, r=r)
    list_of_patterns = np.unique(LBP)
    cd = 0.
    wpt = 0.
    CD = np.zeros(Ref_alpha.shape)
    Np = Ref_alpha.size
    for pp in list_of_patterns:
        homo_patches = LBP == pp
        homo_patches = ndimage.morphology.binary_closing(homo_patches, SE)
        patch_index, num_patches = ndimage.measurements.label(homo_patches)
        for ii in range(1, num_patches):
            idx = np.where(patch_index == ii)
            if idx[0].size > min_num_pixels:
                c_text_ref = CAM.colorfulness_Gao(Ref_alpha[idx], Ref_beta[idx])
                c_text_pro = CAM.colorfulness_Gao(Pro_alpha[idx], Pro_beta[idx])
                if sq:
                    temp = np.power(c_text_ref-c_text_pro,2)
                else:
                    temp = np.abs(c_text_ref - c_text_pro)
                CD[idx] = temp
                if np.isnan(temp):
                    pass
                wp = (1. * idx[0].size) / Np
                cd += wp * temp
                wpt += 1  # wp
    return CD, cd


def conf26_colorfulness_diff_patches(Ref_image, Pro_image, th=0.035, r=2., min_num_pixels = 4, sq = False):
    SE = np.ones((np.int_(2 * r + 1), np.int_(2 * r + 1)))
    Ref_alpha = 1. * Ref_image[:, :, 0] - Ref_image[:, :, 1]
    Ref_beta = 0.5 * (1. * Ref_image[:, :, 0] + Ref_image[:, :, 1]) - Ref_image[:, :, 2]
    Pro_alpha = 1. * Pro_image[:, :, 0] - Pro_image[:, :, 1]
    Pro_beta = 0.5 * (1. * Pro_image[:, :, 0] + Pro_image[:, :, 1]) - Pro_image[:, :, 2]
    LBP, _ = CT.lbp(Ref_image, th=th, r=r)
    list_of_patterns = np.unique(LBP)
    cd = 0.
    wpt = 0.
    CD = np.zeros(Ref_alpha.shape)
    Np = Ref_alpha.size
    for pp in list_of_patterns:
        homo_patches = LBP == pp
        homo_patches = ndimage.morphology.binary_closing(homo_patches, SE)
        patch_index, num_patches = ndimage.measurements.label(homo_patches)
        for ii in range(1, num_patches):
            idx = np.where(patch_index == ii)
            if idx[0].size > min_num_pixels:
                c_text_ref = CAM.colorfulness_Gao(Ref_alpha[idx], Ref_beta[idx])
                c_text_pro = CAM.colorfulness_Gao(Pro_alpha[idx], Pro_beta[idx])
                if sq:
                    temp = np.power(c_text_ref-c_text_pro,2)
                else:
                    temp = np.abs(c_text_ref - c_text_pro)
                CD[idx] = temp
                if np.isnan(temp):
                    pass
                wp = (1. * idx[0].size) / Np
                cd += wp * temp
                wpt += 1  # wp
    return CD, cd


def conf27_colorfulness_diff_patches(Ref_image, Pro_image, th=0.035, r=3., min_num_pixels = 4, sq = False):
    SE = np.ones((np.int_(2 * r + 1), np.int_(2 * r + 1)))
    Ref_alpha = 1. * Ref_image[:, :, 0] - Ref_image[:, :, 1]
    Ref_beta = 0.5 * (1. * Ref_image[:, :, 0] + Ref_image[:, :, 1]) - Ref_image[:, :, 2]
    Pro_alpha = 1. * Pro_image[:, :, 0] - Pro_image[:, :, 1]
    Pro_beta = 0.5 * (1. * Pro_image[:, :, 0] + Pro_image[:, :, 1]) - Pro_image[:, :, 2]
    LBP, _ = CT.lbp(Ref_image, th=th, r=r)
    list_of_patterns = np.unique(LBP)
    cd = 0.
    wpt = 0.
    CD = np.zeros(Ref_alpha.shape)
    Np = Ref_alpha.size
    for pp in list_of_patterns:
        homo_patches = LBP == pp
        homo_patches = ndimage.morphology.binary_closing(homo_patches, SE)
        patch_index, num_patches = ndimage.measurements.label(homo_patches)
        for ii in range(1, num_patches):
            idx = np.where(patch_index == ii)
            if idx[0].size > min_num_pixels:
                c_text_ref = CAM.colorfulness_Gao(Ref_alpha[idx], Ref_beta[idx])
                c_text_pro = CAM.colorfulness_Gao(Pro_alpha[idx], Pro_beta[idx])
                if sq:
                    temp = np.power(c_text_ref-c_text_pro,2)
                else:
                    temp = np.abs(c_text_ref - c_text_pro)
                CD[idx] = temp
                if np.isnan(temp):
                    pass
                wp = (1. * idx[0].size) / Np
                cd += wp * temp
                wpt += 1  # wp
    return CD, cd




def conf28_colorfulness_diff_patches(Ref_image, Pro_image, th=0.04, r=1., min_num_pixels = 4, sq = False):
    SE = np.ones((np.int_(2 * r + 1), np.int_(2 * r + 1)))
    Ref_alpha = 1. * Ref_image[:, :, 0] - Ref_image[:, :, 1]
    Ref_beta = 0.5 * (1. * Ref_image[:, :, 0] + Ref_image[:, :, 1]) - Ref_image[:, :, 2]
    Pro_alpha = 1. * Pro_image[:, :, 0] - Pro_image[:, :, 1]
    Pro_beta = 0.5 * (1. * Pro_image[:, :, 0] + Pro_image[:, :, 1]) - Pro_image[:, :, 2]
    LBP, _ = CT.lbp(Ref_image, th=th, r=r)
    list_of_patterns = np.unique(LBP)
    cd = 0.
    wpt = 0.
    CD = np.zeros(Ref_alpha.shape)
    Np = Ref_alpha.size
    for pp in list_of_patterns:
        homo_patches = LBP == pp
        homo_patches = ndimage.morphology.binary_closing(homo_patches, SE)
        patch_index, num_patches = ndimage.measurements.label(homo_patches)
        for ii in range(1, num_patches):
            idx = np.where(patch_index == ii)
            if idx[0].size > min_num_pixels:
                c_text_ref = CAM.colorfulness_Gao(Ref_alpha[idx], Ref_beta[idx])
                c_text_pro = CAM.colorfulness_Gao(Pro_alpha[idx], Pro_beta[idx])
                if sq:
                    temp = np.power(c_text_ref-c_text_pro,2)
                else:
                    temp = np.abs(c_text_ref - c_text_pro)
                CD[idx] = temp
                if np.isnan(temp):
                    pass
                wp = (1. * idx[0].size) / Np
                cd += wp * temp
                wpt += 1  # wp
    return CD, cd


def conf29_colorfulness_diff_patches(Ref_image, Pro_image, th=0.04, r=1.5, min_num_pixels = 4, sq = False):
    SE = np.ones((np.int_(2 * r + 1), np.int_(2 * r + 1)))
    Ref_alpha = 1. * Ref_image[:, :, 0] - Ref_image[:, :, 1]
    Ref_beta = 0.5 * (1. * Ref_image[:, :, 0] + Ref_image[:, :, 1]) - Ref_image[:, :, 2]
    Pro_alpha = 1. * Pro_image[:, :, 0] - Pro_image[:, :, 1]
    Pro_beta = 0.5 * (1. * Pro_image[:, :, 0] + Pro_image[:, :, 1]) - Pro_image[:, :, 2]
    LBP, _ = CT.lbp(Ref_image, th=th, r=r)
    list_of_patterns = np.unique(LBP)
    cd = 0.
    wpt = 0.
    CD = np.zeros(Ref_alpha.shape)
    Np = Ref_alpha.size
    for pp in list_of_patterns:
        homo_patches = LBP == pp
        homo_patches = ndimage.morphology.binary_closing(homo_patches, SE)
        patch_index, num_patches = ndimage.measurements.label(homo_patches)
        for ii in range(1, num_patches):
            idx = np.where(patch_index == ii)
            if idx[0].size > min_num_pixels:
                c_text_ref = CAM.colorfulness_Gao(Ref_alpha[idx], Ref_beta[idx])
                c_text_pro = CAM.colorfulness_Gao(Pro_alpha[idx], Pro_beta[idx])
                if sq:
                    temp = np.power(c_text_ref-c_text_pro,2)
                else:
                    temp = np.abs(c_text_ref - c_text_pro)
                CD[idx] = temp
                if np.isnan(temp):
                    pass
                wp = (1. * idx[0].size) / Np
                cd += wp * temp
                wpt += 1  # wp
    return CD, cd


def conf30_colorfulness_diff_patches(Ref_image, Pro_image, th=0.04, r=2., min_num_pixels = 4, sq = False):
    SE = np.ones((np.int_(2 * r + 1), np.int_(2 * r + 1)))
    Ref_alpha = 1. * Ref_image[:, :, 0] - Ref_image[:, :, 1]
    Ref_beta = 0.5 * (1. * Ref_image[:, :, 0] + Ref_image[:, :, 1]) - Ref_image[:, :, 2]
    Pro_alpha = 1. * Pro_image[:, :, 0] - Pro_image[:, :, 1]
    Pro_beta = 0.5 * (1. * Pro_image[:, :, 0] + Pro_image[:, :, 1]) - Pro_image[:, :, 2]
    LBP, _ = CT.lbp(Ref_image, th=th, r=r)
    list_of_patterns = np.unique(LBP)
    cd = 0.
    wpt = 0.
    CD = np.zeros(Ref_alpha.shape)
    Np = Ref_alpha.size
    for pp in list_of_patterns:
        homo_patches = LBP == pp
        homo_patches = ndimage.morphology.binary_closing(homo_patches, SE)
        patch_index, num_patches = ndimage.measurements.label(homo_patches)
        for ii in range(1, num_patches):
            idx = np.where(patch_index == ii)
            if idx[0].size > min_num_pixels:
                c_text_ref = CAM.colorfulness_Gao(Ref_alpha[idx], Ref_beta[idx])
                c_text_pro = CAM.colorfulness_Gao(Pro_alpha[idx], Pro_beta[idx])
                if sq:
                    temp = np.power(c_text_ref-c_text_pro,2)
                else:
                    temp = np.abs(c_text_ref - c_text_pro)
                CD[idx] = temp
                if np.isnan(temp):
                    pass
                wp = (1. * idx[0].size) / Np
                cd += wp * temp
                wpt += 1  # wp
    return CD, cd


def conf31_colorfulness_diff_patches(Ref_image, Pro_image, th=0.04, r=3., min_num_pixels = 4, sq = False):
    SE = np.ones((np.int_(2 * r + 1), np.int_(2 * r + 1)))
    Ref_alpha = 1. * Ref_image[:, :, 0] - Ref_image[:, :, 1]
    Ref_beta = 0.5 * (1. * Ref_image[:, :, 0] + Ref_image[:, :, 1]) - Ref_image[:, :, 2]
    Pro_alpha = 1. * Pro_image[:, :, 0] - Pro_image[:, :, 1]
    Pro_beta = 0.5 * (1. * Pro_image[:, :, 0] + Pro_image[:, :, 1]) - Pro_image[:, :, 2]
    LBP, _ = CT.lbp(Ref_image, th=th, r=r)
    list_of_patterns = np.unique(LBP)
    cd = 0.
    wpt = 0.
    CD = np.zeros(Ref_alpha.shape)
    Np = Ref_alpha.size
    for pp in list_of_patterns:
        homo_patches = LBP == pp
        homo_patches = ndimage.morphology.binary_closing(homo_patches, SE)
        patch_index, num_patches = ndimage.measurements.label(homo_patches)
        for ii in range(1, num_patches):
            idx = np.where(patch_index == ii)
            if idx[0].size > min_num_pixels:
                c_text_ref = CAM.colorfulness_Gao(Ref_alpha[idx], Ref_beta[idx])
                c_text_pro = CAM.colorfulness_Gao(Pro_alpha[idx], Pro_beta[idx])
                if sq:
                    temp = np.power(c_text_ref-c_text_pro,2)
                else:
                    temp = np.abs(c_text_ref - c_text_pro)
                CD[idx] = temp
                if np.isnan(temp):
                    pass
                wp = (1. * idx[0].size) / Np
                cd += wp * temp
                wpt += 1  # wp
    return CD, cd



def conf32_colorfulness_diff_patches(Ref_image, Pro_image, th=0.045, r=1., min_num_pixels = 4, sq = False):
    SE = np.ones((np.int_(2 * r + 1), np.int_(2 * r + 1)))
    Ref_alpha = 1. * Ref_image[:, :, 0] - Ref_image[:, :, 1]
    Ref_beta = 0.5 * (1. * Ref_image[:, :, 0] + Ref_image[:, :, 1]) - Ref_image[:, :, 2]
    Pro_alpha = 1. * Pro_image[:, :, 0] - Pro_image[:, :, 1]
    Pro_beta = 0.5 * (1. * Pro_image[:, :, 0] + Pro_image[:, :, 1]) - Pro_image[:, :, 2]
    LBP, _ = CT.lbp(Ref_image, th=th, r=r)
    list_of_patterns = np.unique(LBP)
    cd = 0.
    wpt = 0.
    CD = np.zeros(Ref_alpha.shape)
    Np = Ref_alpha.size
    for pp in list_of_patterns:
        homo_patches = LBP == pp
        homo_patches = ndimage.morphology.binary_closing(homo_patches, SE)
        patch_index, num_patches = ndimage.measurements.label(homo_patches)
        for ii in range(1, num_patches):
            idx = np.where(patch_index == ii)
            if idx[0].size > min_num_pixels:
                c_text_ref = CAM.colorfulness_Gao(Ref_alpha[idx], Ref_beta[idx])
                c_text_pro = CAM.colorfulness_Gao(Pro_alpha[idx], Pro_beta[idx])
                if sq:
                    temp = np.power(c_text_ref-c_text_pro,2)
                else:
                    temp = np.abs(c_text_ref - c_text_pro)
                CD[idx] = temp
                if np.isnan(temp):
                    pass
                wp = (1. * idx[0].size) / Np
                cd += wp * temp
                wpt += 1  # wp
    return CD, cd


def conf33_colorfulness_diff_patches(Ref_image, Pro_image, th=0.045, r=1.5, min_num_pixels = 4, sq = False):
    SE = np.ones((np.int_(2 * r + 1), np.int_(2 * r + 1)))
    Ref_alpha = 1. * Ref_image[:, :, 0] - Ref_image[:, :, 1]
    Ref_beta = 0.5 * (1. * Ref_image[:, :, 0] + Ref_image[:, :, 1]) - Ref_image[:, :, 2]
    Pro_alpha = 1. * Pro_image[:, :, 0] - Pro_image[:, :, 1]
    Pro_beta = 0.5 * (1. * Pro_image[:, :, 0] + Pro_image[:, :, 1]) - Pro_image[:, :, 2]
    LBP, _ = CT.lbp(Ref_image, th=th, r=r)
    list_of_patterns = np.unique(LBP)
    cd = 0.
    wpt = 0.
    CD = np.zeros(Ref_alpha.shape)
    Np = Ref_alpha.size
    for pp in list_of_patterns:
        homo_patches = LBP == pp
        homo_patches = ndimage.morphology.binary_closing(homo_patches, SE)
        patch_index, num_patches = ndimage.measurements.label(homo_patches)
        for ii in range(1, num_patches):
            idx = np.where(patch_index == ii)
            if idx[0].size > min_num_pixels:
                c_text_ref = CAM.colorfulness_Gao(Ref_alpha[idx], Ref_beta[idx])
                c_text_pro = CAM.colorfulness_Gao(Pro_alpha[idx], Pro_beta[idx])
                if sq:
                    temp = np.power(c_text_ref-c_text_pro,2)
                else:
                    temp = np.abs(c_text_ref - c_text_pro)
                CD[idx] = temp
                if np.isnan(temp):
                    pass
                wp = (1. * idx[0].size) / Np
                cd += wp * temp
                wpt += 1  # wp
    return CD, cd


def conf34_colorfulness_diff_patches(Ref_image, Pro_image, th=0.045, r=2., min_num_pixels = 4, sq = False):
    SE = np.ones((np.int_(2 * r + 1), np.int_(2 * r + 1)))
    Ref_alpha = 1. * Ref_image[:, :, 0] - Ref_image[:, :, 1]
    Ref_beta = 0.5 * (1. * Ref_image[:, :, 0] + Ref_image[:, :, 1]) - Ref_image[:, :, 2]
    Pro_alpha = 1. * Pro_image[:, :, 0] - Pro_image[:, :, 1]
    Pro_beta = 0.5 * (1. * Pro_image[:, :, 0] + Pro_image[:, :, 1]) - Pro_image[:, :, 2]
    LBP, _ = CT.lbp(Ref_image, th=th, r=r)
    list_of_patterns = np.unique(LBP)
    cd = 0.
    wpt = 0.
    CD = np.zeros(Ref_alpha.shape)
    Np = Ref_alpha.size
    for pp in list_of_patterns:
        homo_patches = LBP == pp
        homo_patches = ndimage.morphology.binary_closing(homo_patches, SE)
        patch_index, num_patches = ndimage.measurements.label(homo_patches)
        for ii in range(1, num_patches):
            idx = np.where(patch_index == ii)
            if idx[0].size > min_num_pixels:
                c_text_ref = CAM.colorfulness_Gao(Ref_alpha[idx], Ref_beta[idx])
                c_text_pro = CAM.colorfulness_Gao(Pro_alpha[idx], Pro_beta[idx])
                if sq:
                    temp = np.power(c_text_ref-c_text_pro,2)
                else:
                    temp = np.abs(c_text_ref - c_text_pro)
                CD[idx] = temp
                if np.isnan(temp):
                    pass
                wp = (1. * idx[0].size) / Np
                cd += wp * temp
                wpt += 1  # wp
    return CD, cd


def conf35_colorfulness_diff_patches(Ref_image, Pro_image, th=0.045, r=3., min_num_pixels = 4, sq = False):
    SE = np.ones((np.int_(2 * r + 1), np.int_(2 * r + 1)))
    Ref_alpha = 1. * Ref_image[:, :, 0] - Ref_image[:, :, 1]
    Ref_beta = 0.5 * (1. * Ref_image[:, :, 0] + Ref_image[:, :, 1]) - Ref_image[:, :, 2]
    Pro_alpha = 1. * Pro_image[:, :, 0] - Pro_image[:, :, 1]
    Pro_beta = 0.5 * (1. * Pro_image[:, :, 0] + Pro_image[:, :, 1]) - Pro_image[:, :, 2]
    LBP, _ = CT.lbp(Ref_image, th=th, r=r)
    list_of_patterns = np.unique(LBP)
    cd = 0.
    wpt = 0.
    CD = np.zeros(Ref_alpha.shape)
    Np = Ref_alpha.size
    for pp in list_of_patterns:
        homo_patches = LBP == pp
        homo_patches = ndimage.morphology.binary_closing(homo_patches, SE)
        patch_index, num_patches = ndimage.measurements.label(homo_patches)
        for ii in range(1, num_patches):
            idx = np.where(patch_index == ii)
            if idx[0].size > min_num_pixels:
                c_text_ref = CAM.colorfulness_Gao(Ref_alpha[idx], Ref_beta[idx])
                c_text_pro = CAM.colorfulness_Gao(Pro_alpha[idx], Pro_beta[idx])
                if sq:
                    temp = np.power(c_text_ref-c_text_pro,2)
                else:
                    temp = np.abs(c_text_ref - c_text_pro)
                CD[idx] = temp
                if np.isnan(temp):
                    pass
                wp = (1. * idx[0].size) / Np
                cd += wp * temp
                wpt += 1  # wp
    return CD, cd



def conf36_colorfulness_diff_patches(Ref_image, Pro_image, th=0.05, r=1., min_num_pixels = 4, sq = False):
    SE = np.ones((np.int_(2 * r + 1), np.int_(2 * r + 1)))
    Ref_alpha = 1. * Ref_image[:, :, 0] - Ref_image[:, :, 1]
    Ref_beta = 0.5 * (1. * Ref_image[:, :, 0] + Ref_image[:, :, 1]) - Ref_image[:, :, 2]
    Pro_alpha = 1. * Pro_image[:, :, 0] - Pro_image[:, :, 1]
    Pro_beta = 0.5 * (1. * Pro_image[:, :, 0] + Pro_image[:, :, 1]) - Pro_image[:, :, 2]
    LBP, _ = CT.lbp(Ref_image, th=th, r=r)
    list_of_patterns = np.unique(LBP)
    cd = 0.
    wpt = 0.
    CD = np.zeros(Ref_alpha.shape)
    Np = Ref_alpha.size
    for pp in list_of_patterns:
        homo_patches = LBP == pp
        homo_patches = ndimage.morphology.binary_closing(homo_patches, SE)
        patch_index, num_patches = ndimage.measurements.label(homo_patches)
        for ii in range(1, num_patches):
            idx = np.where(patch_index == ii)
            if idx[0].size > min_num_pixels:
                c_text_ref = CAM.colorfulness_Gao(Ref_alpha[idx], Ref_beta[idx])
                c_text_pro = CAM.colorfulness_Gao(Pro_alpha[idx], Pro_beta[idx])
                if sq:
                    temp = np.power(c_text_ref-c_text_pro,2)
                else:
                    temp = np.abs(c_text_ref - c_text_pro)
                CD[idx] = temp
                if np.isnan(temp):
                    pass
                wp = (1. * idx[0].size) / Np
                cd += wp * temp
                wpt += 1  # wp
    return CD, cd


def conf37_colorfulness_diff_patches(Ref_image, Pro_image, th=0.05, r=1.5, min_num_pixels = 4, sq = False):
    SE = np.ones((np.int_(2 * r + 1), np.int_(2 * r + 1)))
    Ref_alpha = 1. * Ref_image[:, :, 0] - Ref_image[:, :, 1]
    Ref_beta = 0.5 * (1. * Ref_image[:, :, 0] + Ref_image[:, :, 1]) - Ref_image[:, :, 2]
    Pro_alpha = 1. * Pro_image[:, :, 0] - Pro_image[:, :, 1]
    Pro_beta = 0.5 * (1. * Pro_image[:, :, 0] + Pro_image[:, :, 1]) - Pro_image[:, :, 2]
    LBP, _ = CT.lbp(Ref_image, th=th, r=r)
    list_of_patterns = np.unique(LBP)
    cd = 0.
    wpt = 0.
    CD = np.zeros(Ref_alpha.shape)
    Np = Ref_alpha.size
    for pp in list_of_patterns:
        homo_patches = LBP == pp
        homo_patches = ndimage.morphology.binary_closing(homo_patches, SE)
        patch_index, num_patches = ndimage.measurements.label(homo_patches)
        for ii in range(1, num_patches):
            idx = np.where(patch_index == ii)
            if idx[0].size > min_num_pixels:
                c_text_ref = CAM.colorfulness_Gao(Ref_alpha[idx], Ref_beta[idx])
                c_text_pro = CAM.colorfulness_Gao(Pro_alpha[idx], Pro_beta[idx])
                if sq:
                    temp = np.power(c_text_ref-c_text_pro,2)
                else:
                    temp = np.abs(c_text_ref - c_text_pro)
                CD[idx] = temp
                if np.isnan(temp):
                    pass
                wp = (1. * idx[0].size) / Np
                cd += wp * temp
                wpt += 1  # wp
    return CD, cd


def conf38_colorfulness_diff_patches(Ref_image, Pro_image, th=0.05, r=2., min_num_pixels = 4, sq = False):
    SE = np.ones((np.int_(2 * r + 1), np.int_(2 * r + 1)))
    Ref_alpha = 1. * Ref_image[:, :, 0] - Ref_image[:, :, 1]
    Ref_beta = 0.5 * (1. * Ref_image[:, :, 0] + Ref_image[:, :, 1]) - Ref_image[:, :, 2]
    Pro_alpha = 1. * Pro_image[:, :, 0] - Pro_image[:, :, 1]
    Pro_beta = 0.5 * (1. * Pro_image[:, :, 0] + Pro_image[:, :, 1]) - Pro_image[:, :, 2]
    LBP, _ = CT.lbp(Ref_image, th=th, r=r)
    list_of_patterns = np.unique(LBP)
    cd = 0.
    wpt = 0.
    CD = np.zeros(Ref_alpha.shape)
    Np = Ref_alpha.size
    for pp in list_of_patterns:
        homo_patches = LBP == pp
        homo_patches = ndimage.morphology.binary_closing(homo_patches, SE)
        patch_index, num_patches = ndimage.measurements.label(homo_patches)
        for ii in range(1, num_patches):
            idx = np.where(patch_index == ii)
            if idx[0].size > min_num_pixels:
                c_text_ref = CAM.colorfulness_Gao(Ref_alpha[idx], Ref_beta[idx])
                c_text_pro = CAM.colorfulness_Gao(Pro_alpha[idx], Pro_beta[idx])
                if sq:
                    temp = np.power(c_text_ref-c_text_pro,2)
                else:
                    temp = np.abs(c_text_ref - c_text_pro)
                CD[idx] = temp
                if np.isnan(temp):
                    pass
                wp = (1. * idx[0].size) / Np
                cd += wp * temp
                wpt += 1  # wp
    return CD, cd


def conf39_colorfulness_diff_patches(Ref_image, Pro_image, th=0.05, r=3., min_num_pixels = 4, sq = False):
    SE = np.ones((np.int_(2 * r + 1), np.int_(2 * r + 1)))
    Ref_alpha = 1. * Ref_image[:, :, 0] - Ref_image[:, :, 1]
    Ref_beta = 0.5 * (1. * Ref_image[:, :, 0] + Ref_image[:, :, 1]) - Ref_image[:, :, 2]
    Pro_alpha = 1. * Pro_image[:, :, 0] - Pro_image[:, :, 1]
    Pro_beta = 0.5 * (1. * Pro_image[:, :, 0] + Pro_image[:, :, 1]) - Pro_image[:, :, 2]
    LBP, _ = CT.lbp(Ref_image, th=th, r=r)
    list_of_patterns = np.unique(LBP)
    cd = 0.
    wpt = 0.
    CD = np.zeros(Ref_alpha.shape)
    Np = Ref_alpha.size
    for pp in list_of_patterns:
        homo_patches = LBP == pp
        homo_patches = ndimage.morphology.binary_closing(homo_patches, SE)
        patch_index, num_patches = ndimage.measurements.label(homo_patches)
        for ii in range(1, num_patches):
            idx = np.where(patch_index == ii)
            if idx[0].size > min_num_pixels:
                c_text_ref = CAM.colorfulness_Gao(Ref_alpha[idx], Ref_beta[idx])
                c_text_pro = CAM.colorfulness_Gao(Pro_alpha[idx], Pro_beta[idx])
                if sq:
                    temp = np.power(c_text_ref-c_text_pro,2)
                else:
                    temp = np.abs(c_text_ref - c_text_pro)
                CD[idx] = temp
                if np.isnan(temp):
                    pass
                wp = (1. * idx[0].size) / Np
                cd += wp * temp
                wpt += 1  # wp
    return CD, cd
