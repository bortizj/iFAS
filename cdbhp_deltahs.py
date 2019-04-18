import numpy as np
from skimage import color
import cython_functions
from scipy import stats
from scipy import ndimage
import my_utilities as MU
import compute_texture as CT
import matplotlib.pyplot as plt
import color_spaces as CAM



def conf00_deltaHS_patches(Ref_image, Pro_image, th=0.005, r=1., min_num_pixels = 9, sq = False):
    SE = np.ones((np.int_(2 * r + 1), np.int_(2 * r + 1)))
    HSI_ref = color.rgb2hsv(Ref_image)
    H_ref = HSI_ref[:, :, 0]
    S_ref = HSI_ref[:, :, 1]
    HSI_pro = color.rgb2hsv(Pro_image)
    H_pro = HSI_pro[:, :, 0]
    S_pro = HSI_pro[:, :, 1]
    LBP, _ = CT.lbp(Ref_image, th=th, r=r)
    list_of_patterns = np.unique(LBP)
    cd = 0.
    wpt = 0.
    CD = np.zeros(H_ref.shape)
    Np = H_ref.size
    for pp in list_of_patterns:
        homo_patches = LBP == pp
        homo_patches = ndimage.morphology.binary_closing(homo_patches, SE)
        patch_index, num_patches = ndimage.measurements.label(homo_patches)
        for ii in range(1, num_patches):
            idx = np.where(patch_index == ii)
            if idx[0].size > min_num_pixels:
                if sq:
                    temp = np.power(CAM.delta_HS(H_ref[idx], H_pro[idx], S_ref[idx], S_pro[idx]),2)
                else:
                    temp = CAM.delta_HS(H_ref[idx], H_pro[idx], S_ref[idx], S_pro[idx])
                CD[idx] = temp
                if np.isnan(temp):
                    pass
                wp = (1. * idx[0].size) / Np
                cd += wp * temp
                wpt += 1  # wp
    return CD, cd



def conf01_deltaHS_patches(Ref_image, Pro_image, th=0.005, r=1.5, min_num_pixels = 9, sq = False):
    SE = np.ones((np.int_(2 * r + 1), np.int_(2 * r + 1)))
    HSI_ref = color.rgb2hsv(Ref_image)
    H_ref = HSI_ref[:, :, 0]
    S_ref = HSI_ref[:, :, 1]
    HSI_pro = color.rgb2hsv(Pro_image)
    H_pro = HSI_pro[:, :, 0]
    S_pro = HSI_pro[:, :, 1]
    LBP, _ = CT.lbp(Ref_image, th=th, r=r)
    list_of_patterns = np.unique(LBP)
    cd = 0.
    wpt = 0.
    CD = np.zeros(H_ref.shape)
    Np = H_ref.size
    for pp in list_of_patterns:
        homo_patches = LBP == pp
        homo_patches = ndimage.morphology.binary_closing(homo_patches, SE)
        patch_index, num_patches = ndimage.measurements.label(homo_patches)
        for ii in range(1, num_patches):
            idx = np.where(patch_index == ii)
            if idx[0].size > min_num_pixels:
                if sq:
                    temp = np.power(CAM.delta_HS(H_ref[idx], H_pro[idx], S_ref[idx], S_pro[idx]),2)
                else:
                    temp = CAM.delta_HS(H_ref[idx], H_pro[idx], S_ref[idx], S_pro[idx])
                CD[idx] = temp
                if np.isnan(temp):
                    pass
                wp = (1. * idx[0].size) / Np
                cd += wp * temp
                wpt += 1  # wp
    return CD, cd



def conf02_deltaHS_patches(Ref_image, Pro_image, th=0.005, r=2, min_num_pixels = 9, sq = False):
    SE = np.ones((np.int_(2 * r + 1), np.int_(2 * r + 1)))
    HSI_ref = color.rgb2hsv(Ref_image)
    H_ref = HSI_ref[:, :, 0]
    S_ref = HSI_ref[:, :, 1]
    HSI_pro = color.rgb2hsv(Pro_image)
    H_pro = HSI_pro[:, :, 0]
    S_pro = HSI_pro[:, :, 1]
    LBP, _ = CT.lbp(Ref_image, th=th, r=r)
    list_of_patterns = np.unique(LBP)
    cd = 0.
    wpt = 0.
    CD = np.zeros(H_ref.shape)
    Np = H_ref.size
    for pp in list_of_patterns:
        homo_patches = LBP == pp
        homo_patches = ndimage.morphology.binary_closing(homo_patches, SE)
        patch_index, num_patches = ndimage.measurements.label(homo_patches)
        for ii in range(1, num_patches):
            idx = np.where(patch_index == ii)
            if idx[0].size > min_num_pixels:
                if sq:
                    temp = np.power(CAM.delta_HS(H_ref[idx], H_pro[idx], S_ref[idx], S_pro[idx]),2)
                else:
                    temp = CAM.delta_HS(H_ref[idx], H_pro[idx], S_ref[idx], S_pro[idx])
                CD[idx] = temp
                if np.isnan(temp):
                    pass
                wp = (1. * idx[0].size) / Np
                cd += wp * temp
                wpt += 1  # wp
    return CD, cd


def conf03_deltaHS_patches(Ref_image, Pro_image, th=0.005, r=3, min_num_pixels = 9, sq = False):
    SE = np.ones((np.int_(2 * r + 1), np.int_(2 * r + 1)))
    HSI_ref = color.rgb2hsv(Ref_image)
    H_ref = HSI_ref[:, :, 0]
    S_ref = HSI_ref[:, :, 1]
    HSI_pro = color.rgb2hsv(Pro_image)
    H_pro = HSI_pro[:, :, 0]
    S_pro = HSI_pro[:, :, 1]
    LBP, _ = CT.lbp(Ref_image, th=th, r=r)
    list_of_patterns = np.unique(LBP)
    cd = 0.
    wpt = 0.
    CD = np.zeros(H_ref.shape)
    Np = H_ref.size
    for pp in list_of_patterns:
        homo_patches = LBP == pp
        homo_patches = ndimage.morphology.binary_closing(homo_patches, SE)
        patch_index, num_patches = ndimage.measurements.label(homo_patches)
        for ii in range(1, num_patches):
            idx = np.where(patch_index == ii)
            if idx[0].size > min_num_pixels:
                if sq:
                    temp = np.power(CAM.delta_HS(H_ref[idx], H_pro[idx], S_ref[idx], S_pro[idx]),2)
                else:
                    temp = CAM.delta_HS(H_ref[idx], H_pro[idx], S_ref[idx], S_pro[idx])
                CD[idx] = temp
                if np.isnan(temp):
                    pass
                wp = (1. * idx[0].size) / Np
                cd += wp * temp
                wpt += 1  # wp
    return CD, cd


def conf04_deltaHS_patches(Ref_image, Pro_image, th=0.01, r=1., min_num_pixels = 9, sq = False):
    SE = np.ones((np.int_(2 * r + 1), np.int_(2 * r + 1)))
    HSI_ref = color.rgb2hsv(Ref_image)
    H_ref = HSI_ref[:, :, 0]
    S_ref = HSI_ref[:, :, 1]
    HSI_pro = color.rgb2hsv(Pro_image)
    H_pro = HSI_pro[:, :, 0]
    S_pro = HSI_pro[:, :, 1]
    LBP, _ = CT.lbp(Ref_image, th=th, r=r)
    list_of_patterns = np.unique(LBP)
    cd = 0.
    wpt = 0.
    CD = np.zeros(H_ref.shape)
    Np = H_ref.size
    for pp in list_of_patterns:
        homo_patches = LBP == pp
        homo_patches = ndimage.morphology.binary_closing(homo_patches, SE)
        patch_index, num_patches = ndimage.measurements.label(homo_patches)
        for ii in range(1, num_patches):
            idx = np.where(patch_index == ii)
            if idx[0].size > min_num_pixels:
                if sq:
                    temp = np.power(CAM.delta_HS(H_ref[idx], H_pro[idx], S_ref[idx], S_pro[idx]),2)
                else:
                    temp = CAM.delta_HS(H_ref[idx], H_pro[idx], S_ref[idx], S_pro[idx])
                CD[idx] = temp
                if np.isnan(temp):
                    pass
                wp = (1. * idx[0].size) / Np
                cd += wp * temp
                wpt += 1  # wp
    return CD, cd



def conf05_deltaHS_patches(Ref_image, Pro_image, th=0.01, r=1.5, min_num_pixels = 9, sq = False):
    SE = np.ones((np.int_(2 * r + 1), np.int_(2 * r + 1)))
    HSI_ref = color.rgb2hsv(Ref_image)
    H_ref = HSI_ref[:, :, 0]
    S_ref = HSI_ref[:, :, 1]
    HSI_pro = color.rgb2hsv(Pro_image)
    H_pro = HSI_pro[:, :, 0]
    S_pro = HSI_pro[:, :, 1]
    LBP, _ = CT.lbp(Ref_image, th=th, r=r)
    list_of_patterns = np.unique(LBP)
    cd = 0.
    wpt = 0.
    CD = np.zeros(H_ref.shape)
    Np = H_ref.size
    for pp in list_of_patterns:
        homo_patches = LBP == pp
        homo_patches = ndimage.morphology.binary_closing(homo_patches, SE)
        patch_index, num_patches = ndimage.measurements.label(homo_patches)
        for ii in range(1, num_patches):
            idx = np.where(patch_index == ii)
            if idx[0].size > min_num_pixels:
                if sq:
                    temp = np.power(CAM.delta_HS(H_ref[idx], H_pro[idx], S_ref[idx], S_pro[idx]),2)
                else:
                    temp = CAM.delta_HS(H_ref[idx], H_pro[idx], S_ref[idx], S_pro[idx])
                CD[idx] = temp
                if np.isnan(temp):
                    pass
                wp = (1. * idx[0].size) / Np
                cd += wp * temp
                wpt += 1  # wp
    return CD, cd



def conf06_deltaHS_patches(Ref_image, Pro_image, th=0.01, r=2, min_num_pixels = 9, sq = False):
    SE = np.ones((np.int_(2 * r + 1), np.int_(2 * r + 1)))
    HSI_ref = color.rgb2hsv(Ref_image)
    H_ref = HSI_ref[:, :, 0]
    S_ref = HSI_ref[:, :, 1]
    HSI_pro = color.rgb2hsv(Pro_image)
    H_pro = HSI_pro[:, :, 0]
    S_pro = HSI_pro[:, :, 1]
    LBP, _ = CT.lbp(Ref_image, th=th, r=r)
    list_of_patterns = np.unique(LBP)
    cd = 0.
    wpt = 0.
    CD = np.zeros(H_ref.shape)
    Np = H_ref.size
    for pp in list_of_patterns:
        homo_patches = LBP == pp
        homo_patches = ndimage.morphology.binary_closing(homo_patches, SE)
        patch_index, num_patches = ndimage.measurements.label(homo_patches)
        for ii in range(1, num_patches):
            idx = np.where(patch_index == ii)
            if idx[0].size > min_num_pixels:
                if sq:
                    temp = np.power(CAM.delta_HS(H_ref[idx], H_pro[idx], S_ref[idx], S_pro[idx]),2)
                else:
                    temp = CAM.delta_HS(H_ref[idx], H_pro[idx], S_ref[idx], S_pro[idx])
                CD[idx] = temp
                if np.isnan(temp):
                    pass
                wp = (1. * idx[0].size) / Np
                cd += wp * temp
                wpt += 1  # wp
    return CD, cd


def conf07_deltaHS_patches(Ref_image, Pro_image, th=0.01, r=3, min_num_pixels = 9, sq = False):
    SE = np.ones((np.int_(2 * r + 1), np.int_(2 * r + 1)))
    HSI_ref = color.rgb2hsv(Ref_image)
    H_ref = HSI_ref[:, :, 0]
    S_ref = HSI_ref[:, :, 1]
    HSI_pro = color.rgb2hsv(Pro_image)
    H_pro = HSI_pro[:, :, 0]
    S_pro = HSI_pro[:, :, 1]
    LBP, _ = CT.lbp(Ref_image, th=th, r=r)
    list_of_patterns = np.unique(LBP)
    cd = 0.
    wpt = 0.
    CD = np.zeros(H_ref.shape)
    Np = H_ref.size
    for pp in list_of_patterns:
        homo_patches = LBP == pp
        homo_patches = ndimage.morphology.binary_closing(homo_patches, SE)
        patch_index, num_patches = ndimage.measurements.label(homo_patches)
        for ii in range(1, num_patches):
            idx = np.where(patch_index == ii)
            if idx[0].size > min_num_pixels:
                if sq:
                    temp = np.power(CAM.delta_HS(H_ref[idx], H_pro[idx], S_ref[idx], S_pro[idx]),2)
                else:
                    temp = CAM.delta_HS(H_ref[idx], H_pro[idx], S_ref[idx], S_pro[idx])
                CD[idx] = temp
                if np.isnan(temp):
                    pass
                wp = (1. * idx[0].size) / Np
                cd += wp * temp
                wpt += 1  # wp
    return CD, cd



def conf08_deltaHS_patches(Ref_image, Pro_image, th=0.015, r=1., min_num_pixels = 9, sq = False):
    SE = np.ones((np.int_(2 * r + 1), np.int_(2 * r + 1)))
    HSI_ref = color.rgb2hsv(Ref_image)
    H_ref = HSI_ref[:, :, 0]
    S_ref = HSI_ref[:, :, 1]
    HSI_pro = color.rgb2hsv(Pro_image)
    H_pro = HSI_pro[:, :, 0]
    S_pro = HSI_pro[:, :, 1]
    LBP, _ = CT.lbp(Ref_image, th=th, r=r)
    list_of_patterns = np.unique(LBP)
    cd = 0.
    wpt = 0.
    CD = np.zeros(H_ref.shape)
    Np = H_ref.size
    for pp in list_of_patterns:
        homo_patches = LBP == pp
        homo_patches = ndimage.morphology.binary_closing(homo_patches, SE)
        patch_index, num_patches = ndimage.measurements.label(homo_patches)
        for ii in range(1, num_patches):
            idx = np.where(patch_index == ii)
            if idx[0].size > min_num_pixels:
                if sq:
                    temp = np.power(CAM.delta_HS(H_ref[idx], H_pro[idx], S_ref[idx], S_pro[idx]),2)
                else:
                    temp = CAM.delta_HS(H_ref[idx], H_pro[idx], S_ref[idx], S_pro[idx])
                CD[idx] = temp
                if np.isnan(temp):
                    pass
                wp = (1. * idx[0].size) / Np
                cd += wp * temp
                wpt += 1  # wp
    return CD, cd



def conf09_deltaHS_patches(Ref_image, Pro_image, th=0.015, r=1.5, min_num_pixels = 9, sq = False):
    SE = np.ones((np.int_(2 * r + 1), np.int_(2 * r + 1)))
    HSI_ref = color.rgb2hsv(Ref_image)
    H_ref = HSI_ref[:, :, 0]
    S_ref = HSI_ref[:, :, 1]
    HSI_pro = color.rgb2hsv(Pro_image)
    H_pro = HSI_pro[:, :, 0]
    S_pro = HSI_pro[:, :, 1]
    LBP, _ = CT.lbp(Ref_image, th=th, r=r)
    list_of_patterns = np.unique(LBP)
    cd = 0.
    wpt = 0.
    CD = np.zeros(H_ref.shape)
    Np = H_ref.size
    for pp in list_of_patterns:
        homo_patches = LBP == pp
        homo_patches = ndimage.morphology.binary_closing(homo_patches, SE)
        patch_index, num_patches = ndimage.measurements.label(homo_patches)
        for ii in range(1, num_patches):
            idx = np.where(patch_index == ii)
            if idx[0].size > min_num_pixels:
                if sq:
                    temp = np.power(CAM.delta_HS(H_ref[idx], H_pro[idx], S_ref[idx], S_pro[idx]),2)
                else:
                    temp = CAM.delta_HS(H_ref[idx], H_pro[idx], S_ref[idx], S_pro[idx])
                CD[idx] = temp
                if np.isnan(temp):
                    pass
                wp = (1. * idx[0].size) / Np
                cd += wp * temp
                wpt += 1  # wp
    return CD, cd



def conf10_deltaHS_patches(Ref_image, Pro_image, th=0.015, r=2, min_num_pixels = 9, sq = False):
    SE = np.ones((np.int_(2 * r + 1), np.int_(2 * r + 1)))
    HSI_ref = color.rgb2hsv(Ref_image)
    H_ref = HSI_ref[:, :, 0]
    S_ref = HSI_ref[:, :, 1]
    HSI_pro = color.rgb2hsv(Pro_image)
    H_pro = HSI_pro[:, :, 0]
    S_pro = HSI_pro[:, :, 1]
    LBP, _ = CT.lbp(Ref_image, th=th, r=r)
    list_of_patterns = np.unique(LBP)
    cd = 0.
    wpt = 0.
    CD = np.zeros(H_ref.shape)
    Np = H_ref.size
    for pp in list_of_patterns:
        homo_patches = LBP == pp
        homo_patches = ndimage.morphology.binary_closing(homo_patches, SE)
        patch_index, num_patches = ndimage.measurements.label(homo_patches)
        for ii in range(1, num_patches):
            idx = np.where(patch_index == ii)
            if idx[0].size > min_num_pixels:
                if sq:
                    temp = np.power(CAM.delta_HS(H_ref[idx], H_pro[idx], S_ref[idx], S_pro[idx]),2)
                else:
                    temp = CAM.delta_HS(H_ref[idx], H_pro[idx], S_ref[idx], S_pro[idx])
                CD[idx] = temp
                if np.isnan(temp):
                    pass
                wp = (1. * idx[0].size) / Np
                cd += wp * temp
                wpt += 1  # wp
    return CD, cd


def conf11_deltaHS_patches(Ref_image, Pro_image, th=0.015, r=3, min_num_pixels = 9, sq = False):
    SE = np.ones((np.int_(2 * r + 1), np.int_(2 * r + 1)))
    HSI_ref = color.rgb2hsv(Ref_image)
    H_ref = HSI_ref[:, :, 0]
    S_ref = HSI_ref[:, :, 1]
    HSI_pro = color.rgb2hsv(Pro_image)
    H_pro = HSI_pro[:, :, 0]
    S_pro = HSI_pro[:, :, 1]
    LBP, _ = CT.lbp(Ref_image, th=th, r=r)
    list_of_patterns = np.unique(LBP)
    cd = 0.
    wpt = 0.
    CD = np.zeros(H_ref.shape)
    Np = H_ref.size
    for pp in list_of_patterns:
        homo_patches = LBP == pp
        homo_patches = ndimage.morphology.binary_closing(homo_patches, SE)
        patch_index, num_patches = ndimage.measurements.label(homo_patches)
        for ii in range(1, num_patches):
            idx = np.where(patch_index == ii)
            if idx[0].size > min_num_pixels:
                if sq:
                    temp = np.power(CAM.delta_HS(H_ref[idx], H_pro[idx], S_ref[idx], S_pro[idx]),2)
                else:
                    temp = CAM.delta_HS(H_ref[idx], H_pro[idx], S_ref[idx], S_pro[idx])
                CD[idx] = temp
                if np.isnan(temp):
                    pass
                wp = (1. * idx[0].size) / Np
                cd += wp * temp
                wpt += 1  # wp
    return CD, cd



def conf12_deltaHS_patches(Ref_image, Pro_image, th=0.02, r=1., min_num_pixels = 9, sq = False):
    SE = np.ones((np.int_(2 * r + 1), np.int_(2 * r + 1)))
    HSI_ref = color.rgb2hsv(Ref_image)
    H_ref = HSI_ref[:, :, 0]
    S_ref = HSI_ref[:, :, 1]
    HSI_pro = color.rgb2hsv(Pro_image)
    H_pro = HSI_pro[:, :, 0]
    S_pro = HSI_pro[:, :, 1]
    LBP, _ = CT.lbp(Ref_image, th=th, r=r)
    list_of_patterns = np.unique(LBP)
    cd = 0.
    wpt = 0.
    CD = np.zeros(H_ref.shape)
    Np = H_ref.size
    for pp in list_of_patterns:
        homo_patches = LBP == pp
        homo_patches = ndimage.morphology.binary_closing(homo_patches, SE)
        patch_index, num_patches = ndimage.measurements.label(homo_patches)
        for ii in range(1, num_patches):
            idx = np.where(patch_index == ii)
            if idx[0].size > min_num_pixels:
                if sq:
                    temp = np.power(CAM.delta_HS(H_ref[idx], H_pro[idx], S_ref[idx], S_pro[idx]),2)
                else:
                    temp = CAM.delta_HS(H_ref[idx], H_pro[idx], S_ref[idx], S_pro[idx])
                CD[idx] = temp
                if np.isnan(temp):
                    pass
                wp = (1. * idx[0].size) / Np
                cd += wp * temp
                wpt += 1  # wp
    return CD, cd



def conf13_deltaHS_patches(Ref_image, Pro_image, th=0.02, r=1.5, min_num_pixels = 9, sq = False):
    SE = np.ones((np.int_(2 * r + 1), np.int_(2 * r + 1)))
    HSI_ref = color.rgb2hsv(Ref_image)
    H_ref = HSI_ref[:, :, 0]
    S_ref = HSI_ref[:, :, 1]
    HSI_pro = color.rgb2hsv(Pro_image)
    H_pro = HSI_pro[:, :, 0]
    S_pro = HSI_pro[:, :, 1]
    LBP, _ = CT.lbp(Ref_image, th=th, r=r)
    list_of_patterns = np.unique(LBP)
    cd = 0.
    wpt = 0.
    CD = np.zeros(H_ref.shape)
    Np = H_ref.size
    for pp in list_of_patterns:
        homo_patches = LBP == pp
        homo_patches = ndimage.morphology.binary_closing(homo_patches, SE)
        patch_index, num_patches = ndimage.measurements.label(homo_patches)
        for ii in range(1, num_patches):
            idx = np.where(patch_index == ii)
            if idx[0].size > min_num_pixels:
                if sq:
                    temp = np.power(CAM.delta_HS(H_ref[idx], H_pro[idx], S_ref[idx], S_pro[idx]),2)
                else:
                    temp = CAM.delta_HS(H_ref[idx], H_pro[idx], S_ref[idx], S_pro[idx])
                CD[idx] = temp
                if np.isnan(temp):
                    pass
                wp = (1. * idx[0].size) / Np
                cd += wp * temp
                wpt += 1  # wp
    return CD, cd



def conf14_deltaHS_patches(Ref_image, Pro_image, th=0.02, r=2, min_num_pixels = 9, sq = False):
    SE = np.ones((np.int_(2 * r + 1), np.int_(2 * r + 1)))
    HSI_ref = color.rgb2hsv(Ref_image)
    H_ref = HSI_ref[:, :, 0]
    S_ref = HSI_ref[:, :, 1]
    HSI_pro = color.rgb2hsv(Pro_image)
    H_pro = HSI_pro[:, :, 0]
    S_pro = HSI_pro[:, :, 1]
    LBP, _ = CT.lbp(Ref_image, th=th, r=r)
    list_of_patterns = np.unique(LBP)
    cd = 0.
    wpt = 0.
    CD = np.zeros(H_ref.shape)
    Np = H_ref.size
    for pp in list_of_patterns:
        homo_patches = LBP == pp
        homo_patches = ndimage.morphology.binary_closing(homo_patches, SE)
        patch_index, num_patches = ndimage.measurements.label(homo_patches)
        for ii in range(1, num_patches):
            idx = np.where(patch_index == ii)
            if idx[0].size > min_num_pixels:
                if sq:
                    temp = np.power(CAM.delta_HS(H_ref[idx], H_pro[idx], S_ref[idx], S_pro[idx]),2)
                else:
                    temp = CAM.delta_HS(H_ref[idx], H_pro[idx], S_ref[idx], S_pro[idx])
                CD[idx] = temp
                if np.isnan(temp):
                    pass
                wp = (1. * idx[0].size) / Np
                cd += wp * temp
                wpt += 1  # wp
    return CD, cd


def conf15_deltaHS_patches(Ref_image, Pro_image, th=0.02, r=3, min_num_pixels = 9, sq = False):
    SE = np.ones((np.int_(2 * r + 1), np.int_(2 * r + 1)))
    HSI_ref = color.rgb2hsv(Ref_image)
    H_ref = HSI_ref[:, :, 0]
    S_ref = HSI_ref[:, :, 1]
    HSI_pro = color.rgb2hsv(Pro_image)
    H_pro = HSI_pro[:, :, 0]
    S_pro = HSI_pro[:, :, 1]
    LBP, _ = CT.lbp(Ref_image, th=th, r=r)
    list_of_patterns = np.unique(LBP)
    cd = 0.
    wpt = 0.
    CD = np.zeros(H_ref.shape)
    Np = H_ref.size
    for pp in list_of_patterns:
        homo_patches = LBP == pp
        homo_patches = ndimage.morphology.binary_closing(homo_patches, SE)
        patch_index, num_patches = ndimage.measurements.label(homo_patches)
        for ii in range(1, num_patches):
            idx = np.where(patch_index == ii)
            if idx[0].size > min_num_pixels:
                if sq:
                    temp = np.power(CAM.delta_HS(H_ref[idx], H_pro[idx], S_ref[idx], S_pro[idx]),2)
                else:
                    temp = CAM.delta_HS(H_ref[idx], H_pro[idx], S_ref[idx], S_pro[idx])
                CD[idx] = temp
                if np.isnan(temp):
                    pass
                wp = (1. * idx[0].size) / Np
                cd += wp * temp
                wpt += 1  # wp
    return CD, cd



def conf16_deltaHS_patches(Ref_image, Pro_image, th=0.025, r=1., min_num_pixels = 9, sq = False):
    SE = np.ones((np.int_(2 * r + 1), np.int_(2 * r + 1)))
    HSI_ref = color.rgb2hsv(Ref_image)
    H_ref = HSI_ref[:, :, 0]
    S_ref = HSI_ref[:, :, 1]
    HSI_pro = color.rgb2hsv(Pro_image)
    H_pro = HSI_pro[:, :, 0]
    S_pro = HSI_pro[:, :, 1]
    LBP, _ = CT.lbp(Ref_image, th=th, r=r)
    list_of_patterns = np.unique(LBP)
    cd = 0.
    wpt = 0.
    CD = np.zeros(H_ref.shape)
    Np = H_ref.size
    for pp in list_of_patterns:
        homo_patches = LBP == pp
        homo_patches = ndimage.morphology.binary_closing(homo_patches, SE)
        patch_index, num_patches = ndimage.measurements.label(homo_patches)
        for ii in range(1, num_patches):
            idx = np.where(patch_index == ii)
            if idx[0].size > min_num_pixels:
                if sq:
                    temp = np.power(CAM.delta_HS(H_ref[idx], H_pro[idx], S_ref[idx], S_pro[idx]),2)
                else:
                    temp = CAM.delta_HS(H_ref[idx], H_pro[idx], S_ref[idx], S_pro[idx])
                CD[idx] = temp
                if np.isnan(temp):
                    pass
                wp = (1. * idx[0].size) / Np
                cd += wp * temp
                wpt += 1  # wp
    return CD, cd



def conf17_deltaHS_patches(Ref_image, Pro_image, th=0.025, r=1.5, min_num_pixels = 9, sq = False):
    SE = np.ones((np.int_(2 * r + 1), np.int_(2 * r + 1)))
    HSI_ref = color.rgb2hsv(Ref_image)
    H_ref = HSI_ref[:, :, 0]
    S_ref = HSI_ref[:, :, 1]
    HSI_pro = color.rgb2hsv(Pro_image)
    H_pro = HSI_pro[:, :, 0]
    S_pro = HSI_pro[:, :, 1]
    LBP, _ = CT.lbp(Ref_image, th=th, r=r)
    list_of_patterns = np.unique(LBP)
    cd = 0.
    wpt = 0.
    CD = np.zeros(H_ref.shape)
    Np = H_ref.size
    for pp in list_of_patterns:
        homo_patches = LBP == pp
        homo_patches = ndimage.morphology.binary_closing(homo_patches, SE)
        patch_index, num_patches = ndimage.measurements.label(homo_patches)
        for ii in range(1, num_patches):
            idx = np.where(patch_index == ii)
            if idx[0].size > min_num_pixels:
                if sq:
                    temp = np.power(CAM.delta_HS(H_ref[idx], H_pro[idx], S_ref[idx], S_pro[idx]),2)
                else:
                    temp = CAM.delta_HS(H_ref[idx], H_pro[idx], S_ref[idx], S_pro[idx])
                CD[idx] = temp
                if np.isnan(temp):
                    pass
                wp = (1. * idx[0].size) / Np
                cd += wp * temp
                wpt += 1  # wp
    return CD, cd



def conf18_deltaHS_patches(Ref_image, Pro_image, th=0.025, r=2, min_num_pixels = 9, sq = False):
    SE = np.ones((np.int_(2 * r + 1), np.int_(2 * r + 1)))
    HSI_ref = color.rgb2hsv(Ref_image)
    H_ref = HSI_ref[:, :, 0]
    S_ref = HSI_ref[:, :, 1]
    HSI_pro = color.rgb2hsv(Pro_image)
    H_pro = HSI_pro[:, :, 0]
    S_pro = HSI_pro[:, :, 1]
    LBP, _ = CT.lbp(Ref_image, th=th, r=r)
    list_of_patterns = np.unique(LBP)
    cd = 0.
    wpt = 0.
    CD = np.zeros(H_ref.shape)
    Np = H_ref.size
    for pp in list_of_patterns:
        homo_patches = LBP == pp
        homo_patches = ndimage.morphology.binary_closing(homo_patches, SE)
        patch_index, num_patches = ndimage.measurements.label(homo_patches)
        for ii in range(1, num_patches):
            idx = np.where(patch_index == ii)
            if idx[0].size > min_num_pixels:
                if sq:
                    temp = np.power(CAM.delta_HS(H_ref[idx], H_pro[idx], S_ref[idx], S_pro[idx]),2)
                else:
                    temp = CAM.delta_HS(H_ref[idx], H_pro[idx], S_ref[idx], S_pro[idx])
                CD[idx] = temp
                if np.isnan(temp):
                    pass
                wp = (1. * idx[0].size) / Np
                cd += wp * temp
                wpt += 1  # wp
    return CD, cd


def conf19_deltaHS_patches(Ref_image, Pro_image, th=0.025, r=3, min_num_pixels = 9, sq = False):
    SE = np.ones((np.int_(2 * r + 1), np.int_(2 * r + 1)))
    HSI_ref = color.rgb2hsv(Ref_image)
    H_ref = HSI_ref[:, :, 0]
    S_ref = HSI_ref[:, :, 1]
    HSI_pro = color.rgb2hsv(Pro_image)
    H_pro = HSI_pro[:, :, 0]
    S_pro = HSI_pro[:, :, 1]
    LBP, _ = CT.lbp(Ref_image, th=th, r=r)
    list_of_patterns = np.unique(LBP)
    cd = 0.
    wpt = 0.
    CD = np.zeros(H_ref.shape)
    Np = H_ref.size
    for pp in list_of_patterns:
        homo_patches = LBP == pp
        homo_patches = ndimage.morphology.binary_closing(homo_patches, SE)
        patch_index, num_patches = ndimage.measurements.label(homo_patches)
        for ii in range(1, num_patches):
            idx = np.where(patch_index == ii)
            if idx[0].size > min_num_pixels:
                if sq:
                    temp = np.power(CAM.delta_HS(H_ref[idx], H_pro[idx], S_ref[idx], S_pro[idx]),2)
                else:
                    temp = CAM.delta_HS(H_ref[idx], H_pro[idx], S_ref[idx], S_pro[idx])
                CD[idx] = temp
                if np.isnan(temp):
                    pass
                wp = (1. * idx[0].size) / Np
                cd += wp * temp
                wpt += 1  # wp
    return CD, cd



def conf20_deltaHS_patches(Ref_image, Pro_image, th=0.03, r=1., min_num_pixels = 9, sq = False):
    SE = np.ones((np.int_(2 * r + 1), np.int_(2 * r + 1)))
    HSI_ref = color.rgb2hsv(Ref_image)
    H_ref = HSI_ref[:, :, 0]
    S_ref = HSI_ref[:, :, 1]
    HSI_pro = color.rgb2hsv(Pro_image)
    H_pro = HSI_pro[:, :, 0]
    S_pro = HSI_pro[:, :, 1]
    LBP, _ = CT.lbp(Ref_image, th=th, r=r)
    list_of_patterns = np.unique(LBP)
    cd = 0.
    wpt = 0.
    CD = np.zeros(H_ref.shape)
    Np = H_ref.size
    for pp in list_of_patterns:
        homo_patches = LBP == pp
        homo_patches = ndimage.morphology.binary_closing(homo_patches, SE)
        patch_index, num_patches = ndimage.measurements.label(homo_patches)
        for ii in range(1, num_patches):
            idx = np.where(patch_index == ii)
            if idx[0].size > min_num_pixels:
                if sq:
                    temp = np.power(CAM.delta_HS(H_ref[idx], H_pro[idx], S_ref[idx], S_pro[idx]),2)
                else:
                    temp = CAM.delta_HS(H_ref[idx], H_pro[idx], S_ref[idx], S_pro[idx])
                CD[idx] = temp
                if np.isnan(temp):
                    pass
                wp = (1. * idx[0].size) / Np
                cd += wp * temp
                wpt += 1  # wp
    return CD, cd



def conf21_deltaHS_patches(Ref_image, Pro_image, th=0.03, r=1.5, min_num_pixels = 9, sq = False):
    SE = np.ones((np.int_(2 * r + 1), np.int_(2 * r + 1)))
    HSI_ref = color.rgb2hsv(Ref_image)
    H_ref = HSI_ref[:, :, 0]
    S_ref = HSI_ref[:, :, 1]
    HSI_pro = color.rgb2hsv(Pro_image)
    H_pro = HSI_pro[:, :, 0]
    S_pro = HSI_pro[:, :, 1]
    LBP, _ = CT.lbp(Ref_image, th=th, r=r)
    list_of_patterns = np.unique(LBP)
    cd = 0.
    wpt = 0.
    CD = np.zeros(H_ref.shape)
    Np = H_ref.size
    for pp in list_of_patterns:
        homo_patches = LBP == pp
        homo_patches = ndimage.morphology.binary_closing(homo_patches, SE)
        patch_index, num_patches = ndimage.measurements.label(homo_patches)
        for ii in range(1, num_patches):
            idx = np.where(patch_index == ii)
            if idx[0].size > min_num_pixels:
                if sq:
                    temp = np.power(CAM.delta_HS(H_ref[idx], H_pro[idx], S_ref[idx], S_pro[idx]),2)
                else:
                    temp = CAM.delta_HS(H_ref[idx], H_pro[idx], S_ref[idx], S_pro[idx])
                CD[idx] = temp
                if np.isnan(temp):
                    pass
                wp = (1. * idx[0].size) / Np
                cd += wp * temp
                wpt += 1  # wp
    return CD, cd



def conf22_deltaHS_patches(Ref_image, Pro_image, th=0.03, r=2, min_num_pixels = 9, sq = False):
    SE = np.ones((np.int_(2 * r + 1), np.int_(2 * r + 1)))
    HSI_ref = color.rgb2hsv(Ref_image)
    H_ref = HSI_ref[:, :, 0]
    S_ref = HSI_ref[:, :, 1]
    HSI_pro = color.rgb2hsv(Pro_image)
    H_pro = HSI_pro[:, :, 0]
    S_pro = HSI_pro[:, :, 1]
    LBP, _ = CT.lbp(Ref_image, th=th, r=r)
    list_of_patterns = np.unique(LBP)
    cd = 0.
    wpt = 0.
    CD = np.zeros(H_ref.shape)
    Np = H_ref.size
    for pp in list_of_patterns:
        homo_patches = LBP == pp
        homo_patches = ndimage.morphology.binary_closing(homo_patches, SE)
        patch_index, num_patches = ndimage.measurements.label(homo_patches)
        for ii in range(1, num_patches):
            idx = np.where(patch_index == ii)
            if idx[0].size > min_num_pixels:
                if sq:
                    temp = np.power(CAM.delta_HS(H_ref[idx], H_pro[idx], S_ref[idx], S_pro[idx]),2)
                else:
                    temp = CAM.delta_HS(H_ref[idx], H_pro[idx], S_ref[idx], S_pro[idx])
                CD[idx] = temp
                if np.isnan(temp):
                    pass
                wp = (1. * idx[0].size) / Np
                cd += wp * temp
                wpt += 1  # wp
    return CD, cd


def conf23_deltaHS_patches(Ref_image, Pro_image, th=0.03, r=3, min_num_pixels = 9, sq = False):
    SE = np.ones((np.int_(2 * r + 1), np.int_(2 * r + 1)))
    HSI_ref = color.rgb2hsv(Ref_image)
    H_ref = HSI_ref[:, :, 0]
    S_ref = HSI_ref[:, :, 1]
    HSI_pro = color.rgb2hsv(Pro_image)
    H_pro = HSI_pro[:, :, 0]
    S_pro = HSI_pro[:, :, 1]
    LBP, _ = CT.lbp(Ref_image, th=th, r=r)
    list_of_patterns = np.unique(LBP)
    cd = 0.
    wpt = 0.
    CD = np.zeros(H_ref.shape)
    Np = H_ref.size
    for pp in list_of_patterns:
        homo_patches = LBP == pp
        homo_patches = ndimage.morphology.binary_closing(homo_patches, SE)
        patch_index, num_patches = ndimage.measurements.label(homo_patches)
        for ii in range(1, num_patches):
            idx = np.where(patch_index == ii)
            if idx[0].size > min_num_pixels:
                if sq:
                    temp = np.power(CAM.delta_HS(H_ref[idx], H_pro[idx], S_ref[idx], S_pro[idx]),2)
                else:
                    temp = CAM.delta_HS(H_ref[idx], H_pro[idx], S_ref[idx], S_pro[idx])
                CD[idx] = temp
                if np.isnan(temp):
                    pass
                wp = (1. * idx[0].size) / Np
                cd += wp * temp
                wpt += 1  # wp
    return CD, cd


def conf24_deltaHS_patches(Ref_image, Pro_image, th=0.035, r=1., min_num_pixels = 9, sq = False):
    SE = np.ones((np.int_(2 * r + 1), np.int_(2 * r + 1)))
    HSI_ref = color.rgb2hsv(Ref_image)
    H_ref = HSI_ref[:, :, 0]
    S_ref = HSI_ref[:, :, 1]
    HSI_pro = color.rgb2hsv(Pro_image)
    H_pro = HSI_pro[:, :, 0]
    S_pro = HSI_pro[:, :, 1]
    LBP, _ = CT.lbp(Ref_image, th=th, r=r)
    list_of_patterns = np.unique(LBP)
    cd = 0.
    wpt = 0.
    CD = np.zeros(H_ref.shape)
    Np = H_ref.size
    for pp in list_of_patterns:
        homo_patches = LBP == pp
        homo_patches = ndimage.morphology.binary_closing(homo_patches, SE)
        patch_index, num_patches = ndimage.measurements.label(homo_patches)
        for ii in range(1, num_patches):
            idx = np.where(patch_index == ii)
            if idx[0].size > min_num_pixels:
                if sq:
                    temp = np.power(CAM.delta_HS(H_ref[idx], H_pro[idx], S_ref[idx], S_pro[idx]),2)
                else:
                    temp = CAM.delta_HS(H_ref[idx], H_pro[idx], S_ref[idx], S_pro[idx])
                CD[idx] = temp
                if np.isnan(temp):
                    pass
                wp = (1. * idx[0].size) / Np
                cd += wp * temp
                wpt += 1  # wp
    return CD, cd



def conf25_deltaHS_patches(Ref_image, Pro_image, th=0.035, r=1.5, min_num_pixels = 9, sq = False):
    SE = np.ones((np.int_(2 * r + 1), np.int_(2 * r + 1)))
    HSI_ref = color.rgb2hsv(Ref_image)
    H_ref = HSI_ref[:, :, 0]
    S_ref = HSI_ref[:, :, 1]
    HSI_pro = color.rgb2hsv(Pro_image)
    H_pro = HSI_pro[:, :, 0]
    S_pro = HSI_pro[:, :, 1]
    LBP, _ = CT.lbp(Ref_image, th=th, r=r)
    list_of_patterns = np.unique(LBP)
    cd = 0.
    wpt = 0.
    CD = np.zeros(H_ref.shape)
    Np = H_ref.size
    for pp in list_of_patterns:
        homo_patches = LBP == pp
        homo_patches = ndimage.morphology.binary_closing(homo_patches, SE)
        patch_index, num_patches = ndimage.measurements.label(homo_patches)
        for ii in range(1, num_patches):
            idx = np.where(patch_index == ii)
            if idx[0].size > min_num_pixels:
                if sq:
                    temp = np.power(CAM.delta_HS(H_ref[idx], H_pro[idx], S_ref[idx], S_pro[idx]),2)
                else:
                    temp = CAM.delta_HS(H_ref[idx], H_pro[idx], S_ref[idx], S_pro[idx])
                CD[idx] = temp
                if np.isnan(temp):
                    pass
                wp = (1. * idx[0].size) / Np
                cd += wp * temp
                wpt += 1  # wp
    return CD, cd



def conf26_deltaHS_patches(Ref_image, Pro_image, th=0.035, r=2, min_num_pixels = 9, sq = False):
    SE = np.ones((np.int_(2 * r + 1), np.int_(2 * r + 1)))
    HSI_ref = color.rgb2hsv(Ref_image)
    H_ref = HSI_ref[:, :, 0]
    S_ref = HSI_ref[:, :, 1]
    HSI_pro = color.rgb2hsv(Pro_image)
    H_pro = HSI_pro[:, :, 0]
    S_pro = HSI_pro[:, :, 1]
    LBP, _ = CT.lbp(Ref_image, th=th, r=r)
    list_of_patterns = np.unique(LBP)
    cd = 0.
    wpt = 0.
    CD = np.zeros(H_ref.shape)
    Np = H_ref.size
    for pp in list_of_patterns:
        homo_patches = LBP == pp
        homo_patches = ndimage.morphology.binary_closing(homo_patches, SE)
        patch_index, num_patches = ndimage.measurements.label(homo_patches)
        for ii in range(1, num_patches):
            idx = np.where(patch_index == ii)
            if idx[0].size > min_num_pixels:
                if sq:
                    temp = np.power(CAM.delta_HS(H_ref[idx], H_pro[idx], S_ref[idx], S_pro[idx]),2)
                else:
                    temp = CAM.delta_HS(H_ref[idx], H_pro[idx], S_ref[idx], S_pro[idx])
                CD[idx] = temp
                if np.isnan(temp):
                    pass
                wp = (1. * idx[0].size) / Np
                cd += wp * temp
                wpt += 1  # wp
    return CD, cd


def conf27_deltaHS_patches(Ref_image, Pro_image, th=0.035, r=3, min_num_pixels = 9, sq = False):
    SE = np.ones((np.int_(2 * r + 1), np.int_(2 * r + 1)))
    HSI_ref = color.rgb2hsv(Ref_image)
    H_ref = HSI_ref[:, :, 0]
    S_ref = HSI_ref[:, :, 1]
    HSI_pro = color.rgb2hsv(Pro_image)
    H_pro = HSI_pro[:, :, 0]
    S_pro = HSI_pro[:, :, 1]
    LBP, _ = CT.lbp(Ref_image, th=th, r=r)
    list_of_patterns = np.unique(LBP)
    cd = 0.
    wpt = 0.
    CD = np.zeros(H_ref.shape)
    Np = H_ref.size
    for pp in list_of_patterns:
        homo_patches = LBP == pp
        homo_patches = ndimage.morphology.binary_closing(homo_patches, SE)
        patch_index, num_patches = ndimage.measurements.label(homo_patches)
        for ii in range(1, num_patches):
            idx = np.where(patch_index == ii)
            if idx[0].size > min_num_pixels:
                if sq:
                    temp = np.power(CAM.delta_HS(H_ref[idx], H_pro[idx], S_ref[idx], S_pro[idx]),2)
                else:
                    temp = CAM.delta_HS(H_ref[idx], H_pro[idx], S_ref[idx], S_pro[idx])
                CD[idx] = temp
                if np.isnan(temp):
                    pass
                wp = (1. * idx[0].size) / Np
                cd += wp * temp
                wpt += 1  # wp
    return CD, cd



def conf28_deltaHS_patches(Ref_image, Pro_image, th=0.04, r=1., min_num_pixels = 9, sq = False):
    SE = np.ones((np.int_(2 * r + 1), np.int_(2 * r + 1)))
    HSI_ref = color.rgb2hsv(Ref_image)
    H_ref = HSI_ref[:, :, 0]
    S_ref = HSI_ref[:, :, 1]
    HSI_pro = color.rgb2hsv(Pro_image)
    H_pro = HSI_pro[:, :, 0]
    S_pro = HSI_pro[:, :, 1]
    LBP, _ = CT.lbp(Ref_image, th=th, r=r)
    list_of_patterns = np.unique(LBP)
    cd = 0.
    wpt = 0.
    CD = np.zeros(H_ref.shape)
    Np = H_ref.size
    for pp in list_of_patterns:
        homo_patches = LBP == pp
        homo_patches = ndimage.morphology.binary_closing(homo_patches, SE)
        patch_index, num_patches = ndimage.measurements.label(homo_patches)
        for ii in range(1, num_patches):
            idx = np.where(patch_index == ii)
            if idx[0].size > min_num_pixels:
                if sq:
                    temp = np.power(CAM.delta_HS(H_ref[idx], H_pro[idx], S_ref[idx], S_pro[idx]),2)
                else:
                    temp = CAM.delta_HS(H_ref[idx], H_pro[idx], S_ref[idx], S_pro[idx])
                CD[idx] = temp
                if np.isnan(temp):
                    pass
                wp = (1. * idx[0].size) / Np
                cd += wp * temp
                wpt += 1  # wp
    return CD, cd



def conf29_deltaHS_patches(Ref_image, Pro_image, th=0.04, r=1.5, min_num_pixels = 9, sq = False):
    SE = np.ones((np.int_(2 * r + 1), np.int_(2 * r + 1)))
    HSI_ref = color.rgb2hsv(Ref_image)
    H_ref = HSI_ref[:, :, 0]
    S_ref = HSI_ref[:, :, 1]
    HSI_pro = color.rgb2hsv(Pro_image)
    H_pro = HSI_pro[:, :, 0]
    S_pro = HSI_pro[:, :, 1]
    LBP, _ = CT.lbp(Ref_image, th=th, r=r)
    list_of_patterns = np.unique(LBP)
    cd = 0.
    wpt = 0.
    CD = np.zeros(H_ref.shape)
    Np = H_ref.size
    for pp in list_of_patterns:
        homo_patches = LBP == pp
        homo_patches = ndimage.morphology.binary_closing(homo_patches, SE)
        patch_index, num_patches = ndimage.measurements.label(homo_patches)
        for ii in range(1, num_patches):
            idx = np.where(patch_index == ii)
            if idx[0].size > min_num_pixels:
                if sq:
                    temp = np.power(CAM.delta_HS(H_ref[idx], H_pro[idx], S_ref[idx], S_pro[idx]),2)
                else:
                    temp = CAM.delta_HS(H_ref[idx], H_pro[idx], S_ref[idx], S_pro[idx])
                CD[idx] = temp
                if np.isnan(temp):
                    pass
                wp = (1. * idx[0].size) / Np
                cd += wp * temp
                wpt += 1  # wp
    return CD, cd



def conf30_deltaHS_patches(Ref_image, Pro_image, th=0.04, r=2, min_num_pixels = 9, sq = False):
    SE = np.ones((np.int_(2 * r + 1), np.int_(2 * r + 1)))
    HSI_ref = color.rgb2hsv(Ref_image)
    H_ref = HSI_ref[:, :, 0]
    S_ref = HSI_ref[:, :, 1]
    HSI_pro = color.rgb2hsv(Pro_image)
    H_pro = HSI_pro[:, :, 0]
    S_pro = HSI_pro[:, :, 1]
    LBP, _ = CT.lbp(Ref_image, th=th, r=r)
    list_of_patterns = np.unique(LBP)
    cd = 0.
    wpt = 0.
    CD = np.zeros(H_ref.shape)
    Np = H_ref.size
    for pp in list_of_patterns:
        homo_patches = LBP == pp
        homo_patches = ndimage.morphology.binary_closing(homo_patches, SE)
        patch_index, num_patches = ndimage.measurements.label(homo_patches)
        for ii in range(1, num_patches):
            idx = np.where(patch_index == ii)
            if idx[0].size > min_num_pixels:
                if sq:
                    temp = np.power(CAM.delta_HS(H_ref[idx], H_pro[idx], S_ref[idx], S_pro[idx]),2)
                else:
                    temp = CAM.delta_HS(H_ref[idx], H_pro[idx], S_ref[idx], S_pro[idx])
                CD[idx] = temp
                if np.isnan(temp):
                    pass
                wp = (1. * idx[0].size) / Np
                cd += wp * temp
                wpt += 1  # wp
    return CD, cd


def conf31_deltaHS_patches(Ref_image, Pro_image, th=0.04, r=3, min_num_pixels = 9, sq = False):
    SE = np.ones((np.int_(2 * r + 1), np.int_(2 * r + 1)))
    HSI_ref = color.rgb2hsv(Ref_image)
    H_ref = HSI_ref[:, :, 0]
    S_ref = HSI_ref[:, :, 1]
    HSI_pro = color.rgb2hsv(Pro_image)
    H_pro = HSI_pro[:, :, 0]
    S_pro = HSI_pro[:, :, 1]
    LBP, _ = CT.lbp(Ref_image, th=th, r=r)
    list_of_patterns = np.unique(LBP)
    cd = 0.
    wpt = 0.
    CD = np.zeros(H_ref.shape)
    Np = H_ref.size
    for pp in list_of_patterns:
        homo_patches = LBP == pp
        homo_patches = ndimage.morphology.binary_closing(homo_patches, SE)
        patch_index, num_patches = ndimage.measurements.label(homo_patches)
        for ii in range(1, num_patches):
            idx = np.where(patch_index == ii)
            if idx[0].size > min_num_pixels:
                if sq:
                    temp = np.power(CAM.delta_HS(H_ref[idx], H_pro[idx], S_ref[idx], S_pro[idx]),2)
                else:
                    temp = CAM.delta_HS(H_ref[idx], H_pro[idx], S_ref[idx], S_pro[idx])
                CD[idx] = temp
                if np.isnan(temp):
                    pass
                wp = (1. * idx[0].size) / Np
                cd += wp * temp
                wpt += 1  # wp
    return CD, cd



def conf32_deltaHS_patches(Ref_image, Pro_image, th=0.045, r=1., min_num_pixels = 9, sq = False):
    SE = np.ones((np.int_(2 * r + 1), np.int_(2 * r + 1)))
    HSI_ref = color.rgb2hsv(Ref_image)
    H_ref = HSI_ref[:, :, 0]
    S_ref = HSI_ref[:, :, 1]
    HSI_pro = color.rgb2hsv(Pro_image)
    H_pro = HSI_pro[:, :, 0]
    S_pro = HSI_pro[:, :, 1]
    LBP, _ = CT.lbp(Ref_image, th=th, r=r)
    list_of_patterns = np.unique(LBP)
    cd = 0.
    wpt = 0.
    CD = np.zeros(H_ref.shape)
    Np = H_ref.size
    for pp in list_of_patterns:
        homo_patches = LBP == pp
        homo_patches = ndimage.morphology.binary_closing(homo_patches, SE)
        patch_index, num_patches = ndimage.measurements.label(homo_patches)
        for ii in range(1, num_patches):
            idx = np.where(patch_index == ii)
            if idx[0].size > min_num_pixels:
                if sq:
                    temp = np.power(CAM.delta_HS(H_ref[idx], H_pro[idx], S_ref[idx], S_pro[idx]),2)
                else:
                    temp = CAM.delta_HS(H_ref[idx], H_pro[idx], S_ref[idx], S_pro[idx])
                CD[idx] = temp
                if np.isnan(temp):
                    pass
                wp = (1. * idx[0].size) / Np
                cd += wp * temp
                wpt += 1  # wp
    return CD, cd



def conf33_deltaHS_patches(Ref_image, Pro_image, th=0.045, r=1.5, min_num_pixels = 9, sq = False):
    SE = np.ones((np.int_(2 * r + 1), np.int_(2 * r + 1)))
    HSI_ref = color.rgb2hsv(Ref_image)
    H_ref = HSI_ref[:, :, 0]
    S_ref = HSI_ref[:, :, 1]
    HSI_pro = color.rgb2hsv(Pro_image)
    H_pro = HSI_pro[:, :, 0]
    S_pro = HSI_pro[:, :, 1]
    LBP, _ = CT.lbp(Ref_image, th=th, r=r)
    list_of_patterns = np.unique(LBP)
    cd = 0.
    wpt = 0.
    CD = np.zeros(H_ref.shape)
    Np = H_ref.size
    for pp in list_of_patterns:
        homo_patches = LBP == pp
        homo_patches = ndimage.morphology.binary_closing(homo_patches, SE)
        patch_index, num_patches = ndimage.measurements.label(homo_patches)
        for ii in range(1, num_patches):
            idx = np.where(patch_index == ii)
            if idx[0].size > min_num_pixels:
                if sq:
                    temp = np.power(CAM.delta_HS(H_ref[idx], H_pro[idx], S_ref[idx], S_pro[idx]),2)
                else:
                    temp = CAM.delta_HS(H_ref[idx], H_pro[idx], S_ref[idx], S_pro[idx])
                CD[idx] = temp
                if np.isnan(temp):
                    pass
                wp = (1. * idx[0].size) / Np
                cd += wp * temp
                wpt += 1  # wp
    return CD, cd



def conf34_deltaHS_patches(Ref_image, Pro_image, th=0.045, r=2, min_num_pixels = 9, sq = False):
    SE = np.ones((np.int_(2 * r + 1), np.int_(2 * r + 1)))
    HSI_ref = color.rgb2hsv(Ref_image)
    H_ref = HSI_ref[:, :, 0]
    S_ref = HSI_ref[:, :, 1]
    HSI_pro = color.rgb2hsv(Pro_image)
    H_pro = HSI_pro[:, :, 0]
    S_pro = HSI_pro[:, :, 1]
    LBP, _ = CT.lbp(Ref_image, th=th, r=r)
    list_of_patterns = np.unique(LBP)
    cd = 0.
    wpt = 0.
    CD = np.zeros(H_ref.shape)
    Np = H_ref.size
    for pp in list_of_patterns:
        homo_patches = LBP == pp
        homo_patches = ndimage.morphology.binary_closing(homo_patches, SE)
        patch_index, num_patches = ndimage.measurements.label(homo_patches)
        for ii in range(1, num_patches):
            idx = np.where(patch_index == ii)
            if idx[0].size > min_num_pixels:
                if sq:
                    temp = np.power(CAM.delta_HS(H_ref[idx], H_pro[idx], S_ref[idx], S_pro[idx]),2)
                else:
                    temp = CAM.delta_HS(H_ref[idx], H_pro[idx], S_ref[idx], S_pro[idx])
                CD[idx] = temp
                if np.isnan(temp):
                    pass
                wp = (1. * idx[0].size) / Np
                cd += wp * temp
                wpt += 1  # wp
    return CD, cd


def conf35_deltaHS_patches(Ref_image, Pro_image, th=0.045, r=3, min_num_pixels = 9, sq = False):
    SE = np.ones((np.int_(2 * r + 1), np.int_(2 * r + 1)))
    HSI_ref = color.rgb2hsv(Ref_image)
    H_ref = HSI_ref[:, :, 0]
    S_ref = HSI_ref[:, :, 1]
    HSI_pro = color.rgb2hsv(Pro_image)
    H_pro = HSI_pro[:, :, 0]
    S_pro = HSI_pro[:, :, 1]
    LBP, _ = CT.lbp(Ref_image, th=th, r=r)
    list_of_patterns = np.unique(LBP)
    cd = 0.
    wpt = 0.
    CD = np.zeros(H_ref.shape)
    Np = H_ref.size
    for pp in list_of_patterns:
        homo_patches = LBP == pp
        homo_patches = ndimage.morphology.binary_closing(homo_patches, SE)
        patch_index, num_patches = ndimage.measurements.label(homo_patches)
        for ii in range(1, num_patches):
            idx = np.where(patch_index == ii)
            if idx[0].size > min_num_pixels:
                if sq:
                    temp = np.power(CAM.delta_HS(H_ref[idx], H_pro[idx], S_ref[idx], S_pro[idx]),2)
                else:
                    temp = CAM.delta_HS(H_ref[idx], H_pro[idx], S_ref[idx], S_pro[idx])
                CD[idx] = temp
                if np.isnan(temp):
                    pass
                wp = (1. * idx[0].size) / Np
                cd += wp * temp
                wpt += 1  # wp
    return CD, cd



def conf36_deltaHS_patches(Ref_image, Pro_image, th=0.05, r=1., min_num_pixels = 9, sq = False):
    SE = np.ones((np.int_(2 * r + 1), np.int_(2 * r + 1)))
    HSI_ref = color.rgb2hsv(Ref_image)
    H_ref = HSI_ref[:, :, 0]
    S_ref = HSI_ref[:, :, 1]
    HSI_pro = color.rgb2hsv(Pro_image)
    H_pro = HSI_pro[:, :, 0]
    S_pro = HSI_pro[:, :, 1]
    LBP, _ = CT.lbp(Ref_image, th=th, r=r)
    list_of_patterns = np.unique(LBP)
    cd = 0.
    wpt = 0.
    CD = np.zeros(H_ref.shape)
    Np = H_ref.size
    for pp in list_of_patterns:
        homo_patches = LBP == pp
        homo_patches = ndimage.morphology.binary_closing(homo_patches, SE)
        patch_index, num_patches = ndimage.measurements.label(homo_patches)
        for ii in range(1, num_patches):
            idx = np.where(patch_index == ii)
            if idx[0].size > min_num_pixels:
                if sq:
                    temp = np.power(CAM.delta_HS(H_ref[idx], H_pro[idx], S_ref[idx], S_pro[idx]),2)
                else:
                    temp = CAM.delta_HS(H_ref[idx], H_pro[idx], S_ref[idx], S_pro[idx])
                CD[idx] = temp
                if np.isnan(temp):
                    pass
                wp = (1. * idx[0].size) / Np
                cd += wp * temp
                wpt += 1  # wp
    return CD, cd



def conf37_deltaHS_patches(Ref_image, Pro_image, th=0.05, r=1.5, min_num_pixels = 9, sq = False):
    SE = np.ones((np.int_(2 * r + 1), np.int_(2 * r + 1)))
    HSI_ref = color.rgb2hsv(Ref_image)
    H_ref = HSI_ref[:, :, 0]
    S_ref = HSI_ref[:, :, 1]
    HSI_pro = color.rgb2hsv(Pro_image)
    H_pro = HSI_pro[:, :, 0]
    S_pro = HSI_pro[:, :, 1]
    LBP, _ = CT.lbp(Ref_image, th=th, r=r)
    list_of_patterns = np.unique(LBP)
    cd = 0.
    wpt = 0.
    CD = np.zeros(H_ref.shape)
    Np = H_ref.size
    for pp in list_of_patterns:
        homo_patches = LBP == pp
        homo_patches = ndimage.morphology.binary_closing(homo_patches, SE)
        patch_index, num_patches = ndimage.measurements.label(homo_patches)
        for ii in range(1, num_patches):
            idx = np.where(patch_index == ii)
            if idx[0].size > min_num_pixels:
                if sq:
                    temp = np.power(CAM.delta_HS(H_ref[idx], H_pro[idx], S_ref[idx], S_pro[idx]),2)
                else:
                    temp = CAM.delta_HS(H_ref[idx], H_pro[idx], S_ref[idx], S_pro[idx])
                CD[idx] = temp
                if np.isnan(temp):
                    pass
                wp = (1. * idx[0].size) / Np
                cd += wp * temp
                wpt += 1  # wp
    return CD, cd



def conf38_deltaHS_patches(Ref_image, Pro_image, th=0.05, r=2, min_num_pixels = 9, sq = False):
    SE = np.ones((np.int_(2 * r + 1), np.int_(2 * r + 1)))
    HSI_ref = color.rgb2hsv(Ref_image)
    H_ref = HSI_ref[:, :, 0]
    S_ref = HSI_ref[:, :, 1]
    HSI_pro = color.rgb2hsv(Pro_image)
    H_pro = HSI_pro[:, :, 0]
    S_pro = HSI_pro[:, :, 1]
    LBP, _ = CT.lbp(Ref_image, th=th, r=r)
    list_of_patterns = np.unique(LBP)
    cd = 0.
    wpt = 0.
    CD = np.zeros(H_ref.shape)
    Np = H_ref.size
    for pp in list_of_patterns:
        homo_patches = LBP == pp
        homo_patches = ndimage.morphology.binary_closing(homo_patches, SE)
        patch_index, num_patches = ndimage.measurements.label(homo_patches)
        for ii in range(1, num_patches):
            idx = np.where(patch_index == ii)
            if idx[0].size > min_num_pixels:
                if sq:
                    temp = np.power(CAM.delta_HS(H_ref[idx], H_pro[idx], S_ref[idx], S_pro[idx]),2)
                else:
                    temp = CAM.delta_HS(H_ref[idx], H_pro[idx], S_ref[idx], S_pro[idx])
                CD[idx] = temp
                if np.isnan(temp):
                    pass
                wp = (1. * idx[0].size) / Np
                cd += wp * temp
                wpt += 1  # wp
    return CD, cd


def conf39_deltaHS_patches(Ref_image, Pro_image, th=0.05, r=3, min_num_pixels = 9, sq = False):
    SE = np.ones((np.int_(2 * r + 1), np.int_(2 * r + 1)))
    HSI_ref = color.rgb2hsv(Ref_image)
    H_ref = HSI_ref[:, :, 0]
    S_ref = HSI_ref[:, :, 1]
    HSI_pro = color.rgb2hsv(Pro_image)
    H_pro = HSI_pro[:, :, 0]
    S_pro = HSI_pro[:, :, 1]
    LBP, _ = CT.lbp(Ref_image, th=th, r=r)
    list_of_patterns = np.unique(LBP)
    cd = 0.
    wpt = 0.
    CD = np.zeros(H_ref.shape)
    Np = H_ref.size
    for pp in list_of_patterns:
        homo_patches = LBP == pp
        homo_patches = ndimage.morphology.binary_closing(homo_patches, SE)
        patch_index, num_patches = ndimage.measurements.label(homo_patches)
        for ii in range(1, num_patches):
            idx = np.where(patch_index == ii)
            if idx[0].size > min_num_pixels:
                if sq:
                    temp = np.power(CAM.delta_HS(H_ref[idx], H_pro[idx], S_ref[idx], S_pro[idx]),2)
                else:
                    temp = CAM.delta_HS(H_ref[idx], H_pro[idx], S_ref[idx], S_pro[idx])
                CD[idx] = temp
                if np.isnan(temp):
                    pass
                wp = (1. * idx[0].size) / Np
                cd += wp * temp
                wpt += 1  # wp
    return CD, cd
