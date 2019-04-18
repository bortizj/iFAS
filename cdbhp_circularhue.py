import numpy as np
from skimage import color
import cython_functions
from scipy import stats
from scipy import ndimage
import my_utilities as MU
import compute_texture as CT
import matplotlib.pyplot as plt
import cd_measures_pack as CDM



def conf00_circularhue_patches(Ref_image, Pro_image, th=0.005, r=1., min_num_pixels = 4, sq = False, mode = 'same'):
    # SE = np.ones((np.int_(2 * r + 1), np.int_(2 * r + 1)))
    LBP, _ = CT.lbp(Ref_image, th=th, r=r)
    SE = np.ones((2, 2))
    list_of_patterns = np.unique(LBP)
    cd = 0.
    wpt = 0.
    Lab_ref = color.rgb2lab(Ref_image)
    Lab_pro = color.rgb2lab(Pro_image)
    H_ref = np.arctan2(Lab_ref[:, :, 2], Lab_ref[:, :, 1])
    H_pro = np.arctan2(Lab_pro[:, :, 2], Lab_pro[:, :, 1])
    C_ref = np.sqrt(np.power(Lab_ref[:, :, 1], 2) + np.power(Lab_ref[:, :, 2], 2))
    C_pro = np.sqrt(np.power(Lab_pro[:, :, 1], 2) + np.power(Lab_pro[:, :, 2], 2))
    DL, _ = MU.SSIM(Lab_ref[:, :, 0], Lab_pro[:, :, 0], mode=mode)
    Kh = (360 * 0.01) ** 2
    Kc = (180 * 0.01) ** 2
    CDw = np.zeros(DL.shape)
    Np = DL.size
    for pp in list_of_patterns:
        homo_patches = LBP == pp
        homo_patches = ndimage.morphology.binary_closing(homo_patches, SE)
        patch_index, num_patches = ndimage.measurements.label(homo_patches)
        for ii in range(1, num_patches):
            idx = np.where(patch_index == ii)
            if idx[0].size > min_num_pixels:
                C_ref_local_mean = np.mean(C_ref[idx])
                C_pro_local_mean = np.mean(C_pro[idx])
                DC_local = (2. * C_ref_local_mean * C_pro_local_mean + Kc) \
                     / (np.power(C_ref_local_mean, 2) + np.power(C_pro_local_mean, 2) + Kc)
                H_ref_mean = np.arctan2(np.mean(np.cos(H_ref[idx])), np.mean(np.sin(H_ref[idx])))
                H_pro_mean = np.arctan2(np.mean(np.cos(H_pro[idx])), np.mean(np.sin(H_pro[idx])))
                DH_local = (2. * H_ref_mean * H_pro_mean + Kh) \
                     / (np.power(H_ref_mean, 2) + np.power(H_pro_mean, 2) + Kh)
                DL_local = np.mean(DL[idx])
                temp = 1. - DH_local * DC_local * DL_local
                if sq:
                    temp = np.power(temp, 2)
                CDw[idx] = temp
                if np.isnan(temp):
                    pass
                wp = (1. * idx[0].size) / Np
                cd += wp * temp
                wpt += 1  # wp
    return CDw, cd


def conf01_circularhue_patches(Ref_image, Pro_image, th=0.005, r=1.5, min_num_pixels = 4, sq = False, mode = 'same'):
    # SE = np.ones((np.int_(2 * r + 1), np.int_(2 * r + 1)))
    LBP, _ = CT.lbp(Ref_image, th=th, r=r)
    SE = np.ones((2, 2))
    list_of_patterns = np.unique(LBP)
    cd = 0.
    wpt = 0.
    Lab_ref = color.rgb2lab(Ref_image)
    Lab_pro = color.rgb2lab(Pro_image)
    H_ref = np.arctan2(Lab_ref[:, :, 2], Lab_ref[:, :, 1])
    H_pro = np.arctan2(Lab_pro[:, :, 2], Lab_pro[:, :, 1])
    C_ref = np.sqrt(np.power(Lab_ref[:, :, 1], 2) + np.power(Lab_ref[:, :, 2], 2))
    C_pro = np.sqrt(np.power(Lab_pro[:, :, 1], 2) + np.power(Lab_pro[:, :, 2], 2))
    DL, _ = MU.SSIM(Lab_ref[:, :, 0], Lab_pro[:, :, 0], mode=mode)
    Kh = (360 * 0.01) ** 2
    Kc = (180 * 0.01) ** 2
    CDw = np.zeros(DL.shape)
    Np = DL.size
    for pp in list_of_patterns:
        homo_patches = LBP == pp
        homo_patches = ndimage.morphology.binary_closing(homo_patches, SE)
        patch_index, num_patches = ndimage.measurements.label(homo_patches)
        for ii in range(1, num_patches):
            idx = np.where(patch_index == ii)
            if idx[0].size > min_num_pixels:
                C_ref_local_mean = np.mean(C_ref[idx])
                C_pro_local_mean = np.mean(C_pro[idx])
                DC_local = (2. * C_ref_local_mean * C_pro_local_mean + Kc) \
                     / (np.power(C_ref_local_mean, 2) + np.power(C_pro_local_mean, 2) + Kc)
                H_ref_mean = np.arctan2(np.mean(np.cos(H_ref[idx])), np.mean(np.sin(H_ref[idx])))
                H_pro_mean = np.arctan2(np.mean(np.cos(H_pro[idx])), np.mean(np.sin(H_pro[idx])))
                DH_local = (2. * H_ref_mean * H_pro_mean + Kh) \
                     / (np.power(H_ref_mean, 2) + np.power(H_pro_mean, 2) + Kh)
                DL_local = np.mean(DL[idx])
                temp = 1. - DH_local * DC_local * DL_local
                if sq:
                    temp = np.power(temp, 2)
                CDw[idx] = temp
                if np.isnan(temp):
                    pass
                wp = (1. * idx[0].size) / Np
                cd += wp * temp
                wpt += 1  # wp
    return CDw, cd



def conf02_circularhue_patches(Ref_image, Pro_image, th=0.005, r=2., min_num_pixels = 4, sq = False, mode = 'same'):
    # SE = np.ones((np.int_(2 * r + 1), np.int_(2 * r + 1)))
    LBP, _ = CT.lbp(Ref_image, th=th, r=r)
    SE = np.ones((2, 2))
    list_of_patterns = np.unique(LBP)
    cd = 0.
    wpt = 0.
    Lab_ref = color.rgb2lab(Ref_image)
    Lab_pro = color.rgb2lab(Pro_image)
    H_ref = np.arctan2(Lab_ref[:, :, 2], Lab_ref[:, :, 1])
    H_pro = np.arctan2(Lab_pro[:, :, 2], Lab_pro[:, :, 1])
    C_ref = np.sqrt(np.power(Lab_ref[:, :, 1], 2) + np.power(Lab_ref[:, :, 2], 2))
    C_pro = np.sqrt(np.power(Lab_pro[:, :, 1], 2) + np.power(Lab_pro[:, :, 2], 2))
    DL, _ = MU.SSIM(Lab_ref[:, :, 0], Lab_pro[:, :, 0], mode=mode)
    Kh = (360 * 0.01) ** 2
    Kc = (180 * 0.01) ** 2
    CDw = np.zeros(DL.shape)
    Np = DL.size
    for pp in list_of_patterns:
        homo_patches = LBP == pp
        homo_patches = ndimage.morphology.binary_closing(homo_patches, SE)
        patch_index, num_patches = ndimage.measurements.label(homo_patches)
        for ii in range(1, num_patches):
            idx = np.where(patch_index == ii)
            if idx[0].size > min_num_pixels:
                C_ref_local_mean = np.mean(C_ref[idx])
                C_pro_local_mean = np.mean(C_pro[idx])
                DC_local = (2. * C_ref_local_mean * C_pro_local_mean + Kc) \
                     / (np.power(C_ref_local_mean, 2) + np.power(C_pro_local_mean, 2) + Kc)
                H_ref_mean = np.arctan2(np.mean(np.cos(H_ref[idx])), np.mean(np.sin(H_ref[idx])))
                H_pro_mean = np.arctan2(np.mean(np.cos(H_pro[idx])), np.mean(np.sin(H_pro[idx])))
                DH_local = (2. * H_ref_mean * H_pro_mean + Kh) \
                     / (np.power(H_ref_mean, 2) + np.power(H_pro_mean, 2) + Kh)
                DL_local = np.mean(DL[idx])
                temp = 1. - DH_local * DC_local * DL_local
                if sq:
                    temp = np.power(temp, 2)
                CDw[idx] = temp
                if np.isnan(temp):
                    pass
                wp = (1. * idx[0].size) / Np
                cd += wp * temp
                wpt += 1  # wp
    return CDw, cd



def conf03_circularhue_patches(Ref_image, Pro_image, th=0.005, r=3., min_num_pixels = 4, sq = False, mode = 'same'):
    # SE = np.ones((np.int_(2 * r + 1), np.int_(2 * r + 1)))
    LBP, _ = CT.lbp(Ref_image, th=th, r=r)
    SE = np.ones((2, 2))
    list_of_patterns = np.unique(LBP)
    cd = 0.
    wpt = 0.
    Lab_ref = color.rgb2lab(Ref_image)
    Lab_pro = color.rgb2lab(Pro_image)
    H_ref = np.arctan2(Lab_ref[:, :, 2], Lab_ref[:, :, 1])
    H_pro = np.arctan2(Lab_pro[:, :, 2], Lab_pro[:, :, 1])
    C_ref = np.sqrt(np.power(Lab_ref[:, :, 1], 2) + np.power(Lab_ref[:, :, 2], 2))
    C_pro = np.sqrt(np.power(Lab_pro[:, :, 1], 2) + np.power(Lab_pro[:, :, 2], 2))
    DL, _ = MU.SSIM(Lab_ref[:, :, 0], Lab_pro[:, :, 0], mode=mode)
    Kh = (360 * 0.01) ** 2
    Kc = (180 * 0.01) ** 2
    CDw = np.zeros(DL.shape)
    Np = DL.size
    for pp in list_of_patterns:
        homo_patches = LBP == pp
        homo_patches = ndimage.morphology.binary_closing(homo_patches, SE)
        patch_index, num_patches = ndimage.measurements.label(homo_patches)
        for ii in range(1, num_patches):
            idx = np.where(patch_index == ii)
            if idx[0].size > min_num_pixels:
                C_ref_local_mean = np.mean(C_ref[idx])
                C_pro_local_mean = np.mean(C_pro[idx])
                DC_local = (2. * C_ref_local_mean * C_pro_local_mean + Kc) \
                     / (np.power(C_ref_local_mean, 2) + np.power(C_pro_local_mean, 2) + Kc)
                H_ref_mean = np.arctan2(np.mean(np.cos(H_ref[idx])), np.mean(np.sin(H_ref[idx])))
                H_pro_mean = np.arctan2(np.mean(np.cos(H_pro[idx])), np.mean(np.sin(H_pro[idx])))
                DH_local = (2. * H_ref_mean * H_pro_mean + Kh) \
                     / (np.power(H_ref_mean, 2) + np.power(H_pro_mean, 2) + Kh)
                DL_local = np.mean(DL[idx])
                temp = 1. - DH_local * DC_local * DL_local
                if sq:
                    temp = np.power(temp, 2)
                CDw[idx] = temp
                if np.isnan(temp):
                    pass
                wp = (1. * idx[0].size) / Np
                cd += wp * temp
                wpt += 1  # wp
    return CDw, cd



def conf04_circularhue_patches(Ref_image, Pro_image, th=0.01, r=1., min_num_pixels = 4, sq = False, mode = 'same'):
    # SE = np.ones((np.int_(2 * r + 1), np.int_(2 * r + 1)))
    LBP, _ = CT.lbp(Ref_image, th=th, r=r)
    SE = np.ones((2, 2))
    list_of_patterns = np.unique(LBP)
    cd = 0.
    wpt = 0.
    Lab_ref = color.rgb2lab(Ref_image)
    Lab_pro = color.rgb2lab(Pro_image)
    H_ref = np.arctan2(Lab_ref[:, :, 2], Lab_ref[:, :, 1])
    H_pro = np.arctan2(Lab_pro[:, :, 2], Lab_pro[:, :, 1])
    C_ref = np.sqrt(np.power(Lab_ref[:, :, 1], 2) + np.power(Lab_ref[:, :, 2], 2))
    C_pro = np.sqrt(np.power(Lab_pro[:, :, 1], 2) + np.power(Lab_pro[:, :, 2], 2))
    DL, _ = MU.SSIM(Lab_ref[:, :, 0], Lab_pro[:, :, 0], mode=mode)
    Kh = (360 * 0.01) ** 2
    Kc = (180 * 0.01) ** 2
    CDw = np.zeros(DL.shape)
    Np = DL.size
    for pp in list_of_patterns:
        homo_patches = LBP == pp
        homo_patches = ndimage.morphology.binary_closing(homo_patches, SE)
        patch_index, num_patches = ndimage.measurements.label(homo_patches)
        for ii in range(1, num_patches):
            idx = np.where(patch_index == ii)
            if idx[0].size > min_num_pixels:
                C_ref_local_mean = np.mean(C_ref[idx])
                C_pro_local_mean = np.mean(C_pro[idx])
                DC_local = (2. * C_ref_local_mean * C_pro_local_mean + Kc) \
                     / (np.power(C_ref_local_mean, 2) + np.power(C_pro_local_mean, 2) + Kc)
                H_ref_mean = np.arctan2(np.mean(np.cos(H_ref[idx])), np.mean(np.sin(H_ref[idx])))
                H_pro_mean = np.arctan2(np.mean(np.cos(H_pro[idx])), np.mean(np.sin(H_pro[idx])))
                DH_local = (2. * H_ref_mean * H_pro_mean + Kh) \
                     / (np.power(H_ref_mean, 2) + np.power(H_pro_mean, 2) + Kh)
                DL_local = np.mean(DL[idx])
                temp = 1. - DH_local * DC_local * DL_local
                if sq:
                    temp = np.power(temp, 2)
                CDw[idx] = temp
                if np.isnan(temp):
                    pass
                wp = (1. * idx[0].size) / Np
                cd += wp * temp
                wpt += 1  # wp
    return CDw, cd


def conf05_circularhue_patches(Ref_image, Pro_image, th=0.01, r=1.5, min_num_pixels = 4, sq = False, mode = 'same'):
    # SE = np.ones((np.int_(2 * r + 1), np.int_(2 * r + 1)))
    LBP, _ = CT.lbp(Ref_image, th=th, r=r)
    SE = np.ones((2, 2))
    list_of_patterns = np.unique(LBP)
    cd = 0.
    wpt = 0.
    Lab_ref = color.rgb2lab(Ref_image)
    Lab_pro = color.rgb2lab(Pro_image)
    H_ref = np.arctan2(Lab_ref[:, :, 2], Lab_ref[:, :, 1])
    H_pro = np.arctan2(Lab_pro[:, :, 2], Lab_pro[:, :, 1])
    C_ref = np.sqrt(np.power(Lab_ref[:, :, 1], 2) + np.power(Lab_ref[:, :, 2], 2))
    C_pro = np.sqrt(np.power(Lab_pro[:, :, 1], 2) + np.power(Lab_pro[:, :, 2], 2))
    DL, _ = MU.SSIM(Lab_ref[:, :, 0], Lab_pro[:, :, 0], mode=mode)
    Kh = (360 * 0.01) ** 2
    Kc = (180 * 0.01) ** 2
    CDw = np.zeros(DL.shape)
    Np = DL.size
    for pp in list_of_patterns:
        homo_patches = LBP == pp
        homo_patches = ndimage.morphology.binary_closing(homo_patches, SE)
        patch_index, num_patches = ndimage.measurements.label(homo_patches)
        for ii in range(1, num_patches):
            idx = np.where(patch_index == ii)
            if idx[0].size > min_num_pixels:
                C_ref_local_mean = np.mean(C_ref[idx])
                C_pro_local_mean = np.mean(C_pro[idx])
                DC_local = (2. * C_ref_local_mean * C_pro_local_mean + Kc) \
                     / (np.power(C_ref_local_mean, 2) + np.power(C_pro_local_mean, 2) + Kc)
                H_ref_mean = np.arctan2(np.mean(np.cos(H_ref[idx])), np.mean(np.sin(H_ref[idx])))
                H_pro_mean = np.arctan2(np.mean(np.cos(H_pro[idx])), np.mean(np.sin(H_pro[idx])))
                DH_local = (2. * H_ref_mean * H_pro_mean + Kh) \
                     / (np.power(H_ref_mean, 2) + np.power(H_pro_mean, 2) + Kh)
                DL_local = np.mean(DL[idx])
                temp = 1. - DH_local * DC_local * DL_local
                if sq:
                    temp = np.power(temp, 2)
                CDw[idx] = temp
                if np.isnan(temp):
                    pass
                wp = (1. * idx[0].size) / Np
                cd += wp * temp
                wpt += 1  # wp
    return CDw, cd



def conf06_circularhue_patches(Ref_image, Pro_image, th=0.01, r=2., min_num_pixels = 4, sq = False, mode = 'same'):
    # SE = np.ones((np.int_(2 * r + 1), np.int_(2 * r + 1)))
    LBP, _ = CT.lbp(Ref_image, th=th, r=r)
    SE = np.ones((2, 2))
    list_of_patterns = np.unique(LBP)
    cd = 0.
    wpt = 0.
    Lab_ref = color.rgb2lab(Ref_image)
    Lab_pro = color.rgb2lab(Pro_image)
    H_ref = np.arctan2(Lab_ref[:, :, 2], Lab_ref[:, :, 1])
    H_pro = np.arctan2(Lab_pro[:, :, 2], Lab_pro[:, :, 1])
    C_ref = np.sqrt(np.power(Lab_ref[:, :, 1], 2) + np.power(Lab_ref[:, :, 2], 2))
    C_pro = np.sqrt(np.power(Lab_pro[:, :, 1], 2) + np.power(Lab_pro[:, :, 2], 2))
    DL, _ = MU.SSIM(Lab_ref[:, :, 0], Lab_pro[:, :, 0], mode=mode)
    Kh = (360 * 0.01) ** 2
    Kc = (180 * 0.01) ** 2
    CDw = np.zeros(DL.shape)
    Np = DL.size
    for pp in list_of_patterns:
        homo_patches = LBP == pp
        homo_patches = ndimage.morphology.binary_closing(homo_patches, SE)
        patch_index, num_patches = ndimage.measurements.label(homo_patches)
        for ii in range(1, num_patches):
            idx = np.where(patch_index == ii)
            if idx[0].size > min_num_pixels:
                C_ref_local_mean = np.mean(C_ref[idx])
                C_pro_local_mean = np.mean(C_pro[idx])
                DC_local = (2. * C_ref_local_mean * C_pro_local_mean + Kc) \
                     / (np.power(C_ref_local_mean, 2) + np.power(C_pro_local_mean, 2) + Kc)
                H_ref_mean = np.arctan2(np.mean(np.cos(H_ref[idx])), np.mean(np.sin(H_ref[idx])))
                H_pro_mean = np.arctan2(np.mean(np.cos(H_pro[idx])), np.mean(np.sin(H_pro[idx])))
                DH_local = (2. * H_ref_mean * H_pro_mean + Kh) \
                     / (np.power(H_ref_mean, 2) + np.power(H_pro_mean, 2) + Kh)
                DL_local = np.mean(DL[idx])
                temp = 1. - DH_local * DC_local * DL_local
                if sq:
                    temp = np.power(temp, 2)
                CDw[idx] = temp
                if np.isnan(temp):
                    pass
                wp = (1. * idx[0].size) / Np
                cd += wp * temp
                wpt += 1  # wp
    return CDw, cd



def conf07_circularhue_patches(Ref_image, Pro_image, th=0.01, r=3., min_num_pixels = 4, sq = False, mode = 'same'):
    # SE = np.ones((np.int_(2 * r + 1), np.int_(2 * r + 1)))
    LBP, _ = CT.lbp(Ref_image, th=th, r=r)
    SE = np.ones((2, 2))
    list_of_patterns = np.unique(LBP)
    cd = 0.
    wpt = 0.
    Lab_ref = color.rgb2lab(Ref_image)
    Lab_pro = color.rgb2lab(Pro_image)
    H_ref = np.arctan2(Lab_ref[:, :, 2], Lab_ref[:, :, 1])
    H_pro = np.arctan2(Lab_pro[:, :, 2], Lab_pro[:, :, 1])
    C_ref = np.sqrt(np.power(Lab_ref[:, :, 1], 2) + np.power(Lab_ref[:, :, 2], 2))
    C_pro = np.sqrt(np.power(Lab_pro[:, :, 1], 2) + np.power(Lab_pro[:, :, 2], 2))
    DL, _ = MU.SSIM(Lab_ref[:, :, 0], Lab_pro[:, :, 0], mode=mode)
    Kh = (360 * 0.01) ** 2
    Kc = (180 * 0.01) ** 2
    CDw = np.zeros(DL.shape)
    Np = DL.size
    for pp in list_of_patterns:
        homo_patches = LBP == pp
        homo_patches = ndimage.morphology.binary_closing(homo_patches, SE)
        patch_index, num_patches = ndimage.measurements.label(homo_patches)
        for ii in range(1, num_patches):
            idx = np.where(patch_index == ii)
            if idx[0].size > min_num_pixels:
                C_ref_local_mean = np.mean(C_ref[idx])
                C_pro_local_mean = np.mean(C_pro[idx])
                DC_local = (2. * C_ref_local_mean * C_pro_local_mean + Kc) \
                     / (np.power(C_ref_local_mean, 2) + np.power(C_pro_local_mean, 2) + Kc)
                H_ref_mean = np.arctan2(np.mean(np.cos(H_ref[idx])), np.mean(np.sin(H_ref[idx])))
                H_pro_mean = np.arctan2(np.mean(np.cos(H_pro[idx])), np.mean(np.sin(H_pro[idx])))
                DH_local = (2. * H_ref_mean * H_pro_mean + Kh) \
                     / (np.power(H_ref_mean, 2) + np.power(H_pro_mean, 2) + Kh)
                DL_local = np.mean(DL[idx])
                temp = 1. - DH_local * DC_local * DL_local
                if sq:
                    temp = np.power(temp, 2)
                CDw[idx] = temp
                if np.isnan(temp):
                    pass
                wp = (1. * idx[0].size) / Np
                cd += wp * temp
                wpt += 1  # wp
    return CDw, cd


def conf08_circularhue_patches(Ref_image, Pro_image, th=0.015, r=1., min_num_pixels = 4, sq = False, mode = 'same'):
    # SE = np.ones((np.int_(2 * r + 1), np.int_(2 * r + 1)))
    LBP, _ = CT.lbp(Ref_image, th=th, r=r)
    SE = np.ones((2, 2))
    list_of_patterns = np.unique(LBP)
    cd = 0.
    wpt = 0.
    Lab_ref = color.rgb2lab(Ref_image)
    Lab_pro = color.rgb2lab(Pro_image)
    H_ref = np.arctan2(Lab_ref[:, :, 2], Lab_ref[:, :, 1])
    H_pro = np.arctan2(Lab_pro[:, :, 2], Lab_pro[:, :, 1])
    C_ref = np.sqrt(np.power(Lab_ref[:, :, 1], 2) + np.power(Lab_ref[:, :, 2], 2))
    C_pro = np.sqrt(np.power(Lab_pro[:, :, 1], 2) + np.power(Lab_pro[:, :, 2], 2))
    DL, _ = MU.SSIM(Lab_ref[:, :, 0], Lab_pro[:, :, 0], mode=mode)
    Kh = (360 * 0.01) ** 2
    Kc = (180 * 0.01) ** 2
    CDw = np.zeros(DL.shape)
    Np = DL.size
    for pp in list_of_patterns:
        homo_patches = LBP == pp
        homo_patches = ndimage.morphology.binary_closing(homo_patches, SE)
        patch_index, num_patches = ndimage.measurements.label(homo_patches)
        for ii in range(1, num_patches):
            idx = np.where(patch_index == ii)
            if idx[0].size > min_num_pixels:
                C_ref_local_mean = np.mean(C_ref[idx])
                C_pro_local_mean = np.mean(C_pro[idx])
                DC_local = (2. * C_ref_local_mean * C_pro_local_mean + Kc) \
                     / (np.power(C_ref_local_mean, 2) + np.power(C_pro_local_mean, 2) + Kc)
                H_ref_mean = np.arctan2(np.mean(np.cos(H_ref[idx])), np.mean(np.sin(H_ref[idx])))
                H_pro_mean = np.arctan2(np.mean(np.cos(H_pro[idx])), np.mean(np.sin(H_pro[idx])))
                DH_local = (2. * H_ref_mean * H_pro_mean + Kh) \
                     / (np.power(H_ref_mean, 2) + np.power(H_pro_mean, 2) + Kh)
                DL_local = np.mean(DL[idx])
                temp = 1. - DH_local * DC_local * DL_local
                if sq:
                    temp = np.power(temp, 2)
                CDw[idx] = temp
                if np.isnan(temp):
                    pass
                wp = (1. * idx[0].size) / Np
                cd += wp * temp
                wpt += 1  # wp
    return CDw, cd


def conf09_circularhue_patches(Ref_image, Pro_image, th=0.015, r=1.5, min_num_pixels = 4, sq = False, mode = 'same'):
    # SE = np.ones((np.int_(2 * r + 1), np.int_(2 * r + 1)))
    LBP, _ = CT.lbp(Ref_image, th=th, r=r)
    SE = np.ones((2, 2))
    list_of_patterns = np.unique(LBP)
    cd = 0.
    wpt = 0.
    Lab_ref = color.rgb2lab(Ref_image)
    Lab_pro = color.rgb2lab(Pro_image)
    H_ref = np.arctan2(Lab_ref[:, :, 2], Lab_ref[:, :, 1])
    H_pro = np.arctan2(Lab_pro[:, :, 2], Lab_pro[:, :, 1])
    C_ref = np.sqrt(np.power(Lab_ref[:, :, 1], 2) + np.power(Lab_ref[:, :, 2], 2))
    C_pro = np.sqrt(np.power(Lab_pro[:, :, 1], 2) + np.power(Lab_pro[:, :, 2], 2))
    DL, _ = MU.SSIM(Lab_ref[:, :, 0], Lab_pro[:, :, 0], mode=mode)
    Kh = (360 * 0.01) ** 2
    Kc = (180 * 0.01) ** 2
    CDw = np.zeros(DL.shape)
    Np = DL.size
    for pp in list_of_patterns:
        homo_patches = LBP == pp
        homo_patches = ndimage.morphology.binary_closing(homo_patches, SE)
        patch_index, num_patches = ndimage.measurements.label(homo_patches)
        for ii in range(1, num_patches):
            idx = np.where(patch_index == ii)
            if idx[0].size > min_num_pixels:
                C_ref_local_mean = np.mean(C_ref[idx])
                C_pro_local_mean = np.mean(C_pro[idx])
                DC_local = (2. * C_ref_local_mean * C_pro_local_mean + Kc) \
                     / (np.power(C_ref_local_mean, 2) + np.power(C_pro_local_mean, 2) + Kc)
                H_ref_mean = np.arctan2(np.mean(np.cos(H_ref[idx])), np.mean(np.sin(H_ref[idx])))
                H_pro_mean = np.arctan2(np.mean(np.cos(H_pro[idx])), np.mean(np.sin(H_pro[idx])))
                DH_local = (2. * H_ref_mean * H_pro_mean + Kh) \
                     / (np.power(H_ref_mean, 2) + np.power(H_pro_mean, 2) + Kh)
                DL_local = np.mean(DL[idx])
                temp = 1. - DH_local * DC_local * DL_local
                if sq:
                    temp = np.power(temp, 2)
                CDw[idx] = temp
                if np.isnan(temp):
                    pass
                wp = (1. * idx[0].size) / Np
                cd += wp * temp
                wpt += 1  # wp
    return CDw, cd



def conf10_circularhue_patches(Ref_image, Pro_image, th=0.015, r=2., min_num_pixels = 4, sq = False, mode = 'same'):
    # SE = np.ones((np.int_(2 * r + 1), np.int_(2 * r + 1)))
    LBP, _ = CT.lbp(Ref_image, th=th, r=r)
    SE = np.ones((2, 2))
    list_of_patterns = np.unique(LBP)
    cd = 0.
    wpt = 0.
    Lab_ref = color.rgb2lab(Ref_image)
    Lab_pro = color.rgb2lab(Pro_image)
    H_ref = np.arctan2(Lab_ref[:, :, 2], Lab_ref[:, :, 1])
    H_pro = np.arctan2(Lab_pro[:, :, 2], Lab_pro[:, :, 1])
    C_ref = np.sqrt(np.power(Lab_ref[:, :, 1], 2) + np.power(Lab_ref[:, :, 2], 2))
    C_pro = np.sqrt(np.power(Lab_pro[:, :, 1], 2) + np.power(Lab_pro[:, :, 2], 2))
    DL, _ = MU.SSIM(Lab_ref[:, :, 0], Lab_pro[:, :, 0], mode=mode)
    Kh = (360 * 0.01) ** 2
    Kc = (180 * 0.01) ** 2
    CDw = np.zeros(DL.shape)
    Np = DL.size
    for pp in list_of_patterns:
        homo_patches = LBP == pp
        homo_patches = ndimage.morphology.binary_closing(homo_patches, SE)
        patch_index, num_patches = ndimage.measurements.label(homo_patches)
        for ii in range(1, num_patches):
            idx = np.where(patch_index == ii)
            if idx[0].size > min_num_pixels:
                C_ref_local_mean = np.mean(C_ref[idx])
                C_pro_local_mean = np.mean(C_pro[idx])
                DC_local = (2. * C_ref_local_mean * C_pro_local_mean + Kc) \
                     / (np.power(C_ref_local_mean, 2) + np.power(C_pro_local_mean, 2) + Kc)
                H_ref_mean = np.arctan2(np.mean(np.cos(H_ref[idx])), np.mean(np.sin(H_ref[idx])))
                H_pro_mean = np.arctan2(np.mean(np.cos(H_pro[idx])), np.mean(np.sin(H_pro[idx])))
                DH_local = (2. * H_ref_mean * H_pro_mean + Kh) \
                     / (np.power(H_ref_mean, 2) + np.power(H_pro_mean, 2) + Kh)
                DL_local = np.mean(DL[idx])
                temp = 1. - DH_local * DC_local * DL_local
                if sq:
                    temp = np.power(temp, 2)
                CDw[idx] = temp
                if np.isnan(temp):
                    pass
                wp = (1. * idx[0].size) / Np
                cd += wp * temp
                wpt += 1  # wp
    return CDw, cd



def conf11_circularhue_patches(Ref_image, Pro_image, th=0.015, r=3., min_num_pixels = 4, sq = False, mode = 'same'):
    # SE = np.ones((np.int_(2 * r + 1), np.int_(2 * r + 1)))
    LBP, _ = CT.lbp(Ref_image, th=th, r=r)
    SE = np.ones((2, 2))
    list_of_patterns = np.unique(LBP)
    cd = 0.
    wpt = 0.
    Lab_ref = color.rgb2lab(Ref_image)
    Lab_pro = color.rgb2lab(Pro_image)
    H_ref = np.arctan2(Lab_ref[:, :, 2], Lab_ref[:, :, 1])
    H_pro = np.arctan2(Lab_pro[:, :, 2], Lab_pro[:, :, 1])
    C_ref = np.sqrt(np.power(Lab_ref[:, :, 1], 2) + np.power(Lab_ref[:, :, 2], 2))
    C_pro = np.sqrt(np.power(Lab_pro[:, :, 1], 2) + np.power(Lab_pro[:, :, 2], 2))
    DL, _ = MU.SSIM(Lab_ref[:, :, 0], Lab_pro[:, :, 0], mode=mode)
    Kh = (360 * 0.01) ** 2
    Kc = (180 * 0.01) ** 2
    CDw = np.zeros(DL.shape)
    Np = DL.size
    for pp in list_of_patterns:
        homo_patches = LBP == pp
        homo_patches = ndimage.morphology.binary_closing(homo_patches, SE)
        patch_index, num_patches = ndimage.measurements.label(homo_patches)
        for ii in range(1, num_patches):
            idx = np.where(patch_index == ii)
            if idx[0].size > min_num_pixels:
                C_ref_local_mean = np.mean(C_ref[idx])
                C_pro_local_mean = np.mean(C_pro[idx])
                DC_local = (2. * C_ref_local_mean * C_pro_local_mean + Kc) \
                     / (np.power(C_ref_local_mean, 2) + np.power(C_pro_local_mean, 2) + Kc)
                H_ref_mean = np.arctan2(np.mean(np.cos(H_ref[idx])), np.mean(np.sin(H_ref[idx])))
                H_pro_mean = np.arctan2(np.mean(np.cos(H_pro[idx])), np.mean(np.sin(H_pro[idx])))
                DH_local = (2. * H_ref_mean * H_pro_mean + Kh) \
                     / (np.power(H_ref_mean, 2) + np.power(H_pro_mean, 2) + Kh)
                DL_local = np.mean(DL[idx])
                temp = 1. - DH_local * DC_local * DL_local
                if sq:
                    temp = np.power(temp, 2)
                CDw[idx] = temp
                if np.isnan(temp):
                    pass
                wp = (1. * idx[0].size) / Np
                cd += wp * temp
                wpt += 1  # wp
    return CDw, cd


def conf12_circularhue_patches(Ref_image, Pro_image, th=0.02, r=1., min_num_pixels = 4, sq = False, mode = 'same'):
    # SE = np.ones((np.int_(2 * r + 1), np.int_(2 * r + 1)))
    LBP, _ = CT.lbp(Ref_image, th=th, r=r)
    SE = np.ones((2, 2))
    list_of_patterns = np.unique(LBP)
    cd = 0.
    wpt = 0.
    Lab_ref = color.rgb2lab(Ref_image)
    Lab_pro = color.rgb2lab(Pro_image)
    H_ref = np.arctan2(Lab_ref[:, :, 2], Lab_ref[:, :, 1])
    H_pro = np.arctan2(Lab_pro[:, :, 2], Lab_pro[:, :, 1])
    C_ref = np.sqrt(np.power(Lab_ref[:, :, 1], 2) + np.power(Lab_ref[:, :, 2], 2))
    C_pro = np.sqrt(np.power(Lab_pro[:, :, 1], 2) + np.power(Lab_pro[:, :, 2], 2))
    DL, _ = MU.SSIM(Lab_ref[:, :, 0], Lab_pro[:, :, 0], mode=mode)
    Kh = (360 * 0.01) ** 2
    Kc = (180 * 0.01) ** 2
    CDw = np.zeros(DL.shape)
    Np = DL.size
    for pp in list_of_patterns:
        homo_patches = LBP == pp
        homo_patches = ndimage.morphology.binary_closing(homo_patches, SE)
        patch_index, num_patches = ndimage.measurements.label(homo_patches)
        for ii in range(1, num_patches):
            idx = np.where(patch_index == ii)
            if idx[0].size > min_num_pixels:
                C_ref_local_mean = np.mean(C_ref[idx])
                C_pro_local_mean = np.mean(C_pro[idx])
                DC_local = (2. * C_ref_local_mean * C_pro_local_mean + Kc) \
                     / (np.power(C_ref_local_mean, 2) + np.power(C_pro_local_mean, 2) + Kc)
                H_ref_mean = np.arctan2(np.mean(np.cos(H_ref[idx])), np.mean(np.sin(H_ref[idx])))
                H_pro_mean = np.arctan2(np.mean(np.cos(H_pro[idx])), np.mean(np.sin(H_pro[idx])))
                DH_local = (2. * H_ref_mean * H_pro_mean + Kh) \
                     / (np.power(H_ref_mean, 2) + np.power(H_pro_mean, 2) + Kh)
                DL_local = np.mean(DL[idx])
                temp = 1. - DH_local * DC_local * DL_local
                if sq:
                    temp = np.power(temp, 2)
                CDw[idx] = temp
                if np.isnan(temp):
                    pass
                wp = (1. * idx[0].size) / Np
                cd += wp * temp
                wpt += 1  # wp
    return CDw, cd


def conf13_circularhue_patches(Ref_image, Pro_image, th=0.02, r=1.5, min_num_pixels = 4, sq = False, mode = 'same'):
    # SE = np.ones((np.int_(2 * r + 1), np.int_(2 * r + 1)))
    LBP, _ = CT.lbp(Ref_image, th=th, r=r)
    SE = np.ones((2, 2))
    list_of_patterns = np.unique(LBP)
    cd = 0.
    wpt = 0.
    Lab_ref = color.rgb2lab(Ref_image)
    Lab_pro = color.rgb2lab(Pro_image)
    H_ref = np.arctan2(Lab_ref[:, :, 2], Lab_ref[:, :, 1])
    H_pro = np.arctan2(Lab_pro[:, :, 2], Lab_pro[:, :, 1])
    C_ref = np.sqrt(np.power(Lab_ref[:, :, 1], 2) + np.power(Lab_ref[:, :, 2], 2))
    C_pro = np.sqrt(np.power(Lab_pro[:, :, 1], 2) + np.power(Lab_pro[:, :, 2], 2))
    DL, _ = MU.SSIM(Lab_ref[:, :, 0], Lab_pro[:, :, 0], mode=mode)
    Kh = (360 * 0.01) ** 2
    Kc = (180 * 0.01) ** 2
    CDw = np.zeros(DL.shape)
    Np = DL.size
    for pp in list_of_patterns:
        homo_patches = LBP == pp
        homo_patches = ndimage.morphology.binary_closing(homo_patches, SE)
        patch_index, num_patches = ndimage.measurements.label(homo_patches)
        for ii in range(1, num_patches):
            idx = np.where(patch_index == ii)
            if idx[0].size > min_num_pixels:
                C_ref_local_mean = np.mean(C_ref[idx])
                C_pro_local_mean = np.mean(C_pro[idx])
                DC_local = (2. * C_ref_local_mean * C_pro_local_mean + Kc) \
                     / (np.power(C_ref_local_mean, 2) + np.power(C_pro_local_mean, 2) + Kc)
                H_ref_mean = np.arctan2(np.mean(np.cos(H_ref[idx])), np.mean(np.sin(H_ref[idx])))
                H_pro_mean = np.arctan2(np.mean(np.cos(H_pro[idx])), np.mean(np.sin(H_pro[idx])))
                DH_local = (2. * H_ref_mean * H_pro_mean + Kh) \
                     / (np.power(H_ref_mean, 2) + np.power(H_pro_mean, 2) + Kh)
                DL_local = np.mean(DL[idx])
                temp = 1. - DH_local * DC_local * DL_local
                if sq:
                    temp = np.power(temp, 2)
                CDw[idx] = temp
                if np.isnan(temp):
                    pass
                wp = (1. * idx[0].size) / Np
                cd += wp * temp
                wpt += 1  # wp
    return CDw, cd



def conf14_circularhue_patches(Ref_image, Pro_image, th=0.02, r=2., min_num_pixels = 4, sq = False, mode = 'same'):
    # SE = np.ones((np.int_(2 * r + 1), np.int_(2 * r + 1)))
    LBP, _ = CT.lbp(Ref_image, th=th, r=r)
    SE = np.ones((2, 2))
    list_of_patterns = np.unique(LBP)
    cd = 0.
    wpt = 0.
    Lab_ref = color.rgb2lab(Ref_image)
    Lab_pro = color.rgb2lab(Pro_image)
    H_ref = np.arctan2(Lab_ref[:, :, 2], Lab_ref[:, :, 1])
    H_pro = np.arctan2(Lab_pro[:, :, 2], Lab_pro[:, :, 1])
    C_ref = np.sqrt(np.power(Lab_ref[:, :, 1], 2) + np.power(Lab_ref[:, :, 2], 2))
    C_pro = np.sqrt(np.power(Lab_pro[:, :, 1], 2) + np.power(Lab_pro[:, :, 2], 2))
    DL, _ = MU.SSIM(Lab_ref[:, :, 0], Lab_pro[:, :, 0], mode=mode)
    Kh = (360 * 0.01) ** 2
    Kc = (180 * 0.01) ** 2
    CDw = np.zeros(DL.shape)
    Np = DL.size
    for pp in list_of_patterns:
        homo_patches = LBP == pp
        homo_patches = ndimage.morphology.binary_closing(homo_patches, SE)
        patch_index, num_patches = ndimage.measurements.label(homo_patches)
        for ii in range(1, num_patches):
            idx = np.where(patch_index == ii)
            if idx[0].size > min_num_pixels:
                C_ref_local_mean = np.mean(C_ref[idx])
                C_pro_local_mean = np.mean(C_pro[idx])
                DC_local = (2. * C_ref_local_mean * C_pro_local_mean + Kc) \
                     / (np.power(C_ref_local_mean, 2) + np.power(C_pro_local_mean, 2) + Kc)
                H_ref_mean = np.arctan2(np.mean(np.cos(H_ref[idx])), np.mean(np.sin(H_ref[idx])))
                H_pro_mean = np.arctan2(np.mean(np.cos(H_pro[idx])), np.mean(np.sin(H_pro[idx])))
                DH_local = (2. * H_ref_mean * H_pro_mean + Kh) \
                     / (np.power(H_ref_mean, 2) + np.power(H_pro_mean, 2) + Kh)
                DL_local = np.mean(DL[idx])
                temp = 1. - DH_local * DC_local * DL_local
                if sq:
                    temp = np.power(temp, 2)
                CDw[idx] = temp
                if np.isnan(temp):
                    pass
                wp = (1. * idx[0].size) / Np
                cd += wp * temp
                wpt += 1  # wp
    return CDw, cd



def conf15_circularhue_patches(Ref_image, Pro_image, th=0.02, r=3., min_num_pixels = 4, sq = False, mode = 'same'):
    # SE = np.ones((np.int_(2 * r + 1), np.int_(2 * r + 1)))
    LBP, _ = CT.lbp(Ref_image, th=th, r=r)
    SE = np.ones((2, 2))
    list_of_patterns = np.unique(LBP)
    cd = 0.
    wpt = 0.
    Lab_ref = color.rgb2lab(Ref_image)
    Lab_pro = color.rgb2lab(Pro_image)
    H_ref = np.arctan2(Lab_ref[:, :, 2], Lab_ref[:, :, 1])
    H_pro = np.arctan2(Lab_pro[:, :, 2], Lab_pro[:, :, 1])
    C_ref = np.sqrt(np.power(Lab_ref[:, :, 1], 2) + np.power(Lab_ref[:, :, 2], 2))
    C_pro = np.sqrt(np.power(Lab_pro[:, :, 1], 2) + np.power(Lab_pro[:, :, 2], 2))
    DL, _ = MU.SSIM(Lab_ref[:, :, 0], Lab_pro[:, :, 0], mode=mode)
    Kh = (360 * 0.01) ** 2
    Kc = (180 * 0.01) ** 2
    CDw = np.zeros(DL.shape)
    Np = DL.size
    for pp in list_of_patterns:
        homo_patches = LBP == pp
        homo_patches = ndimage.morphology.binary_closing(homo_patches, SE)
        patch_index, num_patches = ndimage.measurements.label(homo_patches)
        for ii in range(1, num_patches):
            idx = np.where(patch_index == ii)
            if idx[0].size > min_num_pixels:
                C_ref_local_mean = np.mean(C_ref[idx])
                C_pro_local_mean = np.mean(C_pro[idx])
                DC_local = (2. * C_ref_local_mean * C_pro_local_mean + Kc) \
                     / (np.power(C_ref_local_mean, 2) + np.power(C_pro_local_mean, 2) + Kc)
                H_ref_mean = np.arctan2(np.mean(np.cos(H_ref[idx])), np.mean(np.sin(H_ref[idx])))
                H_pro_mean = np.arctan2(np.mean(np.cos(H_pro[idx])), np.mean(np.sin(H_pro[idx])))
                DH_local = (2. * H_ref_mean * H_pro_mean + Kh) \
                     / (np.power(H_ref_mean, 2) + np.power(H_pro_mean, 2) + Kh)
                DL_local = np.mean(DL[idx])
                temp = 1. - DH_local * DC_local * DL_local
                if sq:
                    temp = np.power(temp, 2)
                CDw[idx] = temp
                if np.isnan(temp):
                    pass
                wp = (1. * idx[0].size) / Np
                cd += wp * temp
                wpt += 1  # wp
    return CDw, cd



def conf16_circularhue_patches(Ref_image, Pro_image, th=0.025, r=1., min_num_pixels = 4, sq = False, mode = 'same'):
    # SE = np.ones((np.int_(2 * r + 1), np.int_(2 * r + 1)))
    LBP, _ = CT.lbp(Ref_image, th=th, r=r)
    SE = np.ones((2, 2))
    list_of_patterns = np.unique(LBP)
    cd = 0.
    wpt = 0.
    Lab_ref = color.rgb2lab(Ref_image)
    Lab_pro = color.rgb2lab(Pro_image)
    H_ref = np.arctan2(Lab_ref[:, :, 2], Lab_ref[:, :, 1])
    H_pro = np.arctan2(Lab_pro[:, :, 2], Lab_pro[:, :, 1])
    C_ref = np.sqrt(np.power(Lab_ref[:, :, 1], 2) + np.power(Lab_ref[:, :, 2], 2))
    C_pro = np.sqrt(np.power(Lab_pro[:, :, 1], 2) + np.power(Lab_pro[:, :, 2], 2))
    DL, _ = MU.SSIM(Lab_ref[:, :, 0], Lab_pro[:, :, 0], mode=mode)
    Kh = (360 * 0.01) ** 2
    Kc = (180 * 0.01) ** 2
    CDw = np.zeros(DL.shape)
    Np = DL.size
    for pp in list_of_patterns:
        homo_patches = LBP == pp
        homo_patches = ndimage.morphology.binary_closing(homo_patches, SE)
        patch_index, num_patches = ndimage.measurements.label(homo_patches)
        for ii in range(1, num_patches):
            idx = np.where(patch_index == ii)
            if idx[0].size > min_num_pixels:
                C_ref_local_mean = np.mean(C_ref[idx])
                C_pro_local_mean = np.mean(C_pro[idx])
                DC_local = (2. * C_ref_local_mean * C_pro_local_mean + Kc) \
                     / (np.power(C_ref_local_mean, 2) + np.power(C_pro_local_mean, 2) + Kc)
                H_ref_mean = np.arctan2(np.mean(np.cos(H_ref[idx])), np.mean(np.sin(H_ref[idx])))
                H_pro_mean = np.arctan2(np.mean(np.cos(H_pro[idx])), np.mean(np.sin(H_pro[idx])))
                DH_local = (2. * H_ref_mean * H_pro_mean + Kh) \
                     / (np.power(H_ref_mean, 2) + np.power(H_pro_mean, 2) + Kh)
                DL_local = np.mean(DL[idx])
                temp = 1. - DH_local * DC_local * DL_local
                if sq:
                    temp = np.power(temp, 2)
                CDw[idx] = temp
                if np.isnan(temp):
                    pass
                wp = (1. * idx[0].size) / Np
                cd += wp * temp
                wpt += 1  # wp
    return CDw, cd


def conf17_circularhue_patches(Ref_image, Pro_image, th=0.025, r=1.5, min_num_pixels = 4, sq = False, mode = 'same'):
    # SE = np.ones((np.int_(2 * r + 1), np.int_(2 * r + 1)))
    LBP, _ = CT.lbp(Ref_image, th=th, r=r)
    SE = np.ones((2, 2))
    list_of_patterns = np.unique(LBP)
    cd = 0.
    wpt = 0.
    Lab_ref = color.rgb2lab(Ref_image)
    Lab_pro = color.rgb2lab(Pro_image)
    H_ref = np.arctan2(Lab_ref[:, :, 2], Lab_ref[:, :, 1])
    H_pro = np.arctan2(Lab_pro[:, :, 2], Lab_pro[:, :, 1])
    C_ref = np.sqrt(np.power(Lab_ref[:, :, 1], 2) + np.power(Lab_ref[:, :, 2], 2))
    C_pro = np.sqrt(np.power(Lab_pro[:, :, 1], 2) + np.power(Lab_pro[:, :, 2], 2))
    DL, _ = MU.SSIM(Lab_ref[:, :, 0], Lab_pro[:, :, 0], mode=mode)
    Kh = (360 * 0.01) ** 2
    Kc = (180 * 0.01) ** 2
    CDw = np.zeros(DL.shape)
    Np = DL.size
    for pp in list_of_patterns:
        homo_patches = LBP == pp
        homo_patches = ndimage.morphology.binary_closing(homo_patches, SE)
        patch_index, num_patches = ndimage.measurements.label(homo_patches)
        for ii in range(1, num_patches):
            idx = np.where(patch_index == ii)
            if idx[0].size > min_num_pixels:
                C_ref_local_mean = np.mean(C_ref[idx])
                C_pro_local_mean = np.mean(C_pro[idx])
                DC_local = (2. * C_ref_local_mean * C_pro_local_mean + Kc) \
                     / (np.power(C_ref_local_mean, 2) + np.power(C_pro_local_mean, 2) + Kc)
                H_ref_mean = np.arctan2(np.mean(np.cos(H_ref[idx])), np.mean(np.sin(H_ref[idx])))
                H_pro_mean = np.arctan2(np.mean(np.cos(H_pro[idx])), np.mean(np.sin(H_pro[idx])))
                DH_local = (2. * H_ref_mean * H_pro_mean + Kh) \
                     / (np.power(H_ref_mean, 2) + np.power(H_pro_mean, 2) + Kh)
                DL_local = np.mean(DL[idx])
                temp = 1. - DH_local * DC_local * DL_local
                if sq:
                    temp = np.power(temp, 2)
                CDw[idx] = temp
                if np.isnan(temp):
                    pass
                wp = (1. * idx[0].size) / Np
                cd += wp * temp
                wpt += 1  # wp
    return CDw, cd



def conf18_circularhue_patches(Ref_image, Pro_image, th=0.025, r=2., min_num_pixels = 4, sq = False, mode = 'same'):
    # SE = np.ones((np.int_(2 * r + 1), np.int_(2 * r + 1)))
    LBP, _ = CT.lbp(Ref_image, th=th, r=r)
    SE = np.ones((2, 2))
    list_of_patterns = np.unique(LBP)
    cd = 0.
    wpt = 0.
    Lab_ref = color.rgb2lab(Ref_image)
    Lab_pro = color.rgb2lab(Pro_image)
    H_ref = np.arctan2(Lab_ref[:, :, 2], Lab_ref[:, :, 1])
    H_pro = np.arctan2(Lab_pro[:, :, 2], Lab_pro[:, :, 1])
    C_ref = np.sqrt(np.power(Lab_ref[:, :, 1], 2) + np.power(Lab_ref[:, :, 2], 2))
    C_pro = np.sqrt(np.power(Lab_pro[:, :, 1], 2) + np.power(Lab_pro[:, :, 2], 2))
    DL, _ = MU.SSIM(Lab_ref[:, :, 0], Lab_pro[:, :, 0], mode=mode)
    Kh = (360 * 0.01) ** 2
    Kc = (180 * 0.01) ** 2
    CDw = np.zeros(DL.shape)
    Np = DL.size
    for pp in list_of_patterns:
        homo_patches = LBP == pp
        homo_patches = ndimage.morphology.binary_closing(homo_patches, SE)
        patch_index, num_patches = ndimage.measurements.label(homo_patches)
        for ii in range(1, num_patches):
            idx = np.where(patch_index == ii)
            if idx[0].size > min_num_pixels:
                C_ref_local_mean = np.mean(C_ref[idx])
                C_pro_local_mean = np.mean(C_pro[idx])
                DC_local = (2. * C_ref_local_mean * C_pro_local_mean + Kc) \
                     / (np.power(C_ref_local_mean, 2) + np.power(C_pro_local_mean, 2) + Kc)
                H_ref_mean = np.arctan2(np.mean(np.cos(H_ref[idx])), np.mean(np.sin(H_ref[idx])))
                H_pro_mean = np.arctan2(np.mean(np.cos(H_pro[idx])), np.mean(np.sin(H_pro[idx])))
                DH_local = (2. * H_ref_mean * H_pro_mean + Kh) \
                     / (np.power(H_ref_mean, 2) + np.power(H_pro_mean, 2) + Kh)
                DL_local = np.mean(DL[idx])
                temp = 1. - DH_local * DC_local * DL_local
                if sq:
                    temp = np.power(temp, 2)
                CDw[idx] = temp
                if np.isnan(temp):
                    pass
                wp = (1. * idx[0].size) / Np
                cd += wp * temp
                wpt += 1  # wp
    return CDw, cd



def conf19_circularhue_patches(Ref_image, Pro_image, th=0.025, r=3., min_num_pixels = 4, sq = False, mode = 'same'):
    # SE = np.ones((np.int_(2 * r + 1), np.int_(2 * r + 1)))
    LBP, _ = CT.lbp(Ref_image, th=th, r=r)
    SE = np.ones((2, 2))
    list_of_patterns = np.unique(LBP)
    cd = 0.
    wpt = 0.
    Lab_ref = color.rgb2lab(Ref_image)
    Lab_pro = color.rgb2lab(Pro_image)
    H_ref = np.arctan2(Lab_ref[:, :, 2], Lab_ref[:, :, 1])
    H_pro = np.arctan2(Lab_pro[:, :, 2], Lab_pro[:, :, 1])
    C_ref = np.sqrt(np.power(Lab_ref[:, :, 1], 2) + np.power(Lab_ref[:, :, 2], 2))
    C_pro = np.sqrt(np.power(Lab_pro[:, :, 1], 2) + np.power(Lab_pro[:, :, 2], 2))
    DL, _ = MU.SSIM(Lab_ref[:, :, 0], Lab_pro[:, :, 0], mode=mode)
    Kh = (360 * 0.01) ** 2
    Kc = (180 * 0.01) ** 2
    CDw = np.zeros(DL.shape)
    Np = DL.size
    for pp in list_of_patterns:
        homo_patches = LBP == pp
        homo_patches = ndimage.morphology.binary_closing(homo_patches, SE)
        patch_index, num_patches = ndimage.measurements.label(homo_patches)
        for ii in range(1, num_patches):
            idx = np.where(patch_index == ii)
            if idx[0].size > min_num_pixels:
                C_ref_local_mean = np.mean(C_ref[idx])
                C_pro_local_mean = np.mean(C_pro[idx])
                DC_local = (2. * C_ref_local_mean * C_pro_local_mean + Kc) \
                     / (np.power(C_ref_local_mean, 2) + np.power(C_pro_local_mean, 2) + Kc)
                H_ref_mean = np.arctan2(np.mean(np.cos(H_ref[idx])), np.mean(np.sin(H_ref[idx])))
                H_pro_mean = np.arctan2(np.mean(np.cos(H_pro[idx])), np.mean(np.sin(H_pro[idx])))
                DH_local = (2. * H_ref_mean * H_pro_mean + Kh) \
                     / (np.power(H_ref_mean, 2) + np.power(H_pro_mean, 2) + Kh)
                DL_local = np.mean(DL[idx])
                temp = 1. - DH_local * DC_local * DL_local
                if sq:
                    temp = np.power(temp, 2)
                CDw[idx] = temp
                if np.isnan(temp):
                    pass
                wp = (1. * idx[0].size) / Np
                cd += wp * temp
                wpt += 1  # wp
    return CDw, cd


def conf20_circularhue_patches(Ref_image, Pro_image, th=0.03, r=1., min_num_pixels = 4, sq = False, mode = 'same'):
    # SE = np.ones((np.int_(2 * r + 1), np.int_(2 * r + 1)))
    LBP, _ = CT.lbp(Ref_image, th=th, r=r)
    SE = np.ones((2, 2))
    list_of_patterns = np.unique(LBP)
    cd = 0.
    wpt = 0.
    Lab_ref = color.rgb2lab(Ref_image)
    Lab_pro = color.rgb2lab(Pro_image)
    H_ref = np.arctan2(Lab_ref[:, :, 2], Lab_ref[:, :, 1])
    H_pro = np.arctan2(Lab_pro[:, :, 2], Lab_pro[:, :, 1])
    C_ref = np.sqrt(np.power(Lab_ref[:, :, 1], 2) + np.power(Lab_ref[:, :, 2], 2))
    C_pro = np.sqrt(np.power(Lab_pro[:, :, 1], 2) + np.power(Lab_pro[:, :, 2], 2))
    DL, _ = MU.SSIM(Lab_ref[:, :, 0], Lab_pro[:, :, 0], mode=mode)
    Kh = (360 * 0.01) ** 2
    Kc = (180 * 0.01) ** 2
    CDw = np.zeros(DL.shape)
    Np = DL.size
    for pp in list_of_patterns:
        homo_patches = LBP == pp
        homo_patches = ndimage.morphology.binary_closing(homo_patches, SE)
        patch_index, num_patches = ndimage.measurements.label(homo_patches)
        for ii in range(1, num_patches):
            idx = np.where(patch_index == ii)
            if idx[0].size > min_num_pixels:
                C_ref_local_mean = np.mean(C_ref[idx])
                C_pro_local_mean = np.mean(C_pro[idx])
                DC_local = (2. * C_ref_local_mean * C_pro_local_mean + Kc) \
                     / (np.power(C_ref_local_mean, 2) + np.power(C_pro_local_mean, 2) + Kc)
                H_ref_mean = np.arctan2(np.mean(np.cos(H_ref[idx])), np.mean(np.sin(H_ref[idx])))
                H_pro_mean = np.arctan2(np.mean(np.cos(H_pro[idx])), np.mean(np.sin(H_pro[idx])))
                DH_local = (2. * H_ref_mean * H_pro_mean + Kh) \
                     / (np.power(H_ref_mean, 2) + np.power(H_pro_mean, 2) + Kh)
                DL_local = np.mean(DL[idx])
                temp = 1. - DH_local * DC_local * DL_local
                if sq:
                    temp = np.power(temp, 2)
                CDw[idx] = temp
                if np.isnan(temp):
                    pass
                wp = (1. * idx[0].size) / Np
                cd += wp * temp
                wpt += 1  # wp
    return CDw, cd


def conf21_circularhue_patches(Ref_image, Pro_image, th=0.03, r=1.5, min_num_pixels = 4, sq = False, mode = 'same'):
    # SE = np.ones((np.int_(2 * r + 1), np.int_(2 * r + 1)))
    LBP, _ = CT.lbp(Ref_image, th=th, r=r)
    SE = np.ones((2, 2))
    list_of_patterns = np.unique(LBP)
    cd = 0.
    wpt = 0.
    Lab_ref = color.rgb2lab(Ref_image)
    Lab_pro = color.rgb2lab(Pro_image)
    H_ref = np.arctan2(Lab_ref[:, :, 2], Lab_ref[:, :, 1])
    H_pro = np.arctan2(Lab_pro[:, :, 2], Lab_pro[:, :, 1])
    C_ref = np.sqrt(np.power(Lab_ref[:, :, 1], 2) + np.power(Lab_ref[:, :, 2], 2))
    C_pro = np.sqrt(np.power(Lab_pro[:, :, 1], 2) + np.power(Lab_pro[:, :, 2], 2))
    DL, _ = MU.SSIM(Lab_ref[:, :, 0], Lab_pro[:, :, 0], mode=mode)
    Kh = (360 * 0.01) ** 2
    Kc = (180 * 0.01) ** 2
    CDw = np.zeros(DL.shape)
    Np = DL.size
    for pp in list_of_patterns:
        homo_patches = LBP == pp
        homo_patches = ndimage.morphology.binary_closing(homo_patches, SE)
        patch_index, num_patches = ndimage.measurements.label(homo_patches)
        for ii in range(1, num_patches):
            idx = np.where(patch_index == ii)
            if idx[0].size > min_num_pixels:
                C_ref_local_mean = np.mean(C_ref[idx])
                C_pro_local_mean = np.mean(C_pro[idx])
                DC_local = (2. * C_ref_local_mean * C_pro_local_mean + Kc) \
                     / (np.power(C_ref_local_mean, 2) + np.power(C_pro_local_mean, 2) + Kc)
                H_ref_mean = np.arctan2(np.mean(np.cos(H_ref[idx])), np.mean(np.sin(H_ref[idx])))
                H_pro_mean = np.arctan2(np.mean(np.cos(H_pro[idx])), np.mean(np.sin(H_pro[idx])))
                DH_local = (2. * H_ref_mean * H_pro_mean + Kh) \
                     / (np.power(H_ref_mean, 2) + np.power(H_pro_mean, 2) + Kh)
                DL_local = np.mean(DL[idx])
                temp = 1. - DH_local * DC_local * DL_local
                if sq:
                    temp = np.power(temp, 2)
                CDw[idx] = temp
                if np.isnan(temp):
                    pass
                wp = (1. * idx[0].size) / Np
                cd += wp * temp
                wpt += 1  # wp
    return CDw, cd



def conf22_circularhue_patches(Ref_image, Pro_image, th=0.03, r=2., min_num_pixels = 4, sq = False, mode = 'same'):
    # SE = np.ones((np.int_(2 * r + 1), np.int_(2 * r + 1)))
    LBP, _ = CT.lbp(Ref_image, th=th, r=r)
    SE = np.ones((2, 2))
    list_of_patterns = np.unique(LBP)
    cd = 0.
    wpt = 0.
    Lab_ref = color.rgb2lab(Ref_image)
    Lab_pro = color.rgb2lab(Pro_image)
    H_ref = np.arctan2(Lab_ref[:, :, 2], Lab_ref[:, :, 1])
    H_pro = np.arctan2(Lab_pro[:, :, 2], Lab_pro[:, :, 1])
    C_ref = np.sqrt(np.power(Lab_ref[:, :, 1], 2) + np.power(Lab_ref[:, :, 2], 2))
    C_pro = np.sqrt(np.power(Lab_pro[:, :, 1], 2) + np.power(Lab_pro[:, :, 2], 2))
    DL, _ = MU.SSIM(Lab_ref[:, :, 0], Lab_pro[:, :, 0], mode=mode)
    Kh = (360 * 0.01) ** 2
    Kc = (180 * 0.01) ** 2
    CDw = np.zeros(DL.shape)
    Np = DL.size
    for pp in list_of_patterns:
        homo_patches = LBP == pp
        homo_patches = ndimage.morphology.binary_closing(homo_patches, SE)
        patch_index, num_patches = ndimage.measurements.label(homo_patches)
        for ii in range(1, num_patches):
            idx = np.where(patch_index == ii)
            if idx[0].size > min_num_pixels:
                C_ref_local_mean = np.mean(C_ref[idx])
                C_pro_local_mean = np.mean(C_pro[idx])
                DC_local = (2. * C_ref_local_mean * C_pro_local_mean + Kc) \
                     / (np.power(C_ref_local_mean, 2) + np.power(C_pro_local_mean, 2) + Kc)
                H_ref_mean = np.arctan2(np.mean(np.cos(H_ref[idx])), np.mean(np.sin(H_ref[idx])))
                H_pro_mean = np.arctan2(np.mean(np.cos(H_pro[idx])), np.mean(np.sin(H_pro[idx])))
                DH_local = (2. * H_ref_mean * H_pro_mean + Kh) \
                     / (np.power(H_ref_mean, 2) + np.power(H_pro_mean, 2) + Kh)
                DL_local = np.mean(DL[idx])
                temp = 1. - DH_local * DC_local * DL_local
                if sq:
                    temp = np.power(temp, 2)
                CDw[idx] = temp
                if np.isnan(temp):
                    pass
                wp = (1. * idx[0].size) / Np
                cd += wp * temp
                wpt += 1  # wp
    return CDw, cd



def conf23_circularhue_patches(Ref_image, Pro_image, th=0.03, r=3., min_num_pixels = 4, sq = False, mode = 'same'):
    # SE = np.ones((np.int_(2 * r + 1), np.int_(2 * r + 1)))
    LBP, _ = CT.lbp(Ref_image, th=th, r=r)
    SE = np.ones((2, 2))
    list_of_patterns = np.unique(LBP)
    cd = 0.
    wpt = 0.
    Lab_ref = color.rgb2lab(Ref_image)
    Lab_pro = color.rgb2lab(Pro_image)
    H_ref = np.arctan2(Lab_ref[:, :, 2], Lab_ref[:, :, 1])
    H_pro = np.arctan2(Lab_pro[:, :, 2], Lab_pro[:, :, 1])
    C_ref = np.sqrt(np.power(Lab_ref[:, :, 1], 2) + np.power(Lab_ref[:, :, 2], 2))
    C_pro = np.sqrt(np.power(Lab_pro[:, :, 1], 2) + np.power(Lab_pro[:, :, 2], 2))
    DL, _ = MU.SSIM(Lab_ref[:, :, 0], Lab_pro[:, :, 0], mode=mode)
    Kh = (360 * 0.01) ** 2
    Kc = (180 * 0.01) ** 2
    CDw = np.zeros(DL.shape)
    Np = DL.size
    for pp in list_of_patterns:
        homo_patches = LBP == pp
        homo_patches = ndimage.morphology.binary_closing(homo_patches, SE)
        patch_index, num_patches = ndimage.measurements.label(homo_patches)
        for ii in range(1, num_patches):
            idx = np.where(patch_index == ii)
            if idx[0].size > min_num_pixels:
                C_ref_local_mean = np.mean(C_ref[idx])
                C_pro_local_mean = np.mean(C_pro[idx])
                DC_local = (2. * C_ref_local_mean * C_pro_local_mean + Kc) \
                     / (np.power(C_ref_local_mean, 2) + np.power(C_pro_local_mean, 2) + Kc)
                H_ref_mean = np.arctan2(np.mean(np.cos(H_ref[idx])), np.mean(np.sin(H_ref[idx])))
                H_pro_mean = np.arctan2(np.mean(np.cos(H_pro[idx])), np.mean(np.sin(H_pro[idx])))
                DH_local = (2. * H_ref_mean * H_pro_mean + Kh) \
                     / (np.power(H_ref_mean, 2) + np.power(H_pro_mean, 2) + Kh)
                DL_local = np.mean(DL[idx])
                temp = 1. - DH_local * DC_local * DL_local
                if sq:
                    temp = np.power(temp, 2)
                CDw[idx] = temp
                if np.isnan(temp):
                    pass
                wp = (1. * idx[0].size) / Np
                cd += wp * temp
                wpt += 1  # wp
    return CDw, cd



def conf24_circularhue_patches(Ref_image, Pro_image, th=0.035, r=1., min_num_pixels = 4, sq = False, mode = 'same'):
    # SE = np.ones((np.int_(2 * r + 1), np.int_(2 * r + 1)))
    LBP, _ = CT.lbp(Ref_image, th=th, r=r)
    SE = np.ones((2, 2))
    list_of_patterns = np.unique(LBP)
    cd = 0.
    wpt = 0.
    Lab_ref = color.rgb2lab(Ref_image)
    Lab_pro = color.rgb2lab(Pro_image)
    H_ref = np.arctan2(Lab_ref[:, :, 2], Lab_ref[:, :, 1])
    H_pro = np.arctan2(Lab_pro[:, :, 2], Lab_pro[:, :, 1])
    C_ref = np.sqrt(np.power(Lab_ref[:, :, 1], 2) + np.power(Lab_ref[:, :, 2], 2))
    C_pro = np.sqrt(np.power(Lab_pro[:, :, 1], 2) + np.power(Lab_pro[:, :, 2], 2))
    DL, _ = MU.SSIM(Lab_ref[:, :, 0], Lab_pro[:, :, 0], mode=mode)
    Kh = (360 * 0.01) ** 2
    Kc = (180 * 0.01) ** 2
    CDw = np.zeros(DL.shape)
    Np = DL.size
    for pp in list_of_patterns:
        homo_patches = LBP == pp
        homo_patches = ndimage.morphology.binary_closing(homo_patches, SE)
        patch_index, num_patches = ndimage.measurements.label(homo_patches)
        for ii in range(1, num_patches):
            idx = np.where(patch_index == ii)
            if idx[0].size > min_num_pixels:
                C_ref_local_mean = np.mean(C_ref[idx])
                C_pro_local_mean = np.mean(C_pro[idx])
                DC_local = (2. * C_ref_local_mean * C_pro_local_mean + Kc) \
                     / (np.power(C_ref_local_mean, 2) + np.power(C_pro_local_mean, 2) + Kc)
                H_ref_mean = np.arctan2(np.mean(np.cos(H_ref[idx])), np.mean(np.sin(H_ref[idx])))
                H_pro_mean = np.arctan2(np.mean(np.cos(H_pro[idx])), np.mean(np.sin(H_pro[idx])))
                DH_local = (2. * H_ref_mean * H_pro_mean + Kh) \
                     / (np.power(H_ref_mean, 2) + np.power(H_pro_mean, 2) + Kh)
                DL_local = np.mean(DL[idx])
                temp = 1. - DH_local * DC_local * DL_local
                if sq:
                    temp = np.power(temp, 2)
                CDw[idx] = temp
                if np.isnan(temp):
                    pass
                wp = (1. * idx[0].size) / Np
                cd += wp * temp
                wpt += 1  # wp
    return CDw, cd


def conf25_circularhue_patches(Ref_image, Pro_image, th=0.035, r=1.5, min_num_pixels = 4, sq = False, mode = 'same'):
    # SE = np.ones((np.int_(2 * r + 1), np.int_(2 * r + 1)))
    LBP, _ = CT.lbp(Ref_image, th=th, r=r)
    SE = np.ones((2, 2))
    list_of_patterns = np.unique(LBP)
    cd = 0.
    wpt = 0.
    Lab_ref = color.rgb2lab(Ref_image)
    Lab_pro = color.rgb2lab(Pro_image)
    H_ref = np.arctan2(Lab_ref[:, :, 2], Lab_ref[:, :, 1])
    H_pro = np.arctan2(Lab_pro[:, :, 2], Lab_pro[:, :, 1])
    C_ref = np.sqrt(np.power(Lab_ref[:, :, 1], 2) + np.power(Lab_ref[:, :, 2], 2))
    C_pro = np.sqrt(np.power(Lab_pro[:, :, 1], 2) + np.power(Lab_pro[:, :, 2], 2))
    DL, _ = MU.SSIM(Lab_ref[:, :, 0], Lab_pro[:, :, 0], mode=mode)
    Kh = (360 * 0.01) ** 2
    Kc = (180 * 0.01) ** 2
    CDw = np.zeros(DL.shape)
    Np = DL.size
    for pp in list_of_patterns:
        homo_patches = LBP == pp
        homo_patches = ndimage.morphology.binary_closing(homo_patches, SE)
        patch_index, num_patches = ndimage.measurements.label(homo_patches)
        for ii in range(1, num_patches):
            idx = np.where(patch_index == ii)
            if idx[0].size > min_num_pixels:
                C_ref_local_mean = np.mean(C_ref[idx])
                C_pro_local_mean = np.mean(C_pro[idx])
                DC_local = (2. * C_ref_local_mean * C_pro_local_mean + Kc) \
                     / (np.power(C_ref_local_mean, 2) + np.power(C_pro_local_mean, 2) + Kc)
                H_ref_mean = np.arctan2(np.mean(np.cos(H_ref[idx])), np.mean(np.sin(H_ref[idx])))
                H_pro_mean = np.arctan2(np.mean(np.cos(H_pro[idx])), np.mean(np.sin(H_pro[idx])))
                DH_local = (2. * H_ref_mean * H_pro_mean + Kh) \
                     / (np.power(H_ref_mean, 2) + np.power(H_pro_mean, 2) + Kh)
                DL_local = np.mean(DL[idx])
                temp = 1. - DH_local * DC_local * DL_local
                if sq:
                    temp = np.power(temp, 2)
                CDw[idx] = temp
                if np.isnan(temp):
                    pass
                wp = (1. * idx[0].size) / Np
                cd += wp * temp
                wpt += 1  # wp
    return CDw, cd



def conf26_circularhue_patches(Ref_image, Pro_image, th=0.035, r=2., min_num_pixels = 4, sq = False, mode = 'same'):
    # SE = np.ones((np.int_(2 * r + 1), np.int_(2 * r + 1)))
    LBP, _ = CT.lbp(Ref_image, th=th, r=r)
    SE = np.ones((2, 2))
    list_of_patterns = np.unique(LBP)
    cd = 0.
    wpt = 0.
    Lab_ref = color.rgb2lab(Ref_image)
    Lab_pro = color.rgb2lab(Pro_image)
    H_ref = np.arctan2(Lab_ref[:, :, 2], Lab_ref[:, :, 1])
    H_pro = np.arctan2(Lab_pro[:, :, 2], Lab_pro[:, :, 1])
    C_ref = np.sqrt(np.power(Lab_ref[:, :, 1], 2) + np.power(Lab_ref[:, :, 2], 2))
    C_pro = np.sqrt(np.power(Lab_pro[:, :, 1], 2) + np.power(Lab_pro[:, :, 2], 2))
    DL, _ = MU.SSIM(Lab_ref[:, :, 0], Lab_pro[:, :, 0], mode=mode)
    Kh = (360 * 0.01) ** 2
    Kc = (180 * 0.01) ** 2
    CDw = np.zeros(DL.shape)
    Np = DL.size
    for pp in list_of_patterns:
        homo_patches = LBP == pp
        homo_patches = ndimage.morphology.binary_closing(homo_patches, SE)
        patch_index, num_patches = ndimage.measurements.label(homo_patches)
        for ii in range(1, num_patches):
            idx = np.where(patch_index == ii)
            if idx[0].size > min_num_pixels:
                C_ref_local_mean = np.mean(C_ref[idx])
                C_pro_local_mean = np.mean(C_pro[idx])
                DC_local = (2. * C_ref_local_mean * C_pro_local_mean + Kc) \
                     / (np.power(C_ref_local_mean, 2) + np.power(C_pro_local_mean, 2) + Kc)
                H_ref_mean = np.arctan2(np.mean(np.cos(H_ref[idx])), np.mean(np.sin(H_ref[idx])))
                H_pro_mean = np.arctan2(np.mean(np.cos(H_pro[idx])), np.mean(np.sin(H_pro[idx])))
                DH_local = (2. * H_ref_mean * H_pro_mean + Kh) \
                     / (np.power(H_ref_mean, 2) + np.power(H_pro_mean, 2) + Kh)
                DL_local = np.mean(DL[idx])
                temp = 1. - DH_local * DC_local * DL_local
                if sq:
                    temp = np.power(temp, 2)
                CDw[idx] = temp
                if np.isnan(temp):
                    pass
                wp = (1. * idx[0].size) / Np
                cd += wp * temp
                wpt += 1  # wp
    return CDw, cd



def conf27_circularhue_patches(Ref_image, Pro_image, th=0.035, r=3., min_num_pixels = 4, sq = False, mode = 'same'):
    # SE = np.ones((np.int_(2 * r + 1), np.int_(2 * r + 1)))
    LBP, _ = CT.lbp(Ref_image, th=th, r=r)
    SE = np.ones((2, 2))
    list_of_patterns = np.unique(LBP)
    cd = 0.
    wpt = 0.
    Lab_ref = color.rgb2lab(Ref_image)
    Lab_pro = color.rgb2lab(Pro_image)
    H_ref = np.arctan2(Lab_ref[:, :, 2], Lab_ref[:, :, 1])
    H_pro = np.arctan2(Lab_pro[:, :, 2], Lab_pro[:, :, 1])
    C_ref = np.sqrt(np.power(Lab_ref[:, :, 1], 2) + np.power(Lab_ref[:, :, 2], 2))
    C_pro = np.sqrt(np.power(Lab_pro[:, :, 1], 2) + np.power(Lab_pro[:, :, 2], 2))
    DL, _ = MU.SSIM(Lab_ref[:, :, 0], Lab_pro[:, :, 0], mode=mode)
    Kh = (360 * 0.01) ** 2
    Kc = (180 * 0.01) ** 2
    CDw = np.zeros(DL.shape)
    Np = DL.size
    for pp in list_of_patterns:
        homo_patches = LBP == pp
        homo_patches = ndimage.morphology.binary_closing(homo_patches, SE)
        patch_index, num_patches = ndimage.measurements.label(homo_patches)
        for ii in range(1, num_patches):
            idx = np.where(patch_index == ii)
            if idx[0].size > min_num_pixels:
                C_ref_local_mean = np.mean(C_ref[idx])
                C_pro_local_mean = np.mean(C_pro[idx])
                DC_local = (2. * C_ref_local_mean * C_pro_local_mean + Kc) \
                     / (np.power(C_ref_local_mean, 2) + np.power(C_pro_local_mean, 2) + Kc)
                H_ref_mean = np.arctan2(np.mean(np.cos(H_ref[idx])), np.mean(np.sin(H_ref[idx])))
                H_pro_mean = np.arctan2(np.mean(np.cos(H_pro[idx])), np.mean(np.sin(H_pro[idx])))
                DH_local = (2. * H_ref_mean * H_pro_mean + Kh) \
                     / (np.power(H_ref_mean, 2) + np.power(H_pro_mean, 2) + Kh)
                DL_local = np.mean(DL[idx])
                temp = 1. - DH_local * DC_local * DL_local
                if sq:
                    temp = np.power(temp, 2)
                CDw[idx] = temp
                if np.isnan(temp):
                    pass
                wp = (1. * idx[0].size) / Np
                cd += wp * temp
                wpt += 1  # wp
    return CDw, cd



def conf28_circularhue_patches(Ref_image, Pro_image, th=0.04, r=1., min_num_pixels = 4, sq = False, mode = 'same'):
    # SE = np.ones((np.int_(2 * r + 1), np.int_(2 * r + 1)))
    LBP, _ = CT.lbp(Ref_image, th=th, r=r)
    SE = np.ones((2, 2))
    list_of_patterns = np.unique(LBP)
    cd = 0.
    wpt = 0.
    Lab_ref = color.rgb2lab(Ref_image)
    Lab_pro = color.rgb2lab(Pro_image)
    H_ref = np.arctan2(Lab_ref[:, :, 2], Lab_ref[:, :, 1])
    H_pro = np.arctan2(Lab_pro[:, :, 2], Lab_pro[:, :, 1])
    C_ref = np.sqrt(np.power(Lab_ref[:, :, 1], 2) + np.power(Lab_ref[:, :, 2], 2))
    C_pro = np.sqrt(np.power(Lab_pro[:, :, 1], 2) + np.power(Lab_pro[:, :, 2], 2))
    DL, _ = MU.SSIM(Lab_ref[:, :, 0], Lab_pro[:, :, 0], mode=mode)
    Kh = (360 * 0.01) ** 2
    Kc = (180 * 0.01) ** 2
    CDw = np.zeros(DL.shape)
    Np = DL.size
    for pp in list_of_patterns:
        homo_patches = LBP == pp
        homo_patches = ndimage.morphology.binary_closing(homo_patches, SE)
        patch_index, num_patches = ndimage.measurements.label(homo_patches)
        for ii in range(1, num_patches):
            idx = np.where(patch_index == ii)
            if idx[0].size > min_num_pixels:
                C_ref_local_mean = np.mean(C_ref[idx])
                C_pro_local_mean = np.mean(C_pro[idx])
                DC_local = (2. * C_ref_local_mean * C_pro_local_mean + Kc) \
                     / (np.power(C_ref_local_mean, 2) + np.power(C_pro_local_mean, 2) + Kc)
                H_ref_mean = np.arctan2(np.mean(np.cos(H_ref[idx])), np.mean(np.sin(H_ref[idx])))
                H_pro_mean = np.arctan2(np.mean(np.cos(H_pro[idx])), np.mean(np.sin(H_pro[idx])))
                DH_local = (2. * H_ref_mean * H_pro_mean + Kh) \
                     / (np.power(H_ref_mean, 2) + np.power(H_pro_mean, 2) + Kh)
                DL_local = np.mean(DL[idx])
                temp = 1. - DH_local * DC_local * DL_local
                if sq:
                    temp = np.power(temp, 2)
                CDw[idx] = temp
                if np.isnan(temp):
                    pass
                wp = (1. * idx[0].size) / Np
                cd += wp * temp
                wpt += 1  # wp
    return CDw, cd


def conf29_circularhue_patches(Ref_image, Pro_image, th=0.04, r=1.5, min_num_pixels = 4, sq = False, mode = 'same'):
    # SE = np.ones((np.int_(2 * r + 1), np.int_(2 * r + 1)))
    LBP, _ = CT.lbp(Ref_image, th=th, r=r)
    SE = np.ones((2, 2))
    list_of_patterns = np.unique(LBP)
    cd = 0.
    wpt = 0.
    Lab_ref = color.rgb2lab(Ref_image)
    Lab_pro = color.rgb2lab(Pro_image)
    H_ref = np.arctan2(Lab_ref[:, :, 2], Lab_ref[:, :, 1])
    H_pro = np.arctan2(Lab_pro[:, :, 2], Lab_pro[:, :, 1])
    C_ref = np.sqrt(np.power(Lab_ref[:, :, 1], 2) + np.power(Lab_ref[:, :, 2], 2))
    C_pro = np.sqrt(np.power(Lab_pro[:, :, 1], 2) + np.power(Lab_pro[:, :, 2], 2))
    DL, _ = MU.SSIM(Lab_ref[:, :, 0], Lab_pro[:, :, 0], mode=mode)
    Kh = (360 * 0.01) ** 2
    Kc = (180 * 0.01) ** 2
    CDw = np.zeros(DL.shape)
    Np = DL.size
    for pp in list_of_patterns:
        homo_patches = LBP == pp
        homo_patches = ndimage.morphology.binary_closing(homo_patches, SE)
        patch_index, num_patches = ndimage.measurements.label(homo_patches)
        for ii in range(1, num_patches):
            idx = np.where(patch_index == ii)
            if idx[0].size > min_num_pixels:
                C_ref_local_mean = np.mean(C_ref[idx])
                C_pro_local_mean = np.mean(C_pro[idx])
                DC_local = (2. * C_ref_local_mean * C_pro_local_mean + Kc) \
                     / (np.power(C_ref_local_mean, 2) + np.power(C_pro_local_mean, 2) + Kc)
                H_ref_mean = np.arctan2(np.mean(np.cos(H_ref[idx])), np.mean(np.sin(H_ref[idx])))
                H_pro_mean = np.arctan2(np.mean(np.cos(H_pro[idx])), np.mean(np.sin(H_pro[idx])))
                DH_local = (2. * H_ref_mean * H_pro_mean + Kh) \
                     / (np.power(H_ref_mean, 2) + np.power(H_pro_mean, 2) + Kh)
                DL_local = np.mean(DL[idx])
                temp = 1. - DH_local * DC_local * DL_local
                if sq:
                    temp = np.power(temp, 2)
                CDw[idx] = temp
                if np.isnan(temp):
                    pass
                wp = (1. * idx[0].size) / Np
                cd += wp * temp
                wpt += 1  # wp
    return CDw, cd



def conf30_circularhue_patches(Ref_image, Pro_image, th=0.04, r=2., min_num_pixels = 4, sq = False, mode = 'same'):
    # SE = np.ones((np.int_(2 * r + 1), np.int_(2 * r + 1)))
    LBP, _ = CT.lbp(Ref_image, th=th, r=r)
    SE = np.ones((2, 2))
    list_of_patterns = np.unique(LBP)
    cd = 0.
    wpt = 0.
    Lab_ref = color.rgb2lab(Ref_image)
    Lab_pro = color.rgb2lab(Pro_image)
    H_ref = np.arctan2(Lab_ref[:, :, 2], Lab_ref[:, :, 1])
    H_pro = np.arctan2(Lab_pro[:, :, 2], Lab_pro[:, :, 1])
    C_ref = np.sqrt(np.power(Lab_ref[:, :, 1], 2) + np.power(Lab_ref[:, :, 2], 2))
    C_pro = np.sqrt(np.power(Lab_pro[:, :, 1], 2) + np.power(Lab_pro[:, :, 2], 2))
    DL, _ = MU.SSIM(Lab_ref[:, :, 0], Lab_pro[:, :, 0], mode=mode)
    Kh = (360 * 0.01) ** 2
    Kc = (180 * 0.01) ** 2
    CDw = np.zeros(DL.shape)
    Np = DL.size
    for pp in list_of_patterns:
        homo_patches = LBP == pp
        homo_patches = ndimage.morphology.binary_closing(homo_patches, SE)
        patch_index, num_patches = ndimage.measurements.label(homo_patches)
        for ii in range(1, num_patches):
            idx = np.where(patch_index == ii)
            if idx[0].size > min_num_pixels:
                C_ref_local_mean = np.mean(C_ref[idx])
                C_pro_local_mean = np.mean(C_pro[idx])
                DC_local = (2. * C_ref_local_mean * C_pro_local_mean + Kc) \
                     / (np.power(C_ref_local_mean, 2) + np.power(C_pro_local_mean, 2) + Kc)
                H_ref_mean = np.arctan2(np.mean(np.cos(H_ref[idx])), np.mean(np.sin(H_ref[idx])))
                H_pro_mean = np.arctan2(np.mean(np.cos(H_pro[idx])), np.mean(np.sin(H_pro[idx])))
                DH_local = (2. * H_ref_mean * H_pro_mean + Kh) \
                     / (np.power(H_ref_mean, 2) + np.power(H_pro_mean, 2) + Kh)
                DL_local = np.mean(DL[idx])
                temp = 1. - DH_local * DC_local * DL_local
                if sq:
                    temp = np.power(temp, 2)
                CDw[idx] = temp
                if np.isnan(temp):
                    pass
                wp = (1. * idx[0].size) / Np
                cd += wp * temp
                wpt += 1  # wp
    return CDw, cd



def conf31_circularhue_patches(Ref_image, Pro_image, th=0.04, r=3., min_num_pixels = 4, sq = False, mode = 'same'):
    # SE = np.ones((np.int_(2 * r + 1), np.int_(2 * r + 1)))
    LBP, _ = CT.lbp(Ref_image, th=th, r=r)
    SE = np.ones((2, 2))
    list_of_patterns = np.unique(LBP)
    cd = 0.
    wpt = 0.
    Lab_ref = color.rgb2lab(Ref_image)
    Lab_pro = color.rgb2lab(Pro_image)
    H_ref = np.arctan2(Lab_ref[:, :, 2], Lab_ref[:, :, 1])
    H_pro = np.arctan2(Lab_pro[:, :, 2], Lab_pro[:, :, 1])
    C_ref = np.sqrt(np.power(Lab_ref[:, :, 1], 2) + np.power(Lab_ref[:, :, 2], 2))
    C_pro = np.sqrt(np.power(Lab_pro[:, :, 1], 2) + np.power(Lab_pro[:, :, 2], 2))
    DL, _ = MU.SSIM(Lab_ref[:, :, 0], Lab_pro[:, :, 0], mode=mode)
    Kh = (360 * 0.01) ** 2
    Kc = (180 * 0.01) ** 2
    CDw = np.zeros(DL.shape)
    Np = DL.size
    for pp in list_of_patterns:
        homo_patches = LBP == pp
        homo_patches = ndimage.morphology.binary_closing(homo_patches, SE)
        patch_index, num_patches = ndimage.measurements.label(homo_patches)
        for ii in range(1, num_patches):
            idx = np.where(patch_index == ii)
            if idx[0].size > min_num_pixels:
                C_ref_local_mean = np.mean(C_ref[idx])
                C_pro_local_mean = np.mean(C_pro[idx])
                DC_local = (2. * C_ref_local_mean * C_pro_local_mean + Kc) \
                     / (np.power(C_ref_local_mean, 2) + np.power(C_pro_local_mean, 2) + Kc)
                H_ref_mean = np.arctan2(np.mean(np.cos(H_ref[idx])), np.mean(np.sin(H_ref[idx])))
                H_pro_mean = np.arctan2(np.mean(np.cos(H_pro[idx])), np.mean(np.sin(H_pro[idx])))
                DH_local = (2. * H_ref_mean * H_pro_mean + Kh) \
                     / (np.power(H_ref_mean, 2) + np.power(H_pro_mean, 2) + Kh)
                DL_local = np.mean(DL[idx])
                temp = 1. - DH_local * DC_local * DL_local
                if sq:
                    temp = np.power(temp, 2)
                CDw[idx] = temp
                if np.isnan(temp):
                    pass
                wp = (1. * idx[0].size) / Np
                cd += wp * temp
                wpt += 1  # wp
    return CDw, cd



def conf32_circularhue_patches(Ref_image, Pro_image, th=0.045, r=1., min_num_pixels = 4, sq = False, mode = 'same'):
    # SE = np.ones((np.int_(2 * r + 1), np.int_(2 * r + 1)))
    LBP, _ = CT.lbp(Ref_image, th=th, r=r)
    SE = np.ones((2, 2))
    list_of_patterns = np.unique(LBP)
    cd = 0.
    wpt = 0.
    Lab_ref = color.rgb2lab(Ref_image)
    Lab_pro = color.rgb2lab(Pro_image)
    H_ref = np.arctan2(Lab_ref[:, :, 2], Lab_ref[:, :, 1])
    H_pro = np.arctan2(Lab_pro[:, :, 2], Lab_pro[:, :, 1])
    C_ref = np.sqrt(np.power(Lab_ref[:, :, 1], 2) + np.power(Lab_ref[:, :, 2], 2))
    C_pro = np.sqrt(np.power(Lab_pro[:, :, 1], 2) + np.power(Lab_pro[:, :, 2], 2))
    DL, _ = MU.SSIM(Lab_ref[:, :, 0], Lab_pro[:, :, 0], mode=mode)
    Kh = (360 * 0.01) ** 2
    Kc = (180 * 0.01) ** 2
    CDw = np.zeros(DL.shape)
    Np = DL.size
    for pp in list_of_patterns:
        homo_patches = LBP == pp
        homo_patches = ndimage.morphology.binary_closing(homo_patches, SE)
        patch_index, num_patches = ndimage.measurements.label(homo_patches)
        for ii in range(1, num_patches):
            idx = np.where(patch_index == ii)
            if idx[0].size > min_num_pixels:
                C_ref_local_mean = np.mean(C_ref[idx])
                C_pro_local_mean = np.mean(C_pro[idx])
                DC_local = (2. * C_ref_local_mean * C_pro_local_mean + Kc) \
                     / (np.power(C_ref_local_mean, 2) + np.power(C_pro_local_mean, 2) + Kc)
                H_ref_mean = np.arctan2(np.mean(np.cos(H_ref[idx])), np.mean(np.sin(H_ref[idx])))
                H_pro_mean = np.arctan2(np.mean(np.cos(H_pro[idx])), np.mean(np.sin(H_pro[idx])))
                DH_local = (2. * H_ref_mean * H_pro_mean + Kh) \
                     / (np.power(H_ref_mean, 2) + np.power(H_pro_mean, 2) + Kh)
                DL_local = np.mean(DL[idx])
                temp = 1. - DH_local * DC_local * DL_local
                if sq:
                    temp = np.power(temp, 2)
                CDw[idx] = temp
                if np.isnan(temp):
                    pass
                wp = (1. * idx[0].size) / Np
                cd += wp * temp
                wpt += 1  # wp
    return CDw, cd


def conf33_circularhue_patches(Ref_image, Pro_image, th=0.045, r=1.5, min_num_pixels = 4, sq = False, mode = 'same'):
    # SE = np.ones((np.int_(2 * r + 1), np.int_(2 * r + 1)))
    LBP, _ = CT.lbp(Ref_image, th=th, r=r)
    SE = np.ones((2, 2))
    list_of_patterns = np.unique(LBP)
    cd = 0.
    wpt = 0.
    Lab_ref = color.rgb2lab(Ref_image)
    Lab_pro = color.rgb2lab(Pro_image)
    H_ref = np.arctan2(Lab_ref[:, :, 2], Lab_ref[:, :, 1])
    H_pro = np.arctan2(Lab_pro[:, :, 2], Lab_pro[:, :, 1])
    C_ref = np.sqrt(np.power(Lab_ref[:, :, 1], 2) + np.power(Lab_ref[:, :, 2], 2))
    C_pro = np.sqrt(np.power(Lab_pro[:, :, 1], 2) + np.power(Lab_pro[:, :, 2], 2))
    DL, _ = MU.SSIM(Lab_ref[:, :, 0], Lab_pro[:, :, 0], mode=mode)
    Kh = (360 * 0.01) ** 2
    Kc = (180 * 0.01) ** 2
    CDw = np.zeros(DL.shape)
    Np = DL.size
    for pp in list_of_patterns:
        homo_patches = LBP == pp
        homo_patches = ndimage.morphology.binary_closing(homo_patches, SE)
        patch_index, num_patches = ndimage.measurements.label(homo_patches)
        for ii in range(1, num_patches):
            idx = np.where(patch_index == ii)
            if idx[0].size > min_num_pixels:
                C_ref_local_mean = np.mean(C_ref[idx])
                C_pro_local_mean = np.mean(C_pro[idx])
                DC_local = (2. * C_ref_local_mean * C_pro_local_mean + Kc) \
                     / (np.power(C_ref_local_mean, 2) + np.power(C_pro_local_mean, 2) + Kc)
                H_ref_mean = np.arctan2(np.mean(np.cos(H_ref[idx])), np.mean(np.sin(H_ref[idx])))
                H_pro_mean = np.arctan2(np.mean(np.cos(H_pro[idx])), np.mean(np.sin(H_pro[idx])))
                DH_local = (2. * H_ref_mean * H_pro_mean + Kh) \
                     / (np.power(H_ref_mean, 2) + np.power(H_pro_mean, 2) + Kh)
                DL_local = np.mean(DL[idx])
                temp = 1. - DH_local * DC_local * DL_local
                if sq:
                    temp = np.power(temp, 2)
                CDw[idx] = temp
                if np.isnan(temp):
                    pass
                wp = (1. * idx[0].size) / Np
                cd += wp * temp
                wpt += 1  # wp
    return CDw, cd



def conf34_circularhue_patches(Ref_image, Pro_image, th=0.045, r=2., min_num_pixels = 4, sq = False, mode = 'same'):
    # SE = np.ones((np.int_(2 * r + 1), np.int_(2 * r + 1)))
    LBP, _ = CT.lbp(Ref_image, th=th, r=r)
    SE = np.ones((2, 2))
    list_of_patterns = np.unique(LBP)
    cd = 0.
    wpt = 0.
    Lab_ref = color.rgb2lab(Ref_image)
    Lab_pro = color.rgb2lab(Pro_image)
    H_ref = np.arctan2(Lab_ref[:, :, 2], Lab_ref[:, :, 1])
    H_pro = np.arctan2(Lab_pro[:, :, 2], Lab_pro[:, :, 1])
    C_ref = np.sqrt(np.power(Lab_ref[:, :, 1], 2) + np.power(Lab_ref[:, :, 2], 2))
    C_pro = np.sqrt(np.power(Lab_pro[:, :, 1], 2) + np.power(Lab_pro[:, :, 2], 2))
    DL, _ = MU.SSIM(Lab_ref[:, :, 0], Lab_pro[:, :, 0], mode=mode)
    Kh = (360 * 0.01) ** 2
    Kc = (180 * 0.01) ** 2
    CDw = np.zeros(DL.shape)
    Np = DL.size
    for pp in list_of_patterns:
        homo_patches = LBP == pp
        homo_patches = ndimage.morphology.binary_closing(homo_patches, SE)
        patch_index, num_patches = ndimage.measurements.label(homo_patches)
        for ii in range(1, num_patches):
            idx = np.where(patch_index == ii)
            if idx[0].size > min_num_pixels:
                C_ref_local_mean = np.mean(C_ref[idx])
                C_pro_local_mean = np.mean(C_pro[idx])
                DC_local = (2. * C_ref_local_mean * C_pro_local_mean + Kc) \
                     / (np.power(C_ref_local_mean, 2) + np.power(C_pro_local_mean, 2) + Kc)
                H_ref_mean = np.arctan2(np.mean(np.cos(H_ref[idx])), np.mean(np.sin(H_ref[idx])))
                H_pro_mean = np.arctan2(np.mean(np.cos(H_pro[idx])), np.mean(np.sin(H_pro[idx])))
                DH_local = (2. * H_ref_mean * H_pro_mean + Kh) \
                     / (np.power(H_ref_mean, 2) + np.power(H_pro_mean, 2) + Kh)
                DL_local = np.mean(DL[idx])
                temp = 1. - DH_local * DC_local * DL_local
                if sq:
                    temp = np.power(temp, 2)
                CDw[idx] = temp
                if np.isnan(temp):
                    pass
                wp = (1. * idx[0].size) / Np
                cd += wp * temp
                wpt += 1  # wp
    return CDw, cd



def conf35_circularhue_patches(Ref_image, Pro_image, th=0.045, r=3., min_num_pixels = 4, sq = False, mode = 'same'):
    # SE = np.ones((np.int_(2 * r + 1), np.int_(2 * r + 1)))
    LBP, _ = CT.lbp(Ref_image, th=th, r=r)
    SE = np.ones((2, 2))
    list_of_patterns = np.unique(LBP)
    cd = 0.
    wpt = 0.
    Lab_ref = color.rgb2lab(Ref_image)
    Lab_pro = color.rgb2lab(Pro_image)
    H_ref = np.arctan2(Lab_ref[:, :, 2], Lab_ref[:, :, 1])
    H_pro = np.arctan2(Lab_pro[:, :, 2], Lab_pro[:, :, 1])
    C_ref = np.sqrt(np.power(Lab_ref[:, :, 1], 2) + np.power(Lab_ref[:, :, 2], 2))
    C_pro = np.sqrt(np.power(Lab_pro[:, :, 1], 2) + np.power(Lab_pro[:, :, 2], 2))
    DL, _ = MU.SSIM(Lab_ref[:, :, 0], Lab_pro[:, :, 0], mode=mode)
    Kh = (360 * 0.01) ** 2
    Kc = (180 * 0.01) ** 2
    CDw = np.zeros(DL.shape)
    Np = DL.size
    for pp in list_of_patterns:
        homo_patches = LBP == pp
        homo_patches = ndimage.morphology.binary_closing(homo_patches, SE)
        patch_index, num_patches = ndimage.measurements.label(homo_patches)
        for ii in range(1, num_patches):
            idx = np.where(patch_index == ii)
            if idx[0].size > min_num_pixels:
                C_ref_local_mean = np.mean(C_ref[idx])
                C_pro_local_mean = np.mean(C_pro[idx])
                DC_local = (2. * C_ref_local_mean * C_pro_local_mean + Kc) \
                     / (np.power(C_ref_local_mean, 2) + np.power(C_pro_local_mean, 2) + Kc)
                H_ref_mean = np.arctan2(np.mean(np.cos(H_ref[idx])), np.mean(np.sin(H_ref[idx])))
                H_pro_mean = np.arctan2(np.mean(np.cos(H_pro[idx])), np.mean(np.sin(H_pro[idx])))
                DH_local = (2. * H_ref_mean * H_pro_mean + Kh) \
                     / (np.power(H_ref_mean, 2) + np.power(H_pro_mean, 2) + Kh)
                DL_local = np.mean(DL[idx])
                temp = 1. - DH_local * DC_local * DL_local
                if sq:
                    temp = np.power(temp, 2)
                CDw[idx] = temp
                if np.isnan(temp):
                    pass
                wp = (1. * idx[0].size) / Np
                cd += wp * temp
                wpt += 1  # wp
    return CDw, cd


def conf36_circularhue_patches(Ref_image, Pro_image, th=0.05, r=1., min_num_pixels = 4, sq = False, mode = 'same'):
    # SE = np.ones((np.int_(2 * r + 1), np.int_(2 * r + 1)))
    LBP, _ = CT.lbp(Ref_image, th=th, r=r)
    SE = np.ones((2, 2))
    list_of_patterns = np.unique(LBP)
    cd = 0.
    wpt = 0.
    Lab_ref = color.rgb2lab(Ref_image)
    Lab_pro = color.rgb2lab(Pro_image)
    H_ref = np.arctan2(Lab_ref[:, :, 2], Lab_ref[:, :, 1])
    H_pro = np.arctan2(Lab_pro[:, :, 2], Lab_pro[:, :, 1])
    C_ref = np.sqrt(np.power(Lab_ref[:, :, 1], 2) + np.power(Lab_ref[:, :, 2], 2))
    C_pro = np.sqrt(np.power(Lab_pro[:, :, 1], 2) + np.power(Lab_pro[:, :, 2], 2))
    DL, _ = MU.SSIM(Lab_ref[:, :, 0], Lab_pro[:, :, 0], mode=mode)
    Kh = (360 * 0.01) ** 2
    Kc = (180 * 0.01) ** 2
    CDw = np.zeros(DL.shape)
    Np = DL.size
    for pp in list_of_patterns:
        homo_patches = LBP == pp
        homo_patches = ndimage.morphology.binary_closing(homo_patches, SE)
        patch_index, num_patches = ndimage.measurements.label(homo_patches)
        for ii in range(1, num_patches):
            idx = np.where(patch_index == ii)
            if idx[0].size > min_num_pixels:
                C_ref_local_mean = np.mean(C_ref[idx])
                C_pro_local_mean = np.mean(C_pro[idx])
                DC_local = (2. * C_ref_local_mean * C_pro_local_mean + Kc) \
                     / (np.power(C_ref_local_mean, 2) + np.power(C_pro_local_mean, 2) + Kc)
                H_ref_mean = np.arctan2(np.mean(np.cos(H_ref[idx])), np.mean(np.sin(H_ref[idx])))
                H_pro_mean = np.arctan2(np.mean(np.cos(H_pro[idx])), np.mean(np.sin(H_pro[idx])))
                DH_local = (2. * H_ref_mean * H_pro_mean + Kh) \
                     / (np.power(H_ref_mean, 2) + np.power(H_pro_mean, 2) + Kh)
                DL_local = np.mean(DL[idx])
                temp = 1. - DH_local * DC_local * DL_local
                if sq:
                    temp = np.power(temp, 2)
                CDw[idx] = temp
                if np.isnan(temp):
                    pass
                wp = (1. * idx[0].size) / Np
                cd += wp * temp
                wpt += 1  # wp
    return CDw, cd


def conf37_circularhue_patches(Ref_image, Pro_image, th=0.05, r=1.5, min_num_pixels = 4, sq = False, mode = 'same'):
    # SE = np.ones((np.int_(2 * r + 1), np.int_(2 * r + 1)))
    LBP, _ = CT.lbp(Ref_image, th=th, r=r)
    SE = np.ones((2, 2))
    list_of_patterns = np.unique(LBP)
    cd = 0.
    wpt = 0.
    Lab_ref = color.rgb2lab(Ref_image)
    Lab_pro = color.rgb2lab(Pro_image)
    H_ref = np.arctan2(Lab_ref[:, :, 2], Lab_ref[:, :, 1])
    H_pro = np.arctan2(Lab_pro[:, :, 2], Lab_pro[:, :, 1])
    C_ref = np.sqrt(np.power(Lab_ref[:, :, 1], 2) + np.power(Lab_ref[:, :, 2], 2))
    C_pro = np.sqrt(np.power(Lab_pro[:, :, 1], 2) + np.power(Lab_pro[:, :, 2], 2))
    DL, _ = MU.SSIM(Lab_ref[:, :, 0], Lab_pro[:, :, 0], mode=mode)
    Kh = (360 * 0.01) ** 2
    Kc = (180 * 0.01) ** 2
    CDw = np.zeros(DL.shape)
    Np = DL.size
    for pp in list_of_patterns:
        homo_patches = LBP == pp
        homo_patches = ndimage.morphology.binary_closing(homo_patches, SE)
        patch_index, num_patches = ndimage.measurements.label(homo_patches)
        for ii in range(1, num_patches):
            idx = np.where(patch_index == ii)
            if idx[0].size > min_num_pixels:
                C_ref_local_mean = np.mean(C_ref[idx])
                C_pro_local_mean = np.mean(C_pro[idx])
                DC_local = (2. * C_ref_local_mean * C_pro_local_mean + Kc) \
                     / (np.power(C_ref_local_mean, 2) + np.power(C_pro_local_mean, 2) + Kc)
                H_ref_mean = np.arctan2(np.mean(np.cos(H_ref[idx])), np.mean(np.sin(H_ref[idx])))
                H_pro_mean = np.arctan2(np.mean(np.cos(H_pro[idx])), np.mean(np.sin(H_pro[idx])))
                DH_local = (2. * H_ref_mean * H_pro_mean + Kh) \
                     / (np.power(H_ref_mean, 2) + np.power(H_pro_mean, 2) + Kh)
                DL_local = np.mean(DL[idx])
                temp = 1. - DH_local * DC_local * DL_local
                if sq:
                    temp = np.power(temp, 2)
                CDw[idx] = temp
                if np.isnan(temp):
                    pass
                wp = (1. * idx[0].size) / Np
                cd += wp * temp
                wpt += 1  # wp
    return CDw, cd



def conf38_circularhue_patches(Ref_image, Pro_image, th=0.05, r=2., min_num_pixels = 4, sq = False, mode = 'same'):
    # SE = np.ones((np.int_(2 * r + 1), np.int_(2 * r + 1)))
    LBP, _ = CT.lbp(Ref_image, th=th, r=r)
    SE = np.ones((2, 2))
    list_of_patterns = np.unique(LBP)
    cd = 0.
    wpt = 0.
    Lab_ref = color.rgb2lab(Ref_image)
    Lab_pro = color.rgb2lab(Pro_image)
    H_ref = np.arctan2(Lab_ref[:, :, 2], Lab_ref[:, :, 1])
    H_pro = np.arctan2(Lab_pro[:, :, 2], Lab_pro[:, :, 1])
    C_ref = np.sqrt(np.power(Lab_ref[:, :, 1], 2) + np.power(Lab_ref[:, :, 2], 2))
    C_pro = np.sqrt(np.power(Lab_pro[:, :, 1], 2) + np.power(Lab_pro[:, :, 2], 2))
    DL, _ = MU.SSIM(Lab_ref[:, :, 0], Lab_pro[:, :, 0], mode=mode)
    Kh = (360 * 0.01) ** 2
    Kc = (180 * 0.01) ** 2
    CDw = np.zeros(DL.shape)
    Np = DL.size
    for pp in list_of_patterns:
        homo_patches = LBP == pp
        homo_patches = ndimage.morphology.binary_closing(homo_patches, SE)
        patch_index, num_patches = ndimage.measurements.label(homo_patches)
        for ii in range(1, num_patches):
            idx = np.where(patch_index == ii)
            if idx[0].size > min_num_pixels:
                C_ref_local_mean = np.mean(C_ref[idx])
                C_pro_local_mean = np.mean(C_pro[idx])
                DC_local = (2. * C_ref_local_mean * C_pro_local_mean + Kc) \
                     / (np.power(C_ref_local_mean, 2) + np.power(C_pro_local_mean, 2) + Kc)
                H_ref_mean = np.arctan2(np.mean(np.cos(H_ref[idx])), np.mean(np.sin(H_ref[idx])))
                H_pro_mean = np.arctan2(np.mean(np.cos(H_pro[idx])), np.mean(np.sin(H_pro[idx])))
                DH_local = (2. * H_ref_mean * H_pro_mean + Kh) \
                     / (np.power(H_ref_mean, 2) + np.power(H_pro_mean, 2) + Kh)
                DL_local = np.mean(DL[idx])
                temp = 1. - DH_local * DC_local * DL_local
                if sq:
                    temp = np.power(temp, 2)
                CDw[idx] = temp
                if np.isnan(temp):
                    pass
                wp = (1. * idx[0].size) / Np
                cd += wp * temp
                wpt += 1  # wp
    return CDw, cd



def conf39_circularhue_patches(Ref_image, Pro_image, th=0.05, r=3., min_num_pixels = 4, sq = False, mode = 'same'):
    # SE = np.ones((np.int_(2 * r + 1), np.int_(2 * r + 1)))
    LBP, _ = CT.lbp(Ref_image, th=th, r=r)
    SE = np.ones((2, 2))
    list_of_patterns = np.unique(LBP)
    cd = 0.
    wpt = 0.
    Lab_ref = color.rgb2lab(Ref_image)
    Lab_pro = color.rgb2lab(Pro_image)
    H_ref = np.arctan2(Lab_ref[:, :, 2], Lab_ref[:, :, 1])
    H_pro = np.arctan2(Lab_pro[:, :, 2], Lab_pro[:, :, 1])
    C_ref = np.sqrt(np.power(Lab_ref[:, :, 1], 2) + np.power(Lab_ref[:, :, 2], 2))
    C_pro = np.sqrt(np.power(Lab_pro[:, :, 1], 2) + np.power(Lab_pro[:, :, 2], 2))
    DL, _ = MU.SSIM(Lab_ref[:, :, 0], Lab_pro[:, :, 0], mode=mode)
    Kh = (360 * 0.01) ** 2
    Kc = (180 * 0.01) ** 2
    CDw = np.zeros(DL.shape)
    Np = DL.size
    for pp in list_of_patterns:
        homo_patches = LBP == pp
        homo_patches = ndimage.morphology.binary_closing(homo_patches, SE)
        patch_index, num_patches = ndimage.measurements.label(homo_patches)
        for ii in range(1, num_patches):
            idx = np.where(patch_index == ii)
            if idx[0].size > min_num_pixels:
                C_ref_local_mean = np.mean(C_ref[idx])
                C_pro_local_mean = np.mean(C_pro[idx])
                DC_local = (2. * C_ref_local_mean * C_pro_local_mean + Kc) \
                     / (np.power(C_ref_local_mean, 2) + np.power(C_pro_local_mean, 2) + Kc)
                H_ref_mean = np.arctan2(np.mean(np.cos(H_ref[idx])), np.mean(np.sin(H_ref[idx])))
                H_pro_mean = np.arctan2(np.mean(np.cos(H_pro[idx])), np.mean(np.sin(H_pro[idx])))
                DH_local = (2. * H_ref_mean * H_pro_mean + Kh) \
                     / (np.power(H_ref_mean, 2) + np.power(H_pro_mean, 2) + Kh)
                DL_local = np.mean(DL[idx])
                temp = 1. - DH_local * DC_local * DL_local
                if sq:
                    temp = np.power(temp, 2)
                CDw[idx] = temp
                if np.isnan(temp):
                    pass
                wp = (1. * idx[0].size) / Np
                cd += wp * temp
                wpt += 1  # wp
    return CDw, cd
