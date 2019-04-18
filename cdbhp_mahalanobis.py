import numpy as np
from skimage import color
import cython_functions
from scipy import stats
from scipy import ndimage
import my_utilities as MU
import compute_texture as CT
import matplotlib.pyplot as plt
import color_spaces as CAM

def cd00_mahalanobis_patches(Ref_image, Pro_image, th=0.005, r=1.5):
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


def cd01_mahalanobis_patches(Ref_image, Pro_image, th=0.01, r=1.5):
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



def cd02_mahalanobis_patches(Ref_image, Pro_image, th=0.015, r=1.5):
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


def cd03_mahalanobis_patches(Ref_image, Pro_image, th=0.02, r=1.5):
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


def cd04_mahalanobis_patches(Ref_image, Pro_image, th=0.025, r=1.5):
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


def cd05_mahalanobis_patches(Ref_image, Pro_image, th=0.03, r=1.5):
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


def cd06_mahalanobis_patches(Ref_image, Pro_image, th=0.035, r=1.5):
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


def cd07_mahalanobis_patches(Ref_image, Pro_image, th=0.04, r=1.5):
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


def cd08_mahalanobis_patches(Ref_image, Pro_image, th=0.045, r=1.5):
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


def cd09_mahalanobis_patches(Ref_image, Pro_image, th=0.05, r=1.5):
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