import numpy as np
from skimage import color
import cython_functions
from scipy import stats
from scipy import ndimage
import my_utilities as MU
import compute_texture as CT
import matplotlib.pyplot as plt
import cd_measures_pack as CDM
import color_spaces as CAM



def conf00_deltaosa_patches(Ref_image, Pro_image, th=0.005, r=1., min_num_pixels = 4, sq = False, KLCH=np.array([1, 1, 1])):
    # SE = np.ones((np.int_(2 * r + 1), np.int_(2 * r + 1)))
    LBP, _ = CT.lbp(Ref_image, th=th, r=r)
    SE = np.ones((2, 2))
    list_of_patterns = np.unique(LBP)
    cd = 0.
    wpt = 0.
    Lab_ref = CAM.RGBToLjg(Ref_image)
    Lstd = Lab_ref[:, :, 0]
    astd = Lab_ref[:, :, 1]
    bstd = Lab_ref[:, :, 2]
    Lab_pro = CAM.RGBToLjg(Pro_image)
    Lsample = Lab_pro[:, :, 0]
    asample = Lab_pro[:, :, 1]
    bsample = Lab_pro[:, :, 2]
    CDw = np.zeros(LBP.shape)
    Np = LBP.size
    for pp in list_of_patterns:
        homo_patches = LBP == pp
        homo_patches = ndimage.morphology.binary_closing(homo_patches, SE)
        patch_index, num_patches = ndimage.measurements.label(homo_patches)
        for ii in range(1, num_patches):
            idx = np.where(patch_index == ii)
            if idx[0].size > min_num_pixels:
                temp = MU.deltaosa(Lstd[idx], astd[idx], bstd[idx], Lsample[idx], asample[idx], bsample[idx])
                if sq:
                    temp = np.power(temp, 2)
                if not np.isnan(temp):
                    CDw[idx] = temp
                    wp = (1. * idx[0].size) / Np
                    cd += wp * temp
                wpt += 1  # wp
    return CDw, cd


def conf01_deltaosa_patches(Ref_image, Pro_image, th=0.005, r=1.5, min_num_pixels = 4, sq = False, KLCH=np.array([1, 1, 1])):
    # SE = np.ones((np.int_(2 * r + 1), np.int_(2 * r + 1)))
    LBP, _ = CT.lbp(Ref_image, th=th, r=r)
    SE = np.ones((2, 2))
    list_of_patterns = np.unique(LBP)
    cd = 0.
    wpt = 0.
    Lab_ref = CAM.RGBToLjg(Ref_image)
    Lstd = Lab_ref[:, :, 0]
    astd = Lab_ref[:, :, 1]
    bstd = Lab_ref[:, :, 2]
    Lab_pro = CAM.RGBToLjg(Pro_image)
    Lsample = Lab_pro[:, :, 0]
    asample = Lab_pro[:, :, 1]
    bsample = Lab_pro[:, :, 2]
    CDw = np.zeros(LBP.shape)
    Np = LBP.size
    for pp in list_of_patterns:
        homo_patches = LBP == pp
        homo_patches = ndimage.morphology.binary_closing(homo_patches, SE)
        patch_index, num_patches = ndimage.measurements.label(homo_patches)
        for ii in range(1, num_patches):
            idx = np.where(patch_index == ii)
            if idx[0].size > min_num_pixels:
                temp = MU.deltaosa(Lstd[idx], astd[idx], bstd[idx], Lsample[idx], asample[idx], bsample[idx])
                if sq:
                    temp = np.power(temp, 2)
                if not np.isnan(temp):
                    CDw[idx] = temp
                    wp = (1. * idx[0].size) / Np
                    cd += wp * temp
                wpt += 1  # wp
    return CDw, cd


def conf02_deltaosa_patches(Ref_image, Pro_image, th=0.005, r=2., min_num_pixels = 4, sq = False, KLCH=np.array([1, 1, 1])):
    # SE = np.ones((np.int_(2 * r + 1), np.int_(2 * r + 1)))
    LBP, _ = CT.lbp(Ref_image, th=th, r=r)
    SE = np.ones((2, 2))
    list_of_patterns = np.unique(LBP)
    cd = 0.
    wpt = 0.
    Lab_ref = CAM.RGBToLjg(Ref_image)
    Lstd = Lab_ref[:, :, 0]
    astd = Lab_ref[:, :, 1]
    bstd = Lab_ref[:, :, 2]
    Lab_pro = CAM.RGBToLjg(Pro_image)
    Lsample = Lab_pro[:, :, 0]
    asample = Lab_pro[:, :, 1]
    bsample = Lab_pro[:, :, 2]
    CDw = np.zeros(LBP.shape)
    Np = LBP.size
    for pp in list_of_patterns:
        homo_patches = LBP == pp
        homo_patches = ndimage.morphology.binary_closing(homo_patches, SE)
        patch_index, num_patches = ndimage.measurements.label(homo_patches)
        for ii in range(1, num_patches):
            idx = np.where(patch_index == ii)
            if idx[0].size > min_num_pixels:
                temp = MU.deltaosa(Lstd[idx], astd[idx], bstd[idx], Lsample[idx], asample[idx], bsample[idx])
                if sq:
                    temp = np.power(temp, 2)
                if not np.isnan(temp):
                    CDw[idx] = temp
                    wp = (1. * idx[0].size) / Np
                    cd += wp * temp
                wpt += 1  # wp
    return CDw, cd



def conf03_deltaosa_patches(Ref_image, Pro_image, th=0.005, r=3., min_num_pixels = 4, sq = False, KLCH=np.array([1, 1, 1])):
    # SE = np.ones((np.int_(2 * r + 1), np.int_(2 * r + 1)))
    LBP, _ = CT.lbp(Ref_image, th=th, r=r)
    SE = np.ones((2, 2))
    list_of_patterns = np.unique(LBP)
    cd = 0.
    wpt = 0.
    Lab_ref = CAM.RGBToLjg(Ref_image)
    Lstd = Lab_ref[:, :, 0]
    astd = Lab_ref[:, :, 1]
    bstd = Lab_ref[:, :, 2]
    Lab_pro = CAM.RGBToLjg(Pro_image)
    Lsample = Lab_pro[:, :, 0]
    asample = Lab_pro[:, :, 1]
    bsample = Lab_pro[:, :, 2]
    CDw = np.zeros(LBP.shape)
    Np = LBP.size
    for pp in list_of_patterns:
        homo_patches = LBP == pp
        homo_patches = ndimage.morphology.binary_closing(homo_patches, SE)
        patch_index, num_patches = ndimage.measurements.label(homo_patches)
        for ii in range(1, num_patches):
            idx = np.where(patch_index == ii)
            if idx[0].size > min_num_pixels:
                temp = MU.deltaosa(Lstd[idx], astd[idx], bstd[idx], Lsample[idx], asample[idx], bsample[idx])
                if sq:
                    temp = np.power(temp, 2)
                if not np.isnan(temp):
                    CDw[idx] = temp
                    wp = (1. * idx[0].size) / Np
                    cd += wp * temp
                wpt += 1  # wp
    return CDw, cd



def conf04_deltaosa_patches(Ref_image, Pro_image, th=0.01, r=1., min_num_pixels = 4, sq = False, KLCH=np.array([1, 1, 1])):
    # SE = np.ones((np.int_(2 * r + 1), np.int_(2 * r + 1)))
    LBP, _ = CT.lbp(Ref_image, th=th, r=r)
    SE = np.ones((2, 2))
    list_of_patterns = np.unique(LBP)
    cd = 0.
    wpt = 0.
    Lab_ref = CAM.RGBToLjg(Ref_image)
    Lstd = Lab_ref[:, :, 0]
    astd = Lab_ref[:, :, 1]
    bstd = Lab_ref[:, :, 2]
    Lab_pro = CAM.RGBToLjg(Pro_image)
    Lsample = Lab_pro[:, :, 0]
    asample = Lab_pro[:, :, 1]
    bsample = Lab_pro[:, :, 2]
    CDw = np.zeros(LBP.shape)
    Np = LBP.size
    for pp in list_of_patterns:
        homo_patches = LBP == pp
        homo_patches = ndimage.morphology.binary_closing(homo_patches, SE)
        patch_index, num_patches = ndimage.measurements.label(homo_patches)
        for ii in range(1, num_patches):
            idx = np.where(patch_index == ii)
            if idx[0].size > min_num_pixels:
                temp = MU.deltaosa(Lstd[idx], astd[idx], bstd[idx], Lsample[idx], asample[idx], bsample[idx])
                if sq:
                    temp = np.power(temp, 2)
                if not np.isnan(temp):
                    CDw[idx] = temp
                    wp = (1. * idx[0].size) / Np
                    cd += wp * temp
                wpt += 1  # wp
    return CDw, cd


def conf05_deltaosa_patches(Ref_image, Pro_image, th=0.01, r=1.5, min_num_pixels = 4, sq = False, KLCH=np.array([1, 1, 1])):
    # SE = np.ones((np.int_(2 * r + 1), np.int_(2 * r + 1)))
    LBP, _ = CT.lbp(Ref_image, th=th, r=r)
    SE = np.ones((2, 2))
    list_of_patterns = np.unique(LBP)
    cd = 0.
    wpt = 0.
    Lab_ref = CAM.RGBToLjg(Ref_image)
    Lstd = Lab_ref[:, :, 0]
    astd = Lab_ref[:, :, 1]
    bstd = Lab_ref[:, :, 2]
    Lab_pro = CAM.RGBToLjg(Pro_image)
    Lsample = Lab_pro[:, :, 0]
    asample = Lab_pro[:, :, 1]
    bsample = Lab_pro[:, :, 2]
    CDw = np.zeros(LBP.shape)
    Np = LBP.size
    for pp in list_of_patterns:
        homo_patches = LBP == pp
        homo_patches = ndimage.morphology.binary_closing(homo_patches, SE)
        patch_index, num_patches = ndimage.measurements.label(homo_patches)
        for ii in range(1, num_patches):
            idx = np.where(patch_index == ii)
            if idx[0].size > min_num_pixels:
                temp = MU.deltaosa(Lstd[idx], astd[idx], bstd[idx], Lsample[idx], asample[idx], bsample[idx])
                if sq:
                    temp = np.power(temp, 2)
                if not np.isnan(temp):
                    CDw[idx] = temp
                    wp = (1. * idx[0].size) / Np
                    cd += wp * temp
                wpt += 1  # wp
    return CDw, cd


def conf06_deltaosa_patches(Ref_image, Pro_image, th=0.01, r=2., min_num_pixels = 4, sq = False, KLCH=np.array([1, 1, 1])):
    # SE = np.ones((np.int_(2 * r + 1), np.int_(2 * r + 1)))
    LBP, _ = CT.lbp(Ref_image, th=th, r=r)
    SE = np.ones((2, 2))
    list_of_patterns = np.unique(LBP)
    cd = 0.
    wpt = 0.
    Lab_ref = CAM.RGBToLjg(Ref_image)
    Lstd = Lab_ref[:, :, 0]
    astd = Lab_ref[:, :, 1]
    bstd = Lab_ref[:, :, 2]
    Lab_pro = CAM.RGBToLjg(Pro_image)
    Lsample = Lab_pro[:, :, 0]
    asample = Lab_pro[:, :, 1]
    bsample = Lab_pro[:, :, 2]
    CDw = np.zeros(LBP.shape)
    Np = LBP.size
    for pp in list_of_patterns:
        homo_patches = LBP == pp
        homo_patches = ndimage.morphology.binary_closing(homo_patches, SE)
        patch_index, num_patches = ndimage.measurements.label(homo_patches)
        for ii in range(1, num_patches):
            idx = np.where(patch_index == ii)
            if idx[0].size > min_num_pixels:
                temp = MU.deltaosa(Lstd[idx], astd[idx], bstd[idx], Lsample[idx], asample[idx], bsample[idx])
                if sq:
                    temp = np.power(temp, 2)
                if not np.isnan(temp):
                    CDw[idx] = temp
                    wp = (1. * idx[0].size) / Np
                    cd += wp * temp
                wpt += 1  # wp
    return CDw, cd



def conf07_deltaosa_patches(Ref_image, Pro_image, th=0.01, r=3., min_num_pixels = 4, sq = False, KLCH=np.array([1, 1, 1])):
    # SE = np.ones((np.int_(2 * r + 1), np.int_(2 * r + 1)))
    LBP, _ = CT.lbp(Ref_image, th=th, r=r)
    SE = np.ones((2, 2))
    list_of_patterns = np.unique(LBP)
    cd = 0.
    wpt = 0.
    Lab_ref = CAM.RGBToLjg(Ref_image)
    Lstd = Lab_ref[:, :, 0]
    astd = Lab_ref[:, :, 1]
    bstd = Lab_ref[:, :, 2]
    Lab_pro = CAM.RGBToLjg(Pro_image)
    Lsample = Lab_pro[:, :, 0]
    asample = Lab_pro[:, :, 1]
    bsample = Lab_pro[:, :, 2]
    CDw = np.zeros(LBP.shape)
    Np = LBP.size
    for pp in list_of_patterns:
        homo_patches = LBP == pp
        homo_patches = ndimage.morphology.binary_closing(homo_patches, SE)
        patch_index, num_patches = ndimage.measurements.label(homo_patches)
        for ii in range(1, num_patches):
            idx = np.where(patch_index == ii)
            if idx[0].size > min_num_pixels:
                temp = MU.deltaosa(Lstd[idx], astd[idx], bstd[idx], Lsample[idx], asample[idx], bsample[idx])
                if sq:
                    temp = np.power(temp, 2)
                if not np.isnan(temp):
                    CDw[idx] = temp
                    wp = (1. * idx[0].size) / Np
                    cd += wp * temp
                wpt += 1  # wp
    return CDw, cd


def conf08_deltaosa_patches(Ref_image, Pro_image, th=0.015, r=1., min_num_pixels = 4, sq = False, KLCH=np.array([1, 1, 1])):
    # SE = np.ones((np.int_(2 * r + 1), np.int_(2 * r + 1)))
    LBP, _ = CT.lbp(Ref_image, th=th, r=r)
    SE = np.ones((2, 2))
    list_of_patterns = np.unique(LBP)
    cd = 0.
    wpt = 0.
    Lab_ref = CAM.RGBToLjg(Ref_image)
    Lstd = Lab_ref[:, :, 0]
    astd = Lab_ref[:, :, 1]
    bstd = Lab_ref[:, :, 2]
    Lab_pro = CAM.RGBToLjg(Pro_image)
    Lsample = Lab_pro[:, :, 0]
    asample = Lab_pro[:, :, 1]
    bsample = Lab_pro[:, :, 2]
    CDw = np.zeros(LBP.shape)
    Np = LBP.size
    for pp in list_of_patterns:
        homo_patches = LBP == pp
        homo_patches = ndimage.morphology.binary_closing(homo_patches, SE)
        patch_index, num_patches = ndimage.measurements.label(homo_patches)
        for ii in range(1, num_patches):
            idx = np.where(patch_index == ii)
            if idx[0].size > min_num_pixels:
                temp = MU.deltaosa(Lstd[idx], astd[idx], bstd[idx], Lsample[idx], asample[idx], bsample[idx])
                if sq:
                    temp = np.power(temp, 2)
                if not np.isnan(temp):
                    CDw[idx] = temp
                    wp = (1. * idx[0].size) / Np
                    cd += wp * temp
                wpt += 1  # wp
    return CDw, cd


def conf09_deltaosa_patches(Ref_image, Pro_image, th=0.015, r=1.5, min_num_pixels = 4, sq = False, KLCH=np.array([1, 1, 1])):
    # SE = np.ones((np.int_(2 * r + 1), np.int_(2 * r + 1)))
    LBP, _ = CT.lbp(Ref_image, th=th, r=r)
    SE = np.ones((2, 2))
    list_of_patterns = np.unique(LBP)
    cd = 0.
    wpt = 0.
    Lab_ref = CAM.RGBToLjg(Ref_image)
    Lstd = Lab_ref[:, :, 0]
    astd = Lab_ref[:, :, 1]
    bstd = Lab_ref[:, :, 2]
    Lab_pro = CAM.RGBToLjg(Pro_image)
    Lsample = Lab_pro[:, :, 0]
    asample = Lab_pro[:, :, 1]
    bsample = Lab_pro[:, :, 2]
    CDw = np.zeros(LBP.shape)
    Np = LBP.size
    for pp in list_of_patterns:
        homo_patches = LBP == pp
        homo_patches = ndimage.morphology.binary_closing(homo_patches, SE)
        patch_index, num_patches = ndimage.measurements.label(homo_patches)
        for ii in range(1, num_patches):
            idx = np.where(patch_index == ii)
            if idx[0].size > min_num_pixels:
                temp = MU.deltaosa(Lstd[idx], astd[idx], bstd[idx], Lsample[idx], asample[idx], bsample[idx])
                if sq:
                    temp = np.power(temp, 2)
                if not np.isnan(temp):
                    CDw[idx] = temp
                    wp = (1. * idx[0].size) / Np
                    cd += wp * temp
                wpt += 1  # wp
    return CDw, cd



def conf10_deltaosa_patches(Ref_image, Pro_image, th=0.015, r=2., min_num_pixels = 4, sq = False, KLCH=np.array([1, 1, 1])):
    # SE = np.ones((np.int_(2 * r + 1), np.int_(2 * r + 1)))
    LBP, _ = CT.lbp(Ref_image, th=th, r=r)
    SE = np.ones((2, 2))
    list_of_patterns = np.unique(LBP)
    cd = 0.
    wpt = 0.
    Lab_ref = CAM.RGBToLjg(Ref_image)
    Lstd = Lab_ref[:, :, 0]
    astd = Lab_ref[:, :, 1]
    bstd = Lab_ref[:, :, 2]
    Lab_pro = CAM.RGBToLjg(Pro_image)
    Lsample = Lab_pro[:, :, 0]
    asample = Lab_pro[:, :, 1]
    bsample = Lab_pro[:, :, 2]
    CDw = np.zeros(LBP.shape)
    Np = LBP.size
    for pp in list_of_patterns:
        homo_patches = LBP == pp
        homo_patches = ndimage.morphology.binary_closing(homo_patches, SE)
        patch_index, num_patches = ndimage.measurements.label(homo_patches)
        for ii in range(1, num_patches):
            idx = np.where(patch_index == ii)
            if idx[0].size > min_num_pixels:
                temp = MU.deltaosa(Lstd[idx], astd[idx], bstd[idx], Lsample[idx], asample[idx], bsample[idx])
                if sq:
                    temp = np.power(temp, 2)
                if not np.isnan(temp):
                    CDw[idx] = temp
                    wp = (1. * idx[0].size) / Np
                    cd += wp * temp
                wpt += 1  # wp
    return CDw, cd



def conf11_deltaosa_patches(Ref_image, Pro_image, th=0.015, r=3., min_num_pixels = 4, sq = False, KLCH=np.array([1, 1, 1])):
    # SE = np.ones((np.int_(2 * r + 1), np.int_(2 * r + 1)))
    LBP, _ = CT.lbp(Ref_image, th=th, r=r)
    SE = np.ones((2, 2))
    list_of_patterns = np.unique(LBP)
    cd = 0.
    wpt = 0.
    Lab_ref = CAM.RGBToLjg(Ref_image)
    Lstd = Lab_ref[:, :, 0]
    astd = Lab_ref[:, :, 1]
    bstd = Lab_ref[:, :, 2]
    Lab_pro = CAM.RGBToLjg(Pro_image)
    Lsample = Lab_pro[:, :, 0]
    asample = Lab_pro[:, :, 1]
    bsample = Lab_pro[:, :, 2]
    CDw = np.zeros(LBP.shape)
    Np = LBP.size
    for pp in list_of_patterns:
        homo_patches = LBP == pp
        homo_patches = ndimage.morphology.binary_closing(homo_patches, SE)
        patch_index, num_patches = ndimage.measurements.label(homo_patches)
        for ii in range(1, num_patches):
            idx = np.where(patch_index == ii)
            if idx[0].size > min_num_pixels:
                temp = MU.deltaosa(Lstd[idx], astd[idx], bstd[idx], Lsample[idx], asample[idx], bsample[idx])
                if sq:
                    temp = np.power(temp, 2)
                if not np.isnan(temp):
                    CDw[idx] = temp
                    wp = (1. * idx[0].size) / Np
                    cd += wp * temp
                wpt += 1  # wp
    return CDw, cd


def conf12_deltaosa_patches(Ref_image, Pro_image, th=0.02, r=1., min_num_pixels = 4, sq = False, KLCH=np.array([1, 1, 1])):
    # SE = np.ones((np.int_(2 * r + 1), np.int_(2 * r + 1)))
    LBP, _ = CT.lbp(Ref_image, th=th, r=r)
    SE = np.ones((2, 2))
    list_of_patterns = np.unique(LBP)
    cd = 0.
    wpt = 0.
    Lab_ref = CAM.RGBToLjg(Ref_image)
    Lstd = Lab_ref[:, :, 0]
    astd = Lab_ref[:, :, 1]
    bstd = Lab_ref[:, :, 2]
    Lab_pro = CAM.RGBToLjg(Pro_image)
    Lsample = Lab_pro[:, :, 0]
    asample = Lab_pro[:, :, 1]
    bsample = Lab_pro[:, :, 2]
    CDw = np.zeros(LBP.shape)
    Np = LBP.size
    for pp in list_of_patterns:
        homo_patches = LBP == pp
        homo_patches = ndimage.morphology.binary_closing(homo_patches, SE)
        patch_index, num_patches = ndimage.measurements.label(homo_patches)
        for ii in range(1, num_patches):
            idx = np.where(patch_index == ii)
            if idx[0].size > min_num_pixels:
                temp = MU.deltaosa(Lstd[idx], astd[idx], bstd[idx], Lsample[idx], asample[idx], bsample[idx])
                if sq:
                    temp = np.power(temp, 2)
                if not np.isnan(temp):
                    CDw[idx] = temp
                    wp = (1. * idx[0].size) / Np
                    cd += wp * temp
                wpt += 1  # wp
    return CDw, cd


def conf13_deltaosa_patches(Ref_image, Pro_image, th=0.02, r=1.5, min_num_pixels = 4, sq = False, KLCH=np.array([1, 1, 1])):
    # SE = np.ones((np.int_(2 * r + 1), np.int_(2 * r + 1)))
    LBP, _ = CT.lbp(Ref_image, th=th, r=r)
    SE = np.ones((2, 2))
    list_of_patterns = np.unique(LBP)
    cd = 0.
    wpt = 0.
    Lab_ref = CAM.RGBToLjg(Ref_image)
    Lstd = Lab_ref[:, :, 0]
    astd = Lab_ref[:, :, 1]
    bstd = Lab_ref[:, :, 2]
    Lab_pro = CAM.RGBToLjg(Pro_image)
    Lsample = Lab_pro[:, :, 0]
    asample = Lab_pro[:, :, 1]
    bsample = Lab_pro[:, :, 2]
    CDw = np.zeros(LBP.shape)
    Np = LBP.size
    for pp in list_of_patterns:
        homo_patches = LBP == pp
        homo_patches = ndimage.morphology.binary_closing(homo_patches, SE)
        patch_index, num_patches = ndimage.measurements.label(homo_patches)
        for ii in range(1, num_patches):
            idx = np.where(patch_index == ii)
            if idx[0].size > min_num_pixels:
                temp = MU.deltaosa(Lstd[idx], astd[idx], bstd[idx], Lsample[idx], asample[idx], bsample[idx])
                if sq:
                    temp = np.power(temp, 2)
                if not np.isnan(temp):
                    CDw[idx] = temp
                    wp = (1. * idx[0].size) / Np
                    cd += wp * temp
                wpt += 1  # wp
    return CDw, cd



def conf14_deltaosa_patches(Ref_image, Pro_image, th=0.02, r=2., min_num_pixels = 4, sq = False, KLCH=np.array([1, 1, 1])):
    # SE = np.ones((np.int_(2 * r + 1), np.int_(2 * r + 1)))
    LBP, _ = CT.lbp(Ref_image, th=th, r=r)
    SE = np.ones((2, 2))
    list_of_patterns = np.unique(LBP)
    cd = 0.
    wpt = 0.
    Lab_ref = CAM.RGBToLjg(Ref_image)
    Lstd = Lab_ref[:, :, 0]
    astd = Lab_ref[:, :, 1]
    bstd = Lab_ref[:, :, 2]
    Lab_pro = CAM.RGBToLjg(Pro_image)
    Lsample = Lab_pro[:, :, 0]
    asample = Lab_pro[:, :, 1]
    bsample = Lab_pro[:, :, 2]
    CDw = np.zeros(LBP.shape)
    Np = LBP.size
    for pp in list_of_patterns:
        homo_patches = LBP == pp
        homo_patches = ndimage.morphology.binary_closing(homo_patches, SE)
        patch_index, num_patches = ndimage.measurements.label(homo_patches)
        for ii in range(1, num_patches):
            idx = np.where(patch_index == ii)
            if idx[0].size > min_num_pixels:
                temp = MU.deltaosa(Lstd[idx], astd[idx], bstd[idx], Lsample[idx], asample[idx], bsample[idx])
                if sq:
                    temp = np.power(temp, 2)
                if not np.isnan(temp):
                    CDw[idx] = temp
                    wp = (1. * idx[0].size) / Np
                    cd += wp * temp
                wpt += 1  # wp
    return CDw, cd



def conf15_deltaosa_patches(Ref_image, Pro_image, th=0.02, r=3., min_num_pixels = 4, sq = False, KLCH=np.array([1, 1, 1])):
    # SE = np.ones((np.int_(2 * r + 1), np.int_(2 * r + 1)))
    LBP, _ = CT.lbp(Ref_image, th=th, r=r)
    SE = np.ones((2, 2))
    list_of_patterns = np.unique(LBP)
    cd = 0.
    wpt = 0.
    Lab_ref = CAM.RGBToLjg(Ref_image)
    Lstd = Lab_ref[:, :, 0]
    astd = Lab_ref[:, :, 1]
    bstd = Lab_ref[:, :, 2]
    Lab_pro = CAM.RGBToLjg(Pro_image)
    Lsample = Lab_pro[:, :, 0]
    asample = Lab_pro[:, :, 1]
    bsample = Lab_pro[:, :, 2]
    CDw = np.zeros(LBP.shape)
    Np = LBP.size
    for pp in list_of_patterns:
        homo_patches = LBP == pp
        homo_patches = ndimage.morphology.binary_closing(homo_patches, SE)
        patch_index, num_patches = ndimage.measurements.label(homo_patches)
        for ii in range(1, num_patches):
            idx = np.where(patch_index == ii)
            if idx[0].size > min_num_pixels:
                temp = MU.deltaosa(Lstd[idx], astd[idx], bstd[idx], Lsample[idx], asample[idx], bsample[idx])
                if sq:
                    temp = np.power(temp, 2)
                if not np.isnan(temp):
                    CDw[idx] = temp
                    wp = (1. * idx[0].size) / Np
                    cd += wp * temp
                wpt += 1  # wp
    return CDw, cd



def conf16_deltaosa_patches(Ref_image, Pro_image, th=0.025, r=1., min_num_pixels = 4, sq = False, KLCH=np.array([1, 1, 1])):
    # SE = np.ones((np.int_(2 * r + 1), np.int_(2 * r + 1)))
    LBP, _ = CT.lbp(Ref_image, th=th, r=r)
    SE = np.ones((2, 2))
    list_of_patterns = np.unique(LBP)
    cd = 0.
    wpt = 0.
    Lab_ref = CAM.RGBToLjg(Ref_image)
    Lstd = Lab_ref[:, :, 0]
    astd = Lab_ref[:, :, 1]
    bstd = Lab_ref[:, :, 2]
    Lab_pro = CAM.RGBToLjg(Pro_image)
    Lsample = Lab_pro[:, :, 0]
    asample = Lab_pro[:, :, 1]
    bsample = Lab_pro[:, :, 2]
    CDw = np.zeros(LBP.shape)
    Np = LBP.size
    for pp in list_of_patterns:
        homo_patches = LBP == pp
        homo_patches = ndimage.morphology.binary_closing(homo_patches, SE)
        patch_index, num_patches = ndimage.measurements.label(homo_patches)
        for ii in range(1, num_patches):
            idx = np.where(patch_index == ii)
            if idx[0].size > min_num_pixels:
                temp = MU.deltaosa(Lstd[idx], astd[idx], bstd[idx], Lsample[idx], asample[idx], bsample[idx])
                if sq:
                    temp = np.power(temp, 2)
                if not np.isnan(temp):
                    CDw[idx] = temp
                    wp = (1. * idx[0].size) / Np
                    cd += wp * temp
                wpt += 1  # wp
    return CDw, cd


def conf17_deltaosa_patches(Ref_image, Pro_image, th=0.025, r=1.5, min_num_pixels = 4, sq = False, KLCH=np.array([1, 1, 1])):
    # SE = np.ones((np.int_(2 * r + 1), np.int_(2 * r + 1)))
    LBP, _ = CT.lbp(Ref_image, th=th, r=r)
    SE = np.ones((2, 2))
    list_of_patterns = np.unique(LBP)
    cd = 0.
    wpt = 0.
    Lab_ref = CAM.RGBToLjg(Ref_image)
    Lstd = Lab_ref[:, :, 0]
    astd = Lab_ref[:, :, 1]
    bstd = Lab_ref[:, :, 2]
    Lab_pro = CAM.RGBToLjg(Pro_image)
    Lsample = Lab_pro[:, :, 0]
    asample = Lab_pro[:, :, 1]
    bsample = Lab_pro[:, :, 2]
    CDw = np.zeros(LBP.shape)
    Np = LBP.size
    for pp in list_of_patterns:
        homo_patches = LBP == pp
        homo_patches = ndimage.morphology.binary_closing(homo_patches, SE)
        patch_index, num_patches = ndimage.measurements.label(homo_patches)
        for ii in range(1, num_patches):
            idx = np.where(patch_index == ii)
            if idx[0].size > min_num_pixels:
                temp = MU.deltaosa(Lstd[idx], astd[idx], bstd[idx], Lsample[idx], asample[idx], bsample[idx])
                if sq:
                    temp = np.power(temp, 2)
                if not np.isnan(temp):
                    CDw[idx] = temp
                    wp = (1. * idx[0].size) / Np
                    cd += wp * temp
                wpt += 1  # wp
    return CDw, cd



def conf18_deltaosa_patches(Ref_image, Pro_image, th=0.025, r=2., min_num_pixels = 4, sq = False, KLCH=np.array([1, 1, 1])):
    # SE = np.ones((np.int_(2 * r + 1), np.int_(2 * r + 1)))
    LBP, _ = CT.lbp(Ref_image, th=th, r=r)
    SE = np.ones((2, 2))
    list_of_patterns = np.unique(LBP)
    cd = 0.
    wpt = 0.
    Lab_ref = CAM.RGBToLjg(Ref_image)
    Lstd = Lab_ref[:, :, 0]
    astd = Lab_ref[:, :, 1]
    bstd = Lab_ref[:, :, 2]
    Lab_pro = CAM.RGBToLjg(Pro_image)
    Lsample = Lab_pro[:, :, 0]
    asample = Lab_pro[:, :, 1]
    bsample = Lab_pro[:, :, 2]
    CDw = np.zeros(LBP.shape)
    Np = LBP.size
    for pp in list_of_patterns:
        homo_patches = LBP == pp
        homo_patches = ndimage.morphology.binary_closing(homo_patches, SE)
        patch_index, num_patches = ndimage.measurements.label(homo_patches)
        for ii in range(1, num_patches):
            idx = np.where(patch_index == ii)
            if idx[0].size > min_num_pixels:
                temp = MU.deltaosa(Lstd[idx], astd[idx], bstd[idx], Lsample[idx], asample[idx], bsample[idx])
                if sq:
                    temp = np.power(temp, 2)
                if not np.isnan(temp):
                    CDw[idx] = temp
                    wp = (1. * idx[0].size) / Np
                    cd += wp * temp
                wpt += 1  # wp
    return CDw, cd



def conf19_deltaosa_patches(Ref_image, Pro_image, th=0.025, r=3., min_num_pixels = 4, sq = False, KLCH=np.array([1, 1, 1])):
    # SE = np.ones((np.int_(2 * r + 1), np.int_(2 * r + 1)))
    LBP, _ = CT.lbp(Ref_image, th=th, r=r)
    SE = np.ones((2, 2))
    list_of_patterns = np.unique(LBP)
    cd = 0.
    wpt = 0.
    Lab_ref = CAM.RGBToLjg(Ref_image)
    Lstd = Lab_ref[:, :, 0]
    astd = Lab_ref[:, :, 1]
    bstd = Lab_ref[:, :, 2]
    Lab_pro = CAM.RGBToLjg(Pro_image)
    Lsample = Lab_pro[:, :, 0]
    asample = Lab_pro[:, :, 1]
    bsample = Lab_pro[:, :, 2]
    CDw = np.zeros(LBP.shape)
    Np = LBP.size
    for pp in list_of_patterns:
        homo_patches = LBP == pp
        homo_patches = ndimage.morphology.binary_closing(homo_patches, SE)
        patch_index, num_patches = ndimage.measurements.label(homo_patches)
        for ii in range(1, num_patches):
            idx = np.where(patch_index == ii)
            if idx[0].size > min_num_pixels:
                temp = MU.deltaosa(Lstd[idx], astd[idx], bstd[idx], Lsample[idx], asample[idx], bsample[idx])
                if sq:
                    temp = np.power(temp, 2)
                if not np.isnan(temp):
                    CDw[idx] = temp
                    wp = (1. * idx[0].size) / Np
                    cd += wp * temp
                wpt += 1  # wp
    return CDw, cd


def conf20_deltaosa_patches(Ref_image, Pro_image, th=0.03, r=1., min_num_pixels = 4, sq = False, KLCH=np.array([1, 1, 1])):
    # SE = np.ones((np.int_(2 * r + 1), np.int_(2 * r + 1)))
    LBP, _ = CT.lbp(Ref_image, th=th, r=r)
    SE = np.ones((2, 2))
    list_of_patterns = np.unique(LBP)
    cd = 0.
    wpt = 0.
    Lab_ref = CAM.RGBToLjg(Ref_image)
    Lstd = Lab_ref[:, :, 0]
    astd = Lab_ref[:, :, 1]
    bstd = Lab_ref[:, :, 2]
    Lab_pro = CAM.RGBToLjg(Pro_image)
    Lsample = Lab_pro[:, :, 0]
    asample = Lab_pro[:, :, 1]
    bsample = Lab_pro[:, :, 2]
    CDw = np.zeros(LBP.shape)
    Np = LBP.size
    for pp in list_of_patterns:
        homo_patches = LBP == pp
        homo_patches = ndimage.morphology.binary_closing(homo_patches, SE)
        patch_index, num_patches = ndimage.measurements.label(homo_patches)
        for ii in range(1, num_patches):
            idx = np.where(patch_index == ii)
            if idx[0].size > min_num_pixels:
                temp = MU.deltaosa(Lstd[idx], astd[idx], bstd[idx], Lsample[idx], asample[idx], bsample[idx])
                if sq:
                    temp = np.power(temp, 2)
                if not np.isnan(temp):
                    CDw[idx] = temp
                    wp = (1. * idx[0].size) / Np
                    cd += wp * temp
                wpt += 1  # wp
    return CDw, cd


def conf21_deltaosa_patches(Ref_image, Pro_image, th=0.03, r=1.5, min_num_pixels = 4, sq = False, KLCH=np.array([1, 1, 1])):
    # SE = np.ones((np.int_(2 * r + 1), np.int_(2 * r + 1)))
    LBP, _ = CT.lbp(Ref_image, th=th, r=r)
    SE = np.ones((2, 2))
    list_of_patterns = np.unique(LBP)
    cd = 0.
    wpt = 0.
    Lab_ref = CAM.RGBToLjg(Ref_image)
    Lstd = Lab_ref[:, :, 0]
    astd = Lab_ref[:, :, 1]
    bstd = Lab_ref[:, :, 2]
    Lab_pro = CAM.RGBToLjg(Pro_image)
    Lsample = Lab_pro[:, :, 0]
    asample = Lab_pro[:, :, 1]
    bsample = Lab_pro[:, :, 2]
    CDw = np.zeros(LBP.shape)
    Np = LBP.size
    for pp in list_of_patterns:
        homo_patches = LBP == pp
        homo_patches = ndimage.morphology.binary_closing(homo_patches, SE)
        patch_index, num_patches = ndimage.measurements.label(homo_patches)
        for ii in range(1, num_patches):
            idx = np.where(patch_index == ii)
            if idx[0].size > min_num_pixels:
                temp = MU.deltaosa(Lstd[idx], astd[idx], bstd[idx], Lsample[idx], asample[idx], bsample[idx])
                if sq:
                    temp = np.power(temp, 2)
                if not np.isnan(temp):
                    CDw[idx] = temp
                    wp = (1. * idx[0].size) / Np
                    cd += wp * temp
                wpt += 1  # wp
    return CDw, cd



def conf22_deltaosa_patches(Ref_image, Pro_image, th=0.03, r=2., min_num_pixels = 4, sq = False, KLCH=np.array([1, 1, 1])):
    # SE = np.ones((np.int_(2 * r + 1), np.int_(2 * r + 1)))
    LBP, _ = CT.lbp(Ref_image, th=th, r=r)
    SE = np.ones((2, 2))
    list_of_patterns = np.unique(LBP)
    cd = 0.
    wpt = 0.
    Lab_ref = CAM.RGBToLjg(Ref_image)
    Lstd = Lab_ref[:, :, 0]
    astd = Lab_ref[:, :, 1]
    bstd = Lab_ref[:, :, 2]
    Lab_pro = CAM.RGBToLjg(Pro_image)
    Lsample = Lab_pro[:, :, 0]
    asample = Lab_pro[:, :, 1]
    bsample = Lab_pro[:, :, 2]
    CDw = np.zeros(LBP.shape)
    Np = LBP.size
    for pp in list_of_patterns:
        homo_patches = LBP == pp
        homo_patches = ndimage.morphology.binary_closing(homo_patches, SE)
        patch_index, num_patches = ndimage.measurements.label(homo_patches)
        for ii in range(1, num_patches):
            idx = np.where(patch_index == ii)
            if idx[0].size > min_num_pixels:
                temp = MU.deltaosa(Lstd[idx], astd[idx], bstd[idx], Lsample[idx], asample[idx], bsample[idx])
                if sq:
                    temp = np.power(temp, 2)
                if not np.isnan(temp):
                    CDw[idx] = temp
                    wp = (1. * idx[0].size) / Np
                    cd += wp * temp
                wpt += 1  # wp
    return CDw, cd



def conf23_deltaosa_patches(Ref_image, Pro_image, th=0.03, r=3., min_num_pixels = 4, sq = False, KLCH=np.array([1, 1, 1])):
    # SE = np.ones((np.int_(2 * r + 1), np.int_(2 * r + 1)))
    LBP, _ = CT.lbp(Ref_image, th=th, r=r)
    SE = np.ones((2, 2))
    list_of_patterns = np.unique(LBP)
    cd = 0.
    wpt = 0.
    Lab_ref = CAM.RGBToLjg(Ref_image)
    Lstd = Lab_ref[:, :, 0]
    astd = Lab_ref[:, :, 1]
    bstd = Lab_ref[:, :, 2]
    Lab_pro = CAM.RGBToLjg(Pro_image)
    Lsample = Lab_pro[:, :, 0]
    asample = Lab_pro[:, :, 1]
    bsample = Lab_pro[:, :, 2]
    CDw = np.zeros(LBP.shape)
    Np = LBP.size
    for pp in list_of_patterns:
        homo_patches = LBP == pp
        homo_patches = ndimage.morphology.binary_closing(homo_patches, SE)
        patch_index, num_patches = ndimage.measurements.label(homo_patches)
        for ii in range(1, num_patches):
            idx = np.where(patch_index == ii)
            if idx[0].size > min_num_pixels:
                temp = MU.deltaosa(Lstd[idx], astd[idx], bstd[idx], Lsample[idx], asample[idx], bsample[idx])
                if sq:
                    temp = np.power(temp, 2)
                if not np.isnan(temp):
                    CDw[idx] = temp
                    wp = (1. * idx[0].size) / Np
                    cd += wp * temp
                wpt += 1  # wp
    return CDw, cd



def conf24_deltaosa_patches(Ref_image, Pro_image, th=0.035, r=1., min_num_pixels = 4, sq = False, KLCH=np.array([1, 1, 1])):
    # SE = np.ones((np.int_(2 * r + 1), np.int_(2 * r + 1)))
    LBP, _ = CT.lbp(Ref_image, th=th, r=r)
    SE = np.ones((2, 2))
    list_of_patterns = np.unique(LBP)
    cd = 0.
    wpt = 0.
    Lab_ref = CAM.RGBToLjg(Ref_image)
    Lstd = Lab_ref[:, :, 0]
    astd = Lab_ref[:, :, 1]
    bstd = Lab_ref[:, :, 2]
    Lab_pro = CAM.RGBToLjg(Pro_image)
    Lsample = Lab_pro[:, :, 0]
    asample = Lab_pro[:, :, 1]
    bsample = Lab_pro[:, :, 2]
    CDw = np.zeros(LBP.shape)
    Np = LBP.size
    for pp in list_of_patterns:
        homo_patches = LBP == pp
        homo_patches = ndimage.morphology.binary_closing(homo_patches, SE)
        patch_index, num_patches = ndimage.measurements.label(homo_patches)
        for ii in range(1, num_patches):
            idx = np.where(patch_index == ii)
            if idx[0].size > min_num_pixels:
                temp = MU.deltaosa(Lstd[idx], astd[idx], bstd[idx], Lsample[idx], asample[idx], bsample[idx])
                if sq:
                    temp = np.power(temp, 2)
                if not np.isnan(temp):
                    CDw[idx] = temp
                    wp = (1. * idx[0].size) / Np
                    cd += wp * temp
                wpt += 1  # wp
    return CDw, cd


def conf25_deltaosa_patches(Ref_image, Pro_image, th=0.035, r=1.5, min_num_pixels = 4, sq = False, KLCH=np.array([1, 1, 1])):
    # SE = np.ones((np.int_(2 * r + 1), np.int_(2 * r + 1)))
    LBP, _ = CT.lbp(Ref_image, th=th, r=r)
    SE = np.ones((2, 2))
    list_of_patterns = np.unique(LBP)
    cd = 0.
    wpt = 0.
    Lab_ref = CAM.RGBToLjg(Ref_image)
    Lstd = Lab_ref[:, :, 0]
    astd = Lab_ref[:, :, 1]
    bstd = Lab_ref[:, :, 2]
    Lab_pro = CAM.RGBToLjg(Pro_image)
    Lsample = Lab_pro[:, :, 0]
    asample = Lab_pro[:, :, 1]
    bsample = Lab_pro[:, :, 2]
    CDw = np.zeros(LBP.shape)
    Np = LBP.size
    for pp in list_of_patterns:
        homo_patches = LBP == pp
        homo_patches = ndimage.morphology.binary_closing(homo_patches, SE)
        patch_index, num_patches = ndimage.measurements.label(homo_patches)
        for ii in range(1, num_patches):
            idx = np.where(patch_index == ii)
            if idx[0].size > min_num_pixels:
                temp = MU.deltaosa(Lstd[idx], astd[idx], bstd[idx], Lsample[idx], asample[idx], bsample[idx])
                if sq:
                    temp = np.power(temp, 2)
                if not np.isnan(temp):
                    CDw[idx] = temp
                    wp = (1. * idx[0].size) / Np
                    cd += wp * temp
                wpt += 1  # wp
    return CDw, cd



def conf26_deltaosa_patches(Ref_image, Pro_image, th=0.035, r=2., min_num_pixels = 4, sq = False, KLCH=np.array([1, 1, 1])):
    # SE = np.ones((np.int_(2 * r + 1), np.int_(2 * r + 1)))
    LBP, _ = CT.lbp(Ref_image, th=th, r=r)
    SE = np.ones((2, 2))
    list_of_patterns = np.unique(LBP)
    cd = 0.
    wpt = 0.
    Lab_ref = CAM.RGBToLjg(Ref_image)
    Lstd = Lab_ref[:, :, 0]
    astd = Lab_ref[:, :, 1]
    bstd = Lab_ref[:, :, 2]
    Lab_pro = CAM.RGBToLjg(Pro_image)
    Lsample = Lab_pro[:, :, 0]
    asample = Lab_pro[:, :, 1]
    bsample = Lab_pro[:, :, 2]
    CDw = np.zeros(LBP.shape)
    Np = LBP.size
    for pp in list_of_patterns:
        homo_patches = LBP == pp
        homo_patches = ndimage.morphology.binary_closing(homo_patches, SE)
        patch_index, num_patches = ndimage.measurements.label(homo_patches)
        for ii in range(1, num_patches):
            idx = np.where(patch_index == ii)
            if idx[0].size > min_num_pixels:
                temp = MU.deltaosa(Lstd[idx], astd[idx], bstd[idx], Lsample[idx], asample[idx], bsample[idx])
                if sq:
                    temp = np.power(temp, 2)
                if not np.isnan(temp):
                    CDw[idx] = temp
                    wp = (1. * idx[0].size) / Np
                    cd += wp * temp
                wpt += 1  # wp
    return CDw, cd



def conf27_deltaosa_patches(Ref_image, Pro_image, th=0.035, r=3., min_num_pixels = 4, sq = False, KLCH=np.array([1, 1, 1])):
    # SE = np.ones((np.int_(2 * r + 1), np.int_(2 * r + 1)))
    LBP, _ = CT.lbp(Ref_image, th=th, r=r)
    SE = np.ones((2, 2))
    list_of_patterns = np.unique(LBP)
    cd = 0.
    wpt = 0.
    Lab_ref = CAM.RGBToLjg(Ref_image)
    Lstd = Lab_ref[:, :, 0]
    astd = Lab_ref[:, :, 1]
    bstd = Lab_ref[:, :, 2]
    Lab_pro = CAM.RGBToLjg(Pro_image)
    Lsample = Lab_pro[:, :, 0]
    asample = Lab_pro[:, :, 1]
    bsample = Lab_pro[:, :, 2]
    CDw = np.zeros(LBP.shape)
    Np = LBP.size
    for pp in list_of_patterns:
        homo_patches = LBP == pp
        homo_patches = ndimage.morphology.binary_closing(homo_patches, SE)
        patch_index, num_patches = ndimage.measurements.label(homo_patches)
        for ii in range(1, num_patches):
            idx = np.where(patch_index == ii)
            if idx[0].size > min_num_pixels:
                temp = MU.deltaosa(Lstd[idx], astd[idx], bstd[idx], Lsample[idx], asample[idx], bsample[idx])
                if sq:
                    temp = np.power(temp, 2)
                if not np.isnan(temp):
                    CDw[idx] = temp
                    wp = (1. * idx[0].size) / Np
                    cd += wp * temp
                wpt += 1  # wp
    return CDw, cd



def conf28_deltaosa_patches(Ref_image, Pro_image, th=0.04, r=1., min_num_pixels = 4, sq = False, KLCH=np.array([1, 1, 1])):
    # SE = np.ones((np.int_(2 * r + 1), np.int_(2 * r + 1)))
    LBP, _ = CT.lbp(Ref_image, th=th, r=r)
    SE = np.ones((2, 2))
    list_of_patterns = np.unique(LBP)
    cd = 0.
    wpt = 0.
    Lab_ref = CAM.RGBToLjg(Ref_image)
    Lstd = Lab_ref[:, :, 0]
    astd = Lab_ref[:, :, 1]
    bstd = Lab_ref[:, :, 2]
    Lab_pro = CAM.RGBToLjg(Pro_image)
    Lsample = Lab_pro[:, :, 0]
    asample = Lab_pro[:, :, 1]
    bsample = Lab_pro[:, :, 2]
    CDw = np.zeros(LBP.shape)
    Np = LBP.size
    for pp in list_of_patterns:
        homo_patches = LBP == pp
        homo_patches = ndimage.morphology.binary_closing(homo_patches, SE)
        patch_index, num_patches = ndimage.measurements.label(homo_patches)
        for ii in range(1, num_patches):
            idx = np.where(patch_index == ii)
            if idx[0].size > min_num_pixels:
                temp = MU.deltaosa(Lstd[idx], astd[idx], bstd[idx], Lsample[idx], asample[idx], bsample[idx])
                if sq:
                    temp = np.power(temp, 2)
                if not np.isnan(temp):
                    CDw[idx] = temp
                    wp = (1. * idx[0].size) / Np
                    cd += wp * temp
                wpt += 1  # wp
    return CDw, cd


def conf29_deltaosa_patches(Ref_image, Pro_image, th=0.04, r=1.5, min_num_pixels = 4, sq = False, KLCH=np.array([1, 1, 1])):
    # SE = np.ones((np.int_(2 * r + 1), np.int_(2 * r + 1)))
    LBP, _ = CT.lbp(Ref_image, th=th, r=r)
    SE = np.ones((2, 2))
    list_of_patterns = np.unique(LBP)
    cd = 0.
    wpt = 0.
    Lab_ref = CAM.RGBToLjg(Ref_image)
    Lstd = Lab_ref[:, :, 0]
    astd = Lab_ref[:, :, 1]
    bstd = Lab_ref[:, :, 2]
    Lab_pro = CAM.RGBToLjg(Pro_image)
    Lsample = Lab_pro[:, :, 0]
    asample = Lab_pro[:, :, 1]
    bsample = Lab_pro[:, :, 2]
    CDw = np.zeros(LBP.shape)
    Np = LBP.size
    for pp in list_of_patterns:
        homo_patches = LBP == pp
        homo_patches = ndimage.morphology.binary_closing(homo_patches, SE)
        patch_index, num_patches = ndimage.measurements.label(homo_patches)
        for ii in range(1, num_patches):
            idx = np.where(patch_index == ii)
            if idx[0].size > min_num_pixels:
                temp = MU.deltaosa(Lstd[idx], astd[idx], bstd[idx], Lsample[idx], asample[idx], bsample[idx])
                if sq:
                    temp = np.power(temp, 2)
                if not np.isnan(temp):
                    CDw[idx] = temp
                    wp = (1. * idx[0].size) / Np
                    cd += wp * temp
                wpt += 1  # wp
    return CDw, cd



def conf30_deltaosa_patches(Ref_image, Pro_image, th=0.04, r=2., min_num_pixels = 4, sq = False, KLCH=np.array([1, 1, 1])):
    # SE = np.ones((np.int_(2 * r + 1), np.int_(2 * r + 1)))
    LBP, _ = CT.lbp(Ref_image, th=th, r=r)
    SE = np.ones((2, 2))
    list_of_patterns = np.unique(LBP)
    cd = 0.
    wpt = 0.
    Lab_ref = CAM.RGBToLjg(Ref_image)
    Lstd = Lab_ref[:, :, 0]
    astd = Lab_ref[:, :, 1]
    bstd = Lab_ref[:, :, 2]
    Lab_pro = CAM.RGBToLjg(Pro_image)
    Lsample = Lab_pro[:, :, 0]
    asample = Lab_pro[:, :, 1]
    bsample = Lab_pro[:, :, 2]
    CDw = np.zeros(LBP.shape)
    Np = LBP.size
    for pp in list_of_patterns:
        homo_patches = LBP == pp
        homo_patches = ndimage.morphology.binary_closing(homo_patches, SE)
        patch_index, num_patches = ndimage.measurements.label(homo_patches)
        for ii in range(1, num_patches):
            idx = np.where(patch_index == ii)
            if idx[0].size > min_num_pixels:
                temp = MU.deltaosa(Lstd[idx], astd[idx], bstd[idx], Lsample[idx], asample[idx], bsample[idx])
                if sq:
                    temp = np.power(temp, 2)
                if not np.isnan(temp):
                    CDw[idx] = temp
                    wp = (1. * idx[0].size) / Np
                    cd += wp * temp
                wpt += 1  # wp
    return CDw, cd



def conf31_deltaosa_patches(Ref_image, Pro_image, th=0.04, r=3., min_num_pixels = 4, sq = False, KLCH=np.array([1, 1, 1])):
    # SE = np.ones((np.int_(2 * r + 1), np.int_(2 * r + 1)))
    LBP, _ = CT.lbp(Ref_image, th=th, r=r)
    SE = np.ones((2, 2))
    list_of_patterns = np.unique(LBP)
    cd = 0.
    wpt = 0.
    Lab_ref = CAM.RGBToLjg(Ref_image)
    Lstd = Lab_ref[:, :, 0]
    astd = Lab_ref[:, :, 1]
    bstd = Lab_ref[:, :, 2]
    Lab_pro = CAM.RGBToLjg(Pro_image)
    Lsample = Lab_pro[:, :, 0]
    asample = Lab_pro[:, :, 1]
    bsample = Lab_pro[:, :, 2]
    CDw = np.zeros(LBP.shape)
    Np = LBP.size
    for pp in list_of_patterns:
        homo_patches = LBP == pp
        homo_patches = ndimage.morphology.binary_closing(homo_patches, SE)
        patch_index, num_patches = ndimage.measurements.label(homo_patches)
        for ii in range(1, num_patches):
            idx = np.where(patch_index == ii)
            if idx[0].size > min_num_pixels:
                temp = MU.deltaosa(Lstd[idx], astd[idx], bstd[idx], Lsample[idx], asample[idx], bsample[idx])
                if sq:
                    temp = np.power(temp, 2)
                if not np.isnan(temp):
                    CDw[idx] = temp
                    wp = (1. * idx[0].size) / Np
                    cd += wp * temp
                wpt += 1  # wp
    return CDw, cd



def conf32_deltaosa_patches(Ref_image, Pro_image, th=0.045, r=1., min_num_pixels = 4, sq = False, KLCH=np.array([1, 1, 1])):
    # SE = np.ones((np.int_(2 * r + 1), np.int_(2 * r + 1)))
    LBP, _ = CT.lbp(Ref_image, th=th, r=r)
    SE = np.ones((2, 2))
    list_of_patterns = np.unique(LBP)
    cd = 0.
    wpt = 0.
    Lab_ref = CAM.RGBToLjg(Ref_image)
    Lstd = Lab_ref[:, :, 0]
    astd = Lab_ref[:, :, 1]
    bstd = Lab_ref[:, :, 2]
    Lab_pro = CAM.RGBToLjg(Pro_image)
    Lsample = Lab_pro[:, :, 0]
    asample = Lab_pro[:, :, 1]
    bsample = Lab_pro[:, :, 2]
    CDw = np.zeros(LBP.shape)
    Np = LBP.size
    for pp in list_of_patterns:
        homo_patches = LBP == pp
        homo_patches = ndimage.morphology.binary_closing(homo_patches, SE)
        patch_index, num_patches = ndimage.measurements.label(homo_patches)
        for ii in range(1, num_patches):
            idx = np.where(patch_index == ii)
            if idx[0].size > min_num_pixels:
                temp = MU.deltaosa(Lstd[idx], astd[idx], bstd[idx], Lsample[idx], asample[idx], bsample[idx])
                if sq:
                    temp = np.power(temp, 2)
                if not np.isnan(temp):
                    CDw[idx] = temp
                    wp = (1. * idx[0].size) / Np
                    cd += wp * temp
                wpt += 1  # wp
    return CDw, cd


def conf33_deltaosa_patches(Ref_image, Pro_image, th=0.045, r=1.5, min_num_pixels = 4, sq = False, KLCH=np.array([1, 1, 1])):
    # SE = np.ones((np.int_(2 * r + 1), np.int_(2 * r + 1)))
    LBP, _ = CT.lbp(Ref_image, th=th, r=r)
    SE = np.ones((2, 2))
    list_of_patterns = np.unique(LBP)
    cd = 0.
    wpt = 0.
    Lab_ref = CAM.RGBToLjg(Ref_image)
    Lstd = Lab_ref[:, :, 0]
    astd = Lab_ref[:, :, 1]
    bstd = Lab_ref[:, :, 2]
    Lab_pro = CAM.RGBToLjg(Pro_image)
    Lsample = Lab_pro[:, :, 0]
    asample = Lab_pro[:, :, 1]
    bsample = Lab_pro[:, :, 2]
    CDw = np.zeros(LBP.shape)
    Np = LBP.size
    for pp in list_of_patterns:
        homo_patches = LBP == pp
        homo_patches = ndimage.morphology.binary_closing(homo_patches, SE)
        patch_index, num_patches = ndimage.measurements.label(homo_patches)
        for ii in range(1, num_patches):
            idx = np.where(patch_index == ii)
            if idx[0].size > min_num_pixels:
                temp = MU.deltaosa(Lstd[idx], astd[idx], bstd[idx], Lsample[idx], asample[idx], bsample[idx])
                if sq:
                    temp = np.power(temp, 2)
                if not np.isnan(temp):
                    CDw[idx] = temp
                    wp = (1. * idx[0].size) / Np
                    cd += wp * temp
                wpt += 1  # wp
    return CDw, cd



def conf34_deltaosa_patches(Ref_image, Pro_image, th=0.045, r=2., min_num_pixels = 4, sq = False, KLCH=np.array([1, 1, 1])):
    # SE = np.ones((np.int_(2 * r + 1), np.int_(2 * r + 1)))
    LBP, _ = CT.lbp(Ref_image, th=th, r=r)
    SE = np.ones((2, 2))
    list_of_patterns = np.unique(LBP)
    cd = 0.
    wpt = 0.
    Lab_ref = CAM.RGBToLjg(Ref_image)
    Lstd = Lab_ref[:, :, 0]
    astd = Lab_ref[:, :, 1]
    bstd = Lab_ref[:, :, 2]
    Lab_pro = CAM.RGBToLjg(Pro_image)
    Lsample = Lab_pro[:, :, 0]
    asample = Lab_pro[:, :, 1]
    bsample = Lab_pro[:, :, 2]
    CDw = np.zeros(LBP.shape)
    Np = LBP.size
    for pp in list_of_patterns:
        homo_patches = LBP == pp
        homo_patches = ndimage.morphology.binary_closing(homo_patches, SE)
        patch_index, num_patches = ndimage.measurements.label(homo_patches)
        for ii in range(1, num_patches):
            idx = np.where(patch_index == ii)
            if idx[0].size > min_num_pixels:
                temp = MU.deltaosa(Lstd[idx], astd[idx], bstd[idx], Lsample[idx], asample[idx], bsample[idx])
                if sq:
                    temp = np.power(temp, 2)
                if not np.isnan(temp):
                    CDw[idx] = temp
                    wp = (1. * idx[0].size) / Np
                    cd += wp * temp
                wpt += 1  # wp
    return CDw, cd



def conf35_deltaosa_patches(Ref_image, Pro_image, th=0.045, r=3., min_num_pixels = 4, sq = False, KLCH=np.array([1, 1, 1])):
    # SE = np.ones((np.int_(2 * r + 1), np.int_(2 * r + 1)))
    LBP, _ = CT.lbp(Ref_image, th=th, r=r)
    SE = np.ones((2, 2))
    list_of_patterns = np.unique(LBP)
    cd = 0.
    wpt = 0.
    Lab_ref = CAM.RGBToLjg(Ref_image)
    Lstd = Lab_ref[:, :, 0]
    astd = Lab_ref[:, :, 1]
    bstd = Lab_ref[:, :, 2]
    Lab_pro = CAM.RGBToLjg(Pro_image)
    Lsample = Lab_pro[:, :, 0]
    asample = Lab_pro[:, :, 1]
    bsample = Lab_pro[:, :, 2]
    CDw = np.zeros(LBP.shape)
    Np = LBP.size
    for pp in list_of_patterns:
        homo_patches = LBP == pp
        homo_patches = ndimage.morphology.binary_closing(homo_patches, SE)
        patch_index, num_patches = ndimage.measurements.label(homo_patches)
        for ii in range(1, num_patches):
            idx = np.where(patch_index == ii)
            if idx[0].size > min_num_pixels:
                temp = MU.deltaosa(Lstd[idx], astd[idx], bstd[idx], Lsample[idx], asample[idx], bsample[idx])
                if sq:
                    temp = np.power(temp, 2)
                if not np.isnan(temp):
                    CDw[idx] = temp
                    wp = (1. * idx[0].size) / Np
                    cd += wp * temp
                wpt += 1  # wp
    return CDw, cd


def conf36_deltaosa_patches(Ref_image, Pro_image, th=0.05, r=1., min_num_pixels = 4, sq = False, KLCH=np.array([1, 1, 1])):
    # SE = np.ones((np.int_(2 * r + 1), np.int_(2 * r + 1)))
    LBP, _ = CT.lbp(Ref_image, th=th, r=r)
    SE = np.ones((2, 2))
    list_of_patterns = np.unique(LBP)
    cd = 0.
    wpt = 0.
    Lab_ref = CAM.RGBToLjg(Ref_image)
    Lstd = Lab_ref[:, :, 0]
    astd = Lab_ref[:, :, 1]
    bstd = Lab_ref[:, :, 2]
    Lab_pro = CAM.RGBToLjg(Pro_image)
    Lsample = Lab_pro[:, :, 0]
    asample = Lab_pro[:, :, 1]
    bsample = Lab_pro[:, :, 2]
    CDw = np.zeros(LBP.shape)
    Np = LBP.size
    for pp in list_of_patterns:
        homo_patches = LBP == pp
        homo_patches = ndimage.morphology.binary_closing(homo_patches, SE)
        patch_index, num_patches = ndimage.measurements.label(homo_patches)
        for ii in range(1, num_patches):
            idx = np.where(patch_index == ii)
            if idx[0].size > min_num_pixels:
                temp = MU.deltaosa(Lstd[idx], astd[idx], bstd[idx], Lsample[idx], asample[idx], bsample[idx])
                if sq:
                    temp = np.power(temp, 2)
                if not np.isnan(temp):
                    CDw[idx] = temp
                    wp = (1. * idx[0].size) / Np
                    cd += wp * temp
                wpt += 1  # wp
    return CDw, cd


def conf37_deltaosa_patches(Ref_image, Pro_image, th=0.05, r=1.5, min_num_pixels = 4, sq = False, KLCH=np.array([1, 1, 1])):
    # SE = np.ones((np.int_(2 * r + 1), np.int_(2 * r + 1)))
    LBP, _ = CT.lbp(Ref_image, th=th, r=r)
    SE = np.ones((2, 2))
    list_of_patterns = np.unique(LBP)
    cd = 0.
    wpt = 0.
    Lab_ref = CAM.RGBToLjg(Ref_image)
    Lstd = Lab_ref[:, :, 0]
    astd = Lab_ref[:, :, 1]
    bstd = Lab_ref[:, :, 2]
    Lab_pro = CAM.RGBToLjg(Pro_image)
    Lsample = Lab_pro[:, :, 0]
    asample = Lab_pro[:, :, 1]
    bsample = Lab_pro[:, :, 2]
    CDw = np.zeros(LBP.shape)
    Np = LBP.size
    for pp in list_of_patterns:
        homo_patches = LBP == pp
        homo_patches = ndimage.morphology.binary_closing(homo_patches, SE)
        patch_index, num_patches = ndimage.measurements.label(homo_patches)
        for ii in range(1, num_patches):
            idx = np.where(patch_index == ii)
            if idx[0].size > min_num_pixels:
                temp = MU.deltaosa(Lstd[idx], astd[idx], bstd[idx], Lsample[idx], asample[idx], bsample[idx])
                if sq:
                    temp = np.power(temp, 2)
                if not np.isnan(temp):
                    CDw[idx] = temp
                    wp = (1. * idx[0].size) / Np
                    cd += wp * temp
                wpt += 1  # wp
    return CDw, cd



def conf38_deltaosa_patches(Ref_image, Pro_image, th=0.05, r=2., min_num_pixels = 4, sq = False, KLCH=np.array([1, 1, 1])):
    # SE = np.ones((np.int_(2 * r + 1), np.int_(2 * r + 1)))
    LBP, _ = CT.lbp(Ref_image, th=th, r=r)
    SE = np.ones((2, 2))
    list_of_patterns = np.unique(LBP)
    cd = 0.
    wpt = 0.
    Lab_ref = CAM.RGBToLjg(Ref_image)
    Lstd = Lab_ref[:, :, 0]
    astd = Lab_ref[:, :, 1]
    bstd = Lab_ref[:, :, 2]
    Lab_pro = CAM.RGBToLjg(Pro_image)
    Lsample = Lab_pro[:, :, 0]
    asample = Lab_pro[:, :, 1]
    bsample = Lab_pro[:, :, 2]
    CDw = np.zeros(LBP.shape)
    Np = LBP.size
    for pp in list_of_patterns:
        homo_patches = LBP == pp
        homo_patches = ndimage.morphology.binary_closing(homo_patches, SE)
        patch_index, num_patches = ndimage.measurements.label(homo_patches)
        for ii in range(1, num_patches):
            idx = np.where(patch_index == ii)
            if idx[0].size > min_num_pixels:
                temp = MU.deltaosa(Lstd[idx], astd[idx], bstd[idx], Lsample[idx], asample[idx], bsample[idx])
                if sq:
                    temp = np.power(temp, 2)
                if not np.isnan(temp):
                    CDw[idx] = temp
                    wp = (1. * idx[0].size) / Np
                    cd += wp * temp
                wpt += 1  # wp
    return CDw, cd



def conf39_deltaosa_patches(Ref_image, Pro_image, th=0.05, r=3., min_num_pixels = 4, sq = False, KLCH=np.array([1, 1, 1])):
    # SE = np.ones((np.int_(2 * r + 1), np.int_(2 * r + 1)))
    LBP, _ = CT.lbp(Ref_image, th=th, r=r)
    SE = np.ones((2, 2))
    list_of_patterns = np.unique(LBP)
    cd = 0.
    wpt = 0.
    Lab_ref = CAM.RGBToLjg(Ref_image)
    Lstd = Lab_ref[:, :, 0]
    astd = Lab_ref[:, :, 1]
    bstd = Lab_ref[:, :, 2]
    Lab_pro = CAM.RGBToLjg(Pro_image)
    Lsample = Lab_pro[:, :, 0]
    asample = Lab_pro[:, :, 1]
    bsample = Lab_pro[:, :, 2]
    CDw = np.zeros(LBP.shape)
    Np = LBP.size
    for pp in list_of_patterns:
        homo_patches = LBP == pp
        homo_patches = ndimage.morphology.binary_closing(homo_patches, SE)
        patch_index, num_patches = ndimage.measurements.label(homo_patches)
        for ii in range(1, num_patches):
            idx = np.where(patch_index == ii)
            if idx[0].size > min_num_pixels:
                temp = MU.deltaosa(Lstd[idx], astd[idx], bstd[idx], Lsample[idx], asample[idx], bsample[idx])
                if sq:
                    temp = np.power(temp, 2)
                if not np.isnan(temp):
                    CDw[idx] = temp
                    wp = (1. * idx[0].size) / Np
                    cd += wp * temp
                wpt += 1  # wp
    return CDw, cd
