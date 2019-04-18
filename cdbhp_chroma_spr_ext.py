import numpy as np
from skimage import color
import cython_functions
from scipy import stats
from scipy import ndimage
import my_utilities as MU
import compute_texture as CT
import matplotlib.pyplot as plt
import color_spaces as CAM

def conf00_sprext_patches(Ref_image, Pro_image, th=0.005, r=1., min_num_pixels = 4, sq = False):
    SE = np.ones((np.int_(2 * r + 1), np.int_(2 * r + 1)))
    Yref = 0.299 * Ref_image[:, :, 0] + 0.587 * Ref_image[:, :, 1] + 0.114 * Ref_image[:, :, 2]
    Ypro = 0.299 * Pro_image[:, :, 0] + 0.587 * Pro_image[:, :, 1] + 0.114 * Pro_image[:, :, 2]
    DL, _ = MU.SSIM(Yref, Ypro, mode='same')
    dl = 0.
    # SE = np.ones((2, 2))
    YCbCr_ref = CAM.ycbcr(Ref_image)
    YCbCr_pro = CAM.ycbcr(Pro_image)
    ECbCr = np.sqrt(np.power(YCbCr_ref[:, :, 1] - YCbCr_pro[:, :, 1], 2) +\
                    np.power(YCbCr_ref[:, :, 2] - YCbCr_pro[:, :, 2], 2))
    LBP, _ = CT.lbp(Ref_image, th=th, r=r)
    list_of_patterns = np.unique(LBP)
    cd = 0.
    wpt = 0.
    CD = np.zeros(LBP.shape)
    Np = LBP.size
    for pp in list_of_patterns:
        homo_patches = LBP == pp
        homo_patches = ndimage.morphology.binary_closing(homo_patches, SE)
        patch_index, num_patches = ndimage.measurements.label(homo_patches)
        for ii in range(1, num_patches):
            idx = np.where(patch_index == ii)
            if idx[0].size > min_num_pixels:
                if sq:
                    chroma_spread = np.std(np.power(ECbCr[idx],2))
                    p = np.sort(np.array(np.power(ECbCr[idx],2)).ravel())[::-1]
                    chroma_extreme = np.mean(p[0:np.int_(np.ceil(p.size * 0.01))]) - p[np.int_(np.ceil(p.size * 0.01)) - 1]
                else:
                    chroma_spread = np.std(ECbCr[idx])
                    p = np.sort(np.array(ECbCr[idx]).ravel())[::-1]
                    chroma_extreme = np.mean(p[0:np.int_(np.ceil(p.size * 0.01))]) - p[np.int_(np.ceil(p.size * 0.01)) - 1]
                temp = 0.0192 * chroma_spread + 0.0076 * chroma_extreme
                templ = np.mean((1.-DL[idx])/2.)
                CD[idx] = 0.7*temp + 0.3*templ
                if np.isnan(temp):
                    pass
                wp = (1. * idx[0].size) / Np
                cd += wp * temp
                dl += wp * templ
                wpt += 1  # wp
    return CD, (0.7*cd + 0.3*dl)



def conf01_sprext_patches(Ref_image, Pro_image, th=0.005, r=1.5, min_num_pixels = 4, sq = False):
    SE = np.ones((np.int_(2 * r + 1), np.int_(2 * r + 1)))
    Yref = 0.299 * Ref_image[:, :, 0] + 0.587 * Ref_image[:, :, 1] + 0.114 * Ref_image[:, :, 2]
    Ypro = 0.299 * Pro_image[:, :, 0] + 0.587 * Pro_image[:, :, 1] + 0.114 * Pro_image[:, :, 2]
    DL, _ = MU.SSIM(Yref, Ypro, mode='same')
    dl = 0.
    # SE = np.ones((2, 2))
    YCbCr_ref = CAM.ycbcr(Ref_image)
    YCbCr_pro = CAM.ycbcr(Pro_image)
    ECbCr = np.sqrt(np.power(YCbCr_ref[:, :, 1] - YCbCr_pro[:, :, 1], 2) + \
                    np.power(YCbCr_ref[:, :, 2] - YCbCr_pro[:, :, 2], 2))
    LBP, _ = CT.lbp(Ref_image, th=th, r=r)
    list_of_patterns = np.unique(LBP)
    cd = 0.
    wpt = 0.
    CD = np.zeros(LBP.shape)
    Np = LBP.size
    for pp in list_of_patterns:
        homo_patches = LBP == pp
        homo_patches = ndimage.morphology.binary_closing(homo_patches, SE)
        patch_index, num_patches = ndimage.measurements.label(homo_patches)
        for ii in range(1, num_patches):
            idx = np.where(patch_index == ii)
            if idx[0].size > min_num_pixels:
                if sq:
                    chroma_spread = np.std(np.power(ECbCr[idx],2))
                    p = np.sort(np.array(np.power(ECbCr[idx],2)).ravel())[::-1]
                    chroma_extreme = np.mean(p[0:np.int_(np.ceil(p.size * 0.01))]) - p[np.int_(np.ceil(p.size * 0.01)) - 1]
                else:
                    chroma_spread = np.std(ECbCr[idx])
                    p = np.sort(np.array(ECbCr[idx]).ravel())[::-1]
                    chroma_extreme = np.mean(p[0:np.int_(np.ceil(p.size * 0.01))]) - p[np.int_(np.ceil(p.size * 0.01)) - 1]
                temp = 0.0192 * chroma_spread + 0.0076 * chroma_extreme
                templ = np.mean((1.-DL[idx])/2.)
                CD[idx] = 0.7*temp + 0.3*templ
                if np.isnan(temp):
                    pass
                wp = (1. * idx[0].size) / Np
                cd += wp * temp
                dl += wp * templ
                wpt += 1  # wp
    return CD, (0.7*cd + 0.3*dl)



def conf02_sprext_patches(Ref_image, Pro_image, th=0.005, r=2., min_num_pixels = 4, sq = False):
    SE = np.ones((np.int_(2 * r + 1), np.int_(2 * r + 1)))
    Yref = 0.299 * Ref_image[:, :, 0] + 0.587 * Ref_image[:, :, 1] + 0.114 * Ref_image[:, :, 2]
    Ypro = 0.299 * Pro_image[:, :, 0] + 0.587 * Pro_image[:, :, 1] + 0.114 * Pro_image[:, :, 2]
    DL, _ = MU.SSIM(Yref, Ypro, mode='same')
    dl = 0.
    # SE = np.ones((2, 2))
    YCbCr_ref = CAM.ycbcr(Ref_image)
    YCbCr_pro = CAM.ycbcr(Pro_image)
    ECbCr = np.sqrt(np.power(YCbCr_ref[:, :, 1] - YCbCr_pro[:, :, 1], 2) + \
                    np.power(YCbCr_ref[:, :, 2] - YCbCr_pro[:, :, 2], 2))
    LBP, _ = CT.lbp(Ref_image, th=th, r=r)
    list_of_patterns = np.unique(LBP)
    cd = 0.
    wpt = 0.
    CD = np.zeros(LBP.shape)
    Np = LBP.size
    for pp in list_of_patterns:
        homo_patches = LBP == pp
        homo_patches = ndimage.morphology.binary_closing(homo_patches, SE)
        patch_index, num_patches = ndimage.measurements.label(homo_patches)
        for ii in range(1, num_patches):
            idx = np.where(patch_index == ii)
            if idx[0].size > min_num_pixels:
                if sq:
                    chroma_spread = np.std(np.power(ECbCr[idx],2))
                    p = np.sort(np.array(np.power(ECbCr[idx],2)).ravel())[::-1]
                    chroma_extreme = np.mean(p[0:np.int_(np.ceil(p.size * 0.01))]) - p[np.int_(np.ceil(p.size * 0.01)) - 1]
                else:
                    chroma_spread = np.std(ECbCr[idx])
                    p = np.sort(np.array(ECbCr[idx]).ravel())[::-1]
                    chroma_extreme = np.mean(p[0:np.int_(np.ceil(p.size * 0.01))]) - p[np.int_(np.ceil(p.size * 0.01)) - 1]
                temp = 0.0192 * chroma_spread + 0.0076 * chroma_extreme
                templ = np.mean((1.-DL[idx])/2.)
                CD[idx] = 0.7*temp + 0.3*templ
                if np.isnan(temp):
                    pass
                wp = (1. * idx[0].size) / Np
                cd += wp * temp
                dl += wp * templ
                wpt += 1  # wp
    return CD, (0.7*cd + 0.3*dl)



def conf03_sprext_patches(Ref_image, Pro_image, th=0.005, r=3., min_num_pixels = 4, sq = False):
    SE = np.ones((np.int_(2 * r + 1), np.int_(2 * r + 1)))
    Yref = 0.299 * Ref_image[:, :, 0] + 0.587 * Ref_image[:, :, 1] + 0.114 * Ref_image[:, :, 2]
    Ypro = 0.299 * Pro_image[:, :, 0] + 0.587 * Pro_image[:, :, 1] + 0.114 * Pro_image[:, :, 2]
    DL, _ = MU.SSIM(Yref, Ypro, mode='same')
    dl = 0.
    # SE = np.ones((2, 2))
    YCbCr_ref = CAM.ycbcr(Ref_image)
    YCbCr_pro = CAM.ycbcr(Pro_image)
    ECbCr = np.sqrt(np.power(YCbCr_ref[:, :, 1] - YCbCr_pro[:, :, 1], 2) + \
                    np.power(YCbCr_ref[:, :, 2] - YCbCr_pro[:, :, 2], 2))
    LBP, _ = CT.lbp(Ref_image, th=th, r=r)
    list_of_patterns = np.unique(LBP)
    cd = 0.
    wpt = 0.
    CD = np.zeros(LBP.shape)
    Np = LBP.size
    for pp in list_of_patterns:
        homo_patches = LBP == pp
        homo_patches = ndimage.morphology.binary_closing(homo_patches, SE)
        patch_index, num_patches = ndimage.measurements.label(homo_patches)
        for ii in range(1, num_patches):
            idx = np.where(patch_index == ii)
            if idx[0].size > min_num_pixels:
                if sq:
                    chroma_spread = np.std(np.power(ECbCr[idx],2))
                    p = np.sort(np.array(np.power(ECbCr[idx],2)).ravel())[::-1]
                    chroma_extreme = np.mean(p[0:np.int_(np.ceil(p.size * 0.01))]) - p[np.int_(np.ceil(p.size * 0.01)) - 1]
                else:
                    chroma_spread = np.std(ECbCr[idx])
                    p = np.sort(np.array(ECbCr[idx]).ravel())[::-1]
                    chroma_extreme = np.mean(p[0:np.int_(np.ceil(p.size * 0.01))]) - p[np.int_(np.ceil(p.size * 0.01)) - 1]
                temp = 0.0192 * chroma_spread + 0.0076 * chroma_extreme
                templ = np.mean((1.-DL[idx])/2.)
                CD[idx] = 0.7*temp + 0.3*templ
                if np.isnan(temp):
                    pass
                wp = (1. * idx[0].size) / Np
                cd += wp * temp
                dl += wp * templ
                wpt += 1  # wp
    return CD, (0.7*cd + 0.3*dl)




def conf04_sprext_patches(Ref_image, Pro_image, th=0.01, r=1., min_num_pixels = 4, sq = False):
    SE = np.ones((np.int_(2 * r + 1), np.int_(2 * r + 1)))
    Yref = 0.299 * Ref_image[:, :, 0] + 0.587 * Ref_image[:, :, 1] + 0.114 * Ref_image[:, :, 2]
    Ypro = 0.299 * Pro_image[:, :, 0] + 0.587 * Pro_image[:, :, 1] + 0.114 * Pro_image[:, :, 2]
    DL, _ = MU.SSIM(Yref, Ypro, mode='same')
    dl = 0.
    # SE = np.ones((2, 2))
    YCbCr_ref = CAM.ycbcr(Ref_image)
    YCbCr_pro = CAM.ycbcr(Pro_image)
    ECbCr = np.sqrt(np.power(YCbCr_ref[:, :, 1] - YCbCr_pro[:, :, 1], 2) + \
                    np.power(YCbCr_ref[:, :, 2] - YCbCr_pro[:, :, 2], 2))
    LBP, _ = CT.lbp(Ref_image, th=th, r=r)
    list_of_patterns = np.unique(LBP)
    cd = 0.
    wpt = 0.
    CD = np.zeros(LBP.shape)
    Np = LBP.size
    for pp in list_of_patterns:
        homo_patches = LBP == pp
        homo_patches = ndimage.morphology.binary_closing(homo_patches, SE)
        patch_index, num_patches = ndimage.measurements.label(homo_patches)
        for ii in range(1, num_patches):
            idx = np.where(patch_index == ii)
            if idx[0].size > min_num_pixels:
                if sq:
                    chroma_spread = np.std(np.power(ECbCr[idx],2))
                    p = np.sort(np.array(np.power(ECbCr[idx],2)).ravel())[::-1]
                    chroma_extreme = np.mean(p[0:np.int_(np.ceil(p.size * 0.01))]) - p[np.int_(np.ceil(p.size * 0.01)) - 1]
                else:
                    chroma_spread = np.std(ECbCr[idx])
                    p = np.sort(np.array(ECbCr[idx]).ravel())[::-1]
                    chroma_extreme = np.mean(p[0:np.int_(np.ceil(p.size * 0.01))]) - p[np.int_(np.ceil(p.size * 0.01)) - 1]
                temp = 0.0192 * chroma_spread + 0.0076 * chroma_extreme
                templ = np.mean((1.-DL[idx])/2.)
                CD[idx] = 0.7*temp + 0.3*templ
                if np.isnan(temp):
                    pass
                wp = (1. * idx[0].size) / Np
                cd += wp * temp
                dl += wp * templ
                wpt += 1  # wp
    return CD, (0.7*cd + 0.3*dl)



def conf05_sprext_patches(Ref_image, Pro_image, th=0.01, r=1.5, min_num_pixels = 4, sq = False):
    SE = np.ones((np.int_(2 * r + 1), np.int_(2 * r + 1)))
    Yref = 0.299 * Ref_image[:, :, 0] + 0.587 * Ref_image[:, :, 1] + 0.114 * Ref_image[:, :, 2]
    Ypro = 0.299 * Pro_image[:, :, 0] + 0.587 * Pro_image[:, :, 1] + 0.114 * Pro_image[:, :, 2]
    DL, _ = MU.SSIM(Yref, Ypro, mode='same')
    dl = 0.
    # SE = np.ones((2, 2))
    YCbCr_ref = CAM.ycbcr(Ref_image)
    YCbCr_pro = CAM.ycbcr(Pro_image)
    ECbCr = np.sqrt(np.power(YCbCr_ref[:, :, 1] - YCbCr_pro[:, :, 1], 2) + \
                    np.power(YCbCr_ref[:, :, 2] - YCbCr_pro[:, :, 2], 2))
    LBP, _ = CT.lbp(Ref_image, th=th, r=r)
    list_of_patterns = np.unique(LBP)
    cd = 0.
    wpt = 0.
    CD = np.zeros(LBP.shape)
    Np = LBP.size
    for pp in list_of_patterns:
        homo_patches = LBP == pp
        homo_patches = ndimage.morphology.binary_closing(homo_patches, SE)
        patch_index, num_patches = ndimage.measurements.label(homo_patches)
        for ii in range(1, num_patches):
            idx = np.where(patch_index == ii)
            if idx[0].size > min_num_pixels:
                if sq:
                    chroma_spread = np.std(np.power(ECbCr[idx],2))
                    p = np.sort(np.array(np.power(ECbCr[idx],2)).ravel())[::-1]
                    chroma_extreme = np.mean(p[0:np.int_(np.ceil(p.size * 0.01))]) - p[np.int_(np.ceil(p.size * 0.01)) - 1]
                else:
                    chroma_spread = np.std(ECbCr[idx])
                    p = np.sort(np.array(ECbCr[idx]).ravel())[::-1]
                    chroma_extreme = np.mean(p[0:np.int_(np.ceil(p.size * 0.01))]) - p[np.int_(np.ceil(p.size * 0.01)) - 1]
                temp = 0.0192 * chroma_spread + 0.0076 * chroma_extreme
                templ = np.mean((1.-DL[idx])/2.)
                CD[idx] = 0.7*temp + 0.3*templ
                if np.isnan(temp):
                    pass
                wp = (1. * idx[0].size) / Np
                cd += wp * temp
                dl += wp * templ
                wpt += 1  # wp
    return CD, (0.7*cd + 0.3*dl)



def conf06_sprext_patches(Ref_image, Pro_image, th=0.01, r=2., min_num_pixels = 4, sq = False):
    SE = np.ones((np.int_(2 * r + 1), np.int_(2 * r + 1)))
    Yref = 0.299 * Ref_image[:, :, 0] + 0.587 * Ref_image[:, :, 1] + 0.114 * Ref_image[:, :, 2]
    Ypro = 0.299 * Pro_image[:, :, 0] + 0.587 * Pro_image[:, :, 1] + 0.114 * Pro_image[:, :, 2]
    DL, _ = MU.SSIM(Yref, Ypro, mode='same')
    dl = 0.
    # SE = np.ones((2, 2))
    YCbCr_ref = CAM.ycbcr(Ref_image)
    YCbCr_pro = CAM.ycbcr(Pro_image)
    ECbCr = np.sqrt(np.power(YCbCr_ref[:, :, 1] - YCbCr_pro[:, :, 1], 2) + \
                    np.power(YCbCr_ref[:, :, 2] - YCbCr_pro[:, :, 2], 2))
    LBP, _ = CT.lbp(Ref_image, th=th, r=r)
    list_of_patterns = np.unique(LBP)
    cd = 0.
    wpt = 0.
    CD = np.zeros(LBP.shape)
    Np = LBP.size
    for pp in list_of_patterns:
        homo_patches = LBP == pp
        homo_patches = ndimage.morphology.binary_closing(homo_patches, SE)
        patch_index, num_patches = ndimage.measurements.label(homo_patches)
        for ii in range(1, num_patches):
            idx = np.where(patch_index == ii)
            if idx[0].size > min_num_pixels:
                if sq:
                    chroma_spread = np.std(np.power(ECbCr[idx],2))
                    p = np.sort(np.array(np.power(ECbCr[idx],2)).ravel())[::-1]
                    chroma_extreme = np.mean(p[0:np.int_(np.ceil(p.size * 0.01))]) - p[np.int_(np.ceil(p.size * 0.01)) - 1]
                else:
                    chroma_spread = np.std(ECbCr[idx])
                    p = np.sort(np.array(ECbCr[idx]).ravel())[::-1]
                    chroma_extreme = np.mean(p[0:np.int_(np.ceil(p.size * 0.01))]) - p[np.int_(np.ceil(p.size * 0.01)) - 1]
                temp = 0.0192 * chroma_spread + 0.0076 * chroma_extreme
                templ = np.mean((1.-DL[idx])/2.)
                CD[idx] = 0.7*temp + 0.3*templ
                if np.isnan(temp):
                    pass
                wp = (1. * idx[0].size) / Np
                cd += wp * temp
                dl += wp * templ
                wpt += 1  # wp
    return CD, (0.7*cd + 0.3*dl)



def conf07_sprext_patches(Ref_image, Pro_image, th=0.01, r=3., min_num_pixels = 4, sq = False):
    SE = np.ones((np.int_(2 * r + 1), np.int_(2 * r + 1)))
    Yref = 0.299 * Ref_image[:, :, 0] + 0.587 * Ref_image[:, :, 1] + 0.114 * Ref_image[:, :, 2]
    Ypro = 0.299 * Pro_image[:, :, 0] + 0.587 * Pro_image[:, :, 1] + 0.114 * Pro_image[:, :, 2]
    DL, _ = MU.SSIM(Yref, Ypro, mode='same')
    dl = 0.
    # SE = np.ones((2, 2))
    YCbCr_ref = CAM.ycbcr(Ref_image)
    YCbCr_pro = CAM.ycbcr(Pro_image)
    ECbCr = np.sqrt(np.power(YCbCr_ref[:, :, 1] - YCbCr_pro[:, :, 1], 2) + \
                    np.power(YCbCr_ref[:, :, 2] - YCbCr_pro[:, :, 2], 2))
    LBP, _ = CT.lbp(Ref_image, th=th, r=r)
    list_of_patterns = np.unique(LBP)
    cd = 0.
    wpt = 0.
    CD = np.zeros(LBP.shape)
    Np = LBP.size
    for pp in list_of_patterns:
        homo_patches = LBP == pp
        homo_patches = ndimage.morphology.binary_closing(homo_patches, SE)
        patch_index, num_patches = ndimage.measurements.label(homo_patches)
        for ii in range(1, num_patches):
            idx = np.where(patch_index == ii)
            if idx[0].size > min_num_pixels:
                if sq:
                    chroma_spread = np.std(np.power(ECbCr[idx],2))
                    p = np.sort(np.array(np.power(ECbCr[idx],2)).ravel())[::-1]
                    chroma_extreme = np.mean(p[0:np.int_(np.ceil(p.size * 0.01))]) - p[np.int_(np.ceil(p.size * 0.01)) - 1]
                else:
                    chroma_spread = np.std(ECbCr[idx])
                    p = np.sort(np.array(ECbCr[idx]).ravel())[::-1]
                    chroma_extreme = np.mean(p[0:np.int_(np.ceil(p.size * 0.01))]) - p[np.int_(np.ceil(p.size * 0.01)) - 1]
                temp = 0.0192 * chroma_spread + 0.0076 * chroma_extreme
                templ = np.mean((1.-DL[idx])/2.)
                CD[idx] = 0.7*temp + 0.3*templ
                if np.isnan(temp):
                    pass
                wp = (1. * idx[0].size) / Np
                cd += wp * temp
                dl += wp * templ
                wpt += 1  # wp
    return CD, (0.7*cd + 0.3*dl)


def conf08_sprext_patches(Ref_image, Pro_image, th=0.015, r=1., min_num_pixels = 4, sq = False):
    SE = np.ones((np.int_(2 * r + 1), np.int_(2 * r + 1)))
    Yref = 0.299 * Ref_image[:, :, 0] + 0.587 * Ref_image[:, :, 1] + 0.114 * Ref_image[:, :, 2]
    Ypro = 0.299 * Pro_image[:, :, 0] + 0.587 * Pro_image[:, :, 1] + 0.114 * Pro_image[:, :, 2]
    DL, _ = MU.SSIM(Yref, Ypro, mode='same')
    dl = 0.
    # SE = np.ones((2, 2))
    YCbCr_ref = CAM.ycbcr(Ref_image)
    YCbCr_pro = CAM.ycbcr(Pro_image)
    ECbCr = np.sqrt(np.power(YCbCr_ref[:, :, 1] - YCbCr_pro[:, :, 1], 2) + \
                    np.power(YCbCr_ref[:, :, 2] - YCbCr_pro[:, :, 2], 2))
    LBP, _ = CT.lbp(Ref_image, th=th, r=r)
    list_of_patterns = np.unique(LBP)
    cd = 0.
    wpt = 0.
    CD = np.zeros(LBP.shape)
    Np = LBP.size
    for pp in list_of_patterns:
        homo_patches = LBP == pp
        homo_patches = ndimage.morphology.binary_closing(homo_patches, SE)
        patch_index, num_patches = ndimage.measurements.label(homo_patches)
        for ii in range(1, num_patches):
            idx = np.where(patch_index == ii)
            if idx[0].size > min_num_pixels:
                if sq:
                    chroma_spread = np.std(np.power(ECbCr[idx],2))
                    p = np.sort(np.array(np.power(ECbCr[idx],2)).ravel())[::-1]
                    chroma_extreme = np.mean(p[0:np.int_(np.ceil(p.size * 0.01))]) - p[np.int_(np.ceil(p.size * 0.01)) - 1]
                else:
                    chroma_spread = np.std(ECbCr[idx])
                    p = np.sort(np.array(ECbCr[idx]).ravel())[::-1]
                    chroma_extreme = np.mean(p[0:np.int_(np.ceil(p.size * 0.01))]) - p[np.int_(np.ceil(p.size * 0.01)) - 1]
                temp = 0.0192 * chroma_spread + 0.0076 * chroma_extreme
                templ = np.mean((1.-DL[idx])/2.)
                CD[idx] = 0.7*temp + 0.3*templ
                if np.isnan(temp):
                    pass
                wp = (1. * idx[0].size) / Np
                cd += wp * temp
                dl += wp * templ
                wpt += 1  # wp
    return CD, (0.7*cd + 0.3*dl)



def conf09_sprext_patches(Ref_image, Pro_image, th=0.015, r=1.5, min_num_pixels = 4, sq = False):
    SE = np.ones((np.int_(2 * r + 1), np.int_(2 * r + 1)))
    Yref = 0.299 * Ref_image[:, :, 0] + 0.587 * Ref_image[:, :, 1] + 0.114 * Ref_image[:, :, 2]
    Ypro = 0.299 * Pro_image[:, :, 0] + 0.587 * Pro_image[:, :, 1] + 0.114 * Pro_image[:, :, 2]
    DL, _ = MU.SSIM(Yref, Ypro, mode='same')
    dl = 0.
    # SE = np.ones((2, 2))
    YCbCr_ref = CAM.ycbcr(Ref_image)
    YCbCr_pro = CAM.ycbcr(Pro_image)
    ECbCr = np.sqrt(np.power(YCbCr_ref[:, :, 1] - YCbCr_pro[:, :, 1], 2) + \
                    np.power(YCbCr_ref[:, :, 2] - YCbCr_pro[:, :, 2], 2))
    LBP, _ = CT.lbp(Ref_image, th=th, r=r)
    list_of_patterns = np.unique(LBP)
    cd = 0.
    wpt = 0.
    CD = np.zeros(LBP.shape)
    Np = LBP.size
    for pp in list_of_patterns:
        homo_patches = LBP == pp
        homo_patches = ndimage.morphology.binary_closing(homo_patches, SE)
        patch_index, num_patches = ndimage.measurements.label(homo_patches)
        for ii in range(1, num_patches):
            idx = np.where(patch_index == ii)
            if idx[0].size > min_num_pixels:
                if sq:
                    chroma_spread = np.std(np.power(ECbCr[idx],2))
                    p = np.sort(np.array(np.power(ECbCr[idx],2)).ravel())[::-1]
                    chroma_extreme = np.mean(p[0:np.int_(np.ceil(p.size * 0.01))]) - p[np.int_(np.ceil(p.size * 0.01)) - 1]
                else:
                    chroma_spread = np.std(ECbCr[idx])
                    p = np.sort(np.array(ECbCr[idx]).ravel())[::-1]
                    chroma_extreme = np.mean(p[0:np.int_(np.ceil(p.size * 0.01))]) - p[np.int_(np.ceil(p.size * 0.01)) - 1]
                temp = 0.0192 * chroma_spread + 0.0076 * chroma_extreme
                templ = np.mean((1.-DL[idx])/2.)
                CD[idx] = 0.7*temp + 0.3*templ
                if np.isnan(temp):
                    pass
                wp = (1. * idx[0].size) / Np
                cd += wp * temp
                dl += wp * templ
                wpt += 1  # wp
    return CD, (0.7*cd + 0.3*dl)



def conf10_sprext_patches(Ref_image, Pro_image, th=0.015, r=2., min_num_pixels = 4, sq = False):
    SE = np.ones((np.int_(2 * r + 1), np.int_(2 * r + 1)))
    Yref = 0.299 * Ref_image[:, :, 0] + 0.587 * Ref_image[:, :, 1] + 0.114 * Ref_image[:, :, 2]
    Ypro = 0.299 * Pro_image[:, :, 0] + 0.587 * Pro_image[:, :, 1] + 0.114 * Pro_image[:, :, 2]
    DL, _ = MU.SSIM(Yref, Ypro, mode='same')
    dl = 0.
    # SE = np.ones((2, 2))
    YCbCr_ref = CAM.ycbcr(Ref_image)
    YCbCr_pro = CAM.ycbcr(Pro_image)
    ECbCr = np.sqrt(np.power(YCbCr_ref[:, :, 1] - YCbCr_pro[:, :, 1], 2) + \
                    np.power(YCbCr_ref[:, :, 2] - YCbCr_pro[:, :, 2], 2))
    LBP, _ = CT.lbp(Ref_image, th=th, r=r)
    list_of_patterns = np.unique(LBP)
    cd = 0.
    wpt = 0.
    CD = np.zeros(LBP.shape)
    Np = LBP.size
    for pp in list_of_patterns:
        homo_patches = LBP == pp
        homo_patches = ndimage.morphology.binary_closing(homo_patches, SE)
        patch_index, num_patches = ndimage.measurements.label(homo_patches)
        for ii in range(1, num_patches):
            idx = np.where(patch_index == ii)
            if idx[0].size > min_num_pixels:
                if sq:
                    chroma_spread = np.std(np.power(ECbCr[idx],2))
                    p = np.sort(np.array(np.power(ECbCr[idx],2)).ravel())[::-1]
                    chroma_extreme = np.mean(p[0:np.int_(np.ceil(p.size * 0.01))]) - p[np.int_(np.ceil(p.size * 0.01)) - 1]
                else:
                    chroma_spread = np.std(ECbCr[idx])
                    p = np.sort(np.array(ECbCr[idx]).ravel())[::-1]
                    chroma_extreme = np.mean(p[0:np.int_(np.ceil(p.size * 0.01))]) - p[np.int_(np.ceil(p.size * 0.01)) - 1]
                temp = 0.0192 * chroma_spread + 0.0076 * chroma_extreme
                templ = np.mean((1.-DL[idx])/2.)
                CD[idx] = 0.7*temp + 0.3*templ
                if np.isnan(temp):
                    pass
                wp = (1. * idx[0].size) / Np
                cd += wp * temp
                dl += wp * templ
                wpt += 1  # wp
    return CD, (0.7*cd + 0.3*dl)



def conf11_sprext_patches(Ref_image, Pro_image, th=0.015, r=3., min_num_pixels = 4, sq = False):
    SE = np.ones((np.int_(2 * r + 1), np.int_(2 * r + 1)))
    Yref = 0.299 * Ref_image[:, :, 0] + 0.587 * Ref_image[:, :, 1] + 0.114 * Ref_image[:, :, 2]
    Ypro = 0.299 * Pro_image[:, :, 0] + 0.587 * Pro_image[:, :, 1] + 0.114 * Pro_image[:, :, 2]
    DL, _ = MU.SSIM(Yref, Ypro, mode='same')
    dl = 0.
    # SE = np.ones((2, 2))
    YCbCr_ref = CAM.ycbcr(Ref_image)
    YCbCr_pro = CAM.ycbcr(Pro_image)
    ECbCr = np.sqrt(np.power(YCbCr_ref[:, :, 1] - YCbCr_pro[:, :, 1], 2) + \
                    np.power(YCbCr_ref[:, :, 2] - YCbCr_pro[:, :, 2], 2))
    LBP, _ = CT.lbp(Ref_image, th=th, r=r)
    list_of_patterns = np.unique(LBP)
    cd = 0.
    wpt = 0.
    CD = np.zeros(LBP.shape)
    Np = LBP.size
    for pp in list_of_patterns:
        homo_patches = LBP == pp
        homo_patches = ndimage.morphology.binary_closing(homo_patches, SE)
        patch_index, num_patches = ndimage.measurements.label(homo_patches)
        for ii in range(1, num_patches):
            idx = np.where(patch_index == ii)
            if idx[0].size > min_num_pixels:
                if sq:
                    chroma_spread = np.std(np.power(ECbCr[idx],2))
                    p = np.sort(np.array(np.power(ECbCr[idx],2)).ravel())[::-1]
                    chroma_extreme = np.mean(p[0:np.int_(np.ceil(p.size * 0.01))]) - p[np.int_(np.ceil(p.size * 0.01)) - 1]
                else:
                    chroma_spread = np.std(ECbCr[idx])
                    p = np.sort(np.array(ECbCr[idx]).ravel())[::-1]
                    chroma_extreme = np.mean(p[0:np.int_(np.ceil(p.size * 0.01))]) - p[np.int_(np.ceil(p.size * 0.01)) - 1]
                temp = 0.0192 * chroma_spread + 0.0076 * chroma_extreme
                templ = np.mean((1.-DL[idx])/2.)
                CD[idx] = 0.7*temp + 0.3*templ
                if np.isnan(temp):
                    pass
                wp = (1. * idx[0].size) / Np
                cd += wp * temp
                dl += wp * templ
                wpt += 1  # wp
    return CD, (0.7*cd + 0.3*dl)


def conf12_sprext_patches(Ref_image, Pro_image, th=0.02, r=1., min_num_pixels = 4, sq = False):
    SE = np.ones((np.int_(2 * r + 1), np.int_(2 * r + 1)))
    Yref = 0.299 * Ref_image[:, :, 0] + 0.587 * Ref_image[:, :, 1] + 0.114 * Ref_image[:, :, 2]
    Ypro = 0.299 * Pro_image[:, :, 0] + 0.587 * Pro_image[:, :, 1] + 0.114 * Pro_image[:, :, 2]
    DL, _ = MU.SSIM(Yref, Ypro, mode='same')
    dl = 0.
    # SE = np.ones((2, 2))
    YCbCr_ref = CAM.ycbcr(Ref_image)
    YCbCr_pro = CAM.ycbcr(Pro_image)
    ECbCr = np.sqrt(np.power(YCbCr_ref[:, :, 1] - YCbCr_pro[:, :, 1], 2) + \
                    np.power(YCbCr_ref[:, :, 2] - YCbCr_pro[:, :, 2], 2))
    LBP, _ = CT.lbp(Ref_image, th=th, r=r)
    list_of_patterns = np.unique(LBP)
    cd = 0.
    wpt = 0.
    CD = np.zeros(LBP.shape)
    Np = LBP.size
    for pp in list_of_patterns:
        homo_patches = LBP == pp
        homo_patches = ndimage.morphology.binary_closing(homo_patches, SE)
        patch_index, num_patches = ndimage.measurements.label(homo_patches)
        for ii in range(1, num_patches):
            idx = np.where(patch_index == ii)
            if idx[0].size > min_num_pixels:
                if sq:
                    chroma_spread = np.std(np.power(ECbCr[idx],2))
                    p = np.sort(np.array(np.power(ECbCr[idx],2)).ravel())[::-1]
                    chroma_extreme = np.mean(p[0:np.int_(np.ceil(p.size * 0.01))]) - p[np.int_(np.ceil(p.size * 0.01)) - 1]
                else:
                    chroma_spread = np.std(ECbCr[idx])
                    p = np.sort(np.array(ECbCr[idx]).ravel())[::-1]
                    chroma_extreme = np.mean(p[0:np.int_(np.ceil(p.size * 0.01))]) - p[np.int_(np.ceil(p.size * 0.01)) - 1]
                temp = 0.0192 * chroma_spread + 0.0076 * chroma_extreme
                templ = np.mean((1.-DL[idx])/2.)
                CD[idx] = 0.7*temp + 0.3*templ
                if np.isnan(temp):
                    pass
                wp = (1. * idx[0].size) / Np
                cd += wp * temp
                dl += wp * templ
                wpt += 1  # wp
    return CD, (0.7*cd + 0.3*dl)



def conf13_sprext_patches(Ref_image, Pro_image, th=0.02, r=1.5, min_num_pixels = 4, sq = False):
    SE = np.ones((np.int_(2 * r + 1), np.int_(2 * r + 1)))
    Yref = 0.299 * Ref_image[:, :, 0] + 0.587 * Ref_image[:, :, 1] + 0.114 * Ref_image[:, :, 2]
    Ypro = 0.299 * Pro_image[:, :, 0] + 0.587 * Pro_image[:, :, 1] + 0.114 * Pro_image[:, :, 2]
    DL, _ = MU.SSIM(Yref, Ypro, mode='same')
    dl = 0.
    # SE = np.ones((2, 2))
    YCbCr_ref = CAM.ycbcr(Ref_image)
    YCbCr_pro = CAM.ycbcr(Pro_image)
    ECbCr = np.sqrt(np.power(YCbCr_ref[:, :, 1] - YCbCr_pro[:, :, 1], 2) + \
                    np.power(YCbCr_ref[:, :, 2] - YCbCr_pro[:, :, 2], 2))
    LBP, _ = CT.lbp(Ref_image, th=th, r=r)
    list_of_patterns = np.unique(LBP)
    cd = 0.
    wpt = 0.
    CD = np.zeros(LBP.shape)
    Np = LBP.size
    for pp in list_of_patterns:
        homo_patches = LBP == pp
        homo_patches = ndimage.morphology.binary_closing(homo_patches, SE)
        patch_index, num_patches = ndimage.measurements.label(homo_patches)
        for ii in range(1, num_patches):
            idx = np.where(patch_index == ii)
            if idx[0].size > min_num_pixels:
                if sq:
                    chroma_spread = np.std(np.power(ECbCr[idx],2))
                    p = np.sort(np.array(np.power(ECbCr[idx],2)).ravel())[::-1]
                    chroma_extreme = np.mean(p[0:np.int_(np.ceil(p.size * 0.01))]) - p[np.int_(np.ceil(p.size * 0.01)) - 1]
                else:
                    chroma_spread = np.std(ECbCr[idx])
                    p = np.sort(np.array(ECbCr[idx]).ravel())[::-1]
                    chroma_extreme = np.mean(p[0:np.int_(np.ceil(p.size * 0.01))]) - p[np.int_(np.ceil(p.size * 0.01)) - 1]
                temp = 0.0192 * chroma_spread + 0.0076 * chroma_extreme
                templ = np.mean((1.-DL[idx])/2.)
                CD[idx] = 0.7*temp + 0.3*templ
                if np.isnan(temp):
                    pass
                wp = (1. * idx[0].size) / Np
                cd += wp * temp
                dl += wp * templ
                wpt += 1  # wp
    return CD, (0.7*cd + 0.3*dl)



def conf14_sprext_patches(Ref_image, Pro_image, th=0.02, r=2., min_num_pixels = 4, sq = False):
    SE = np.ones((np.int_(2 * r + 1), np.int_(2 * r + 1)))
    Yref = 0.299 * Ref_image[:, :, 0] + 0.587 * Ref_image[:, :, 1] + 0.114 * Ref_image[:, :, 2]
    Ypro = 0.299 * Pro_image[:, :, 0] + 0.587 * Pro_image[:, :, 1] + 0.114 * Pro_image[:, :, 2]
    DL, _ = MU.SSIM(Yref, Ypro, mode='same')
    dl = 0.
    # SE = np.ones((2, 2))
    YCbCr_ref = CAM.ycbcr(Ref_image)
    YCbCr_pro = CAM.ycbcr(Pro_image)
    ECbCr = np.sqrt(np.power(YCbCr_ref[:, :, 1] - YCbCr_pro[:, :, 1], 2) + \
                    np.power(YCbCr_ref[:, :, 2] - YCbCr_pro[:, :, 2], 2))
    LBP, _ = CT.lbp(Ref_image, th=th, r=r)
    list_of_patterns = np.unique(LBP)
    cd = 0.
    wpt = 0.
    CD = np.zeros(LBP.shape)
    Np = LBP.size
    for pp in list_of_patterns:
        homo_patches = LBP == pp
        homo_patches = ndimage.morphology.binary_closing(homo_patches, SE)
        patch_index, num_patches = ndimage.measurements.label(homo_patches)
        for ii in range(1, num_patches):
            idx = np.where(patch_index == ii)
            if idx[0].size > min_num_pixels:
                if sq:
                    chroma_spread = np.std(np.power(ECbCr[idx],2))
                    p = np.sort(np.array(np.power(ECbCr[idx],2)).ravel())[::-1]
                    chroma_extreme = np.mean(p[0:np.int_(np.ceil(p.size * 0.01))]) - p[np.int_(np.ceil(p.size * 0.01)) - 1]
                else:
                    chroma_spread = np.std(ECbCr[idx])
                    p = np.sort(np.array(ECbCr[idx]).ravel())[::-1]
                    chroma_extreme = np.mean(p[0:np.int_(np.ceil(p.size * 0.01))]) - p[np.int_(np.ceil(p.size * 0.01)) - 1]
                temp = 0.0192 * chroma_spread + 0.0076 * chroma_extreme
                templ = np.mean((1.-DL[idx])/2.)
                CD[idx] = 0.7*temp + 0.3*templ
                if np.isnan(temp):
                    pass
                wp = (1. * idx[0].size) / Np
                cd += wp * temp
                dl += wp * templ
                wpt += 1  # wp
    return CD, (0.7*cd + 0.3*dl)



def conf15_sprext_patches(Ref_image, Pro_image, th=0.02, r=3., min_num_pixels = 4, sq = False):
    SE = np.ones((np.int_(2 * r + 1), np.int_(2 * r + 1)))
    Yref = 0.299 * Ref_image[:, :, 0] + 0.587 * Ref_image[:, :, 1] + 0.114 * Ref_image[:, :, 2]
    Ypro = 0.299 * Pro_image[:, :, 0] + 0.587 * Pro_image[:, :, 1] + 0.114 * Pro_image[:, :, 2]
    DL, _ = MU.SSIM(Yref, Ypro, mode='same')
    dl = 0.
    # SE = np.ones((2, 2))
    YCbCr_ref = CAM.ycbcr(Ref_image)
    YCbCr_pro = CAM.ycbcr(Pro_image)
    ECbCr = np.sqrt(np.power(YCbCr_ref[:, :, 1] - YCbCr_pro[:, :, 1], 2) + \
                    np.power(YCbCr_ref[:, :, 2] - YCbCr_pro[:, :, 2], 2))
    LBP, _ = CT.lbp(Ref_image, th=th, r=r)
    list_of_patterns = np.unique(LBP)
    cd = 0.
    wpt = 0.
    CD = np.zeros(LBP.shape)
    Np = LBP.size
    for pp in list_of_patterns:
        homo_patches = LBP == pp
        homo_patches = ndimage.morphology.binary_closing(homo_patches, SE)
        patch_index, num_patches = ndimage.measurements.label(homo_patches)
        for ii in range(1, num_patches):
            idx = np.where(patch_index == ii)
            if idx[0].size > min_num_pixels:
                if sq:
                    chroma_spread = np.std(np.power(ECbCr[idx],2))
                    p = np.sort(np.array(np.power(ECbCr[idx],2)).ravel())[::-1]
                    chroma_extreme = np.mean(p[0:np.int_(np.ceil(p.size * 0.01))]) - p[np.int_(np.ceil(p.size * 0.01)) - 1]
                else:
                    chroma_spread = np.std(ECbCr[idx])
                    p = np.sort(np.array(ECbCr[idx]).ravel())[::-1]
                    chroma_extreme = np.mean(p[0:np.int_(np.ceil(p.size * 0.01))]) - p[np.int_(np.ceil(p.size * 0.01)) - 1]
                temp = 0.0192 * chroma_spread + 0.0076 * chroma_extreme
                templ = np.mean((1.-DL[idx])/2.)
                CD[idx] = 0.7*temp + 0.3*templ
                if np.isnan(temp):
                    pass
                wp = (1. * idx[0].size) / Np
                cd += wp * temp
                dl += wp * templ
                wpt += 1  # wp
    return CD, (0.7*cd + 0.3*dl)


def conf16_sprext_patches(Ref_image, Pro_image, th=0.025, r=1., min_num_pixels = 4, sq = False):
    SE = np.ones((np.int_(2 * r + 1), np.int_(2 * r + 1)))
    Yref = 0.299 * Ref_image[:, :, 0] + 0.587 * Ref_image[:, :, 1] + 0.114 * Ref_image[:, :, 2]
    Ypro = 0.299 * Pro_image[:, :, 0] + 0.587 * Pro_image[:, :, 1] + 0.114 * Pro_image[:, :, 2]
    DL, _ = MU.SSIM(Yref, Ypro, mode='same')
    dl = 0.
    # SE = np.ones((2, 2))
    YCbCr_ref = CAM.ycbcr(Ref_image)
    YCbCr_pro = CAM.ycbcr(Pro_image)
    ECbCr = np.sqrt(np.power(YCbCr_ref[:, :, 1] - YCbCr_pro[:, :, 1], 2) + \
                    np.power(YCbCr_ref[:, :, 2] - YCbCr_pro[:, :, 2], 2))
    LBP, _ = CT.lbp(Ref_image, th=th, r=r)
    list_of_patterns = np.unique(LBP)
    cd = 0.
    wpt = 0.
    CD = np.zeros(LBP.shape)
    Np = LBP.size
    for pp in list_of_patterns:
        homo_patches = LBP == pp
        homo_patches = ndimage.morphology.binary_closing(homo_patches, SE)
        patch_index, num_patches = ndimage.measurements.label(homo_patches)
        for ii in range(1, num_patches):
            idx = np.where(patch_index == ii)
            if idx[0].size > min_num_pixels:
                if sq:
                    chroma_spread = np.std(np.power(ECbCr[idx],2))
                    p = np.sort(np.array(np.power(ECbCr[idx],2)).ravel())[::-1]
                    chroma_extreme = np.mean(p[0:np.int_(np.ceil(p.size * 0.01))]) - p[np.int_(np.ceil(p.size * 0.01)) - 1]
                else:
                    chroma_spread = np.std(ECbCr[idx])
                    p = np.sort(np.array(ECbCr[idx]).ravel())[::-1]
                    chroma_extreme = np.mean(p[0:np.int_(np.ceil(p.size * 0.01))]) - p[np.int_(np.ceil(p.size * 0.01)) - 1]
                temp = 0.0192 * chroma_spread + 0.0076 * chroma_extreme
                templ = np.mean((1.-DL[idx])/2.)
                CD[idx] = 0.7*temp + 0.3*templ
                if np.isnan(temp):
                    pass
                wp = (1. * idx[0].size) / Np
                cd += wp * temp
                dl += wp * templ
                wpt += 1  # wp
    return CD, (0.7*cd + 0.3*dl)



def conf17_sprext_patches(Ref_image, Pro_image, th=0.025, r=1.5, min_num_pixels = 4, sq = False):
    SE = np.ones((np.int_(2 * r + 1), np.int_(2 * r + 1)))
    Yref = 0.299 * Ref_image[:, :, 0] + 0.587 * Ref_image[:, :, 1] + 0.114 * Ref_image[:, :, 2]
    Ypro = 0.299 * Pro_image[:, :, 0] + 0.587 * Pro_image[:, :, 1] + 0.114 * Pro_image[:, :, 2]
    DL, _ = MU.SSIM(Yref, Ypro, mode='same')
    dl = 0.
    # SE = np.ones((2, 2))
    YCbCr_ref = CAM.ycbcr(Ref_image)
    YCbCr_pro = CAM.ycbcr(Pro_image)
    ECbCr = np.sqrt(np.power(YCbCr_ref[:, :, 1] - YCbCr_pro[:, :, 1], 2) + \
                    np.power(YCbCr_ref[:, :, 2] - YCbCr_pro[:, :, 2], 2))
    LBP, _ = CT.lbp(Ref_image, th=th, r=r)
    list_of_patterns = np.unique(LBP)
    cd = 0.
    wpt = 0.
    CD = np.zeros(LBP.shape)
    Np = LBP.size
    for pp in list_of_patterns:
        homo_patches = LBP == pp
        homo_patches = ndimage.morphology.binary_closing(homo_patches, SE)
        patch_index, num_patches = ndimage.measurements.label(homo_patches)
        for ii in range(1, num_patches):
            idx = np.where(patch_index == ii)
            if idx[0].size > min_num_pixels:
                if sq:
                    chroma_spread = np.std(np.power(ECbCr[idx],2))
                    p = np.sort(np.array(np.power(ECbCr[idx],2)).ravel())[::-1]
                    chroma_extreme = np.mean(p[0:np.int_(np.ceil(p.size * 0.01))]) - p[np.int_(np.ceil(p.size * 0.01)) - 1]
                else:
                    chroma_spread = np.std(ECbCr[idx])
                    p = np.sort(np.array(ECbCr[idx]).ravel())[::-1]
                    chroma_extreme = np.mean(p[0:np.int_(np.ceil(p.size * 0.01))]) - p[np.int_(np.ceil(p.size * 0.01)) - 1]
                temp = 0.0192 * chroma_spread + 0.0076 * chroma_extreme
                templ = np.mean((1.-DL[idx])/2.)
                CD[idx] = 0.7*temp + 0.3*templ
                if np.isnan(temp):
                    pass
                wp = (1. * idx[0].size) / Np
                cd += wp * temp
                dl += wp * templ
                wpt += 1  # wp
    return CD, (0.7*cd + 0.3*dl)



def conf18_sprext_patches(Ref_image, Pro_image, th=0.025, r=2., min_num_pixels = 4, sq = False):
    SE = np.ones((np.int_(2 * r + 1), np.int_(2 * r + 1)))
    Yref = 0.299 * Ref_image[:, :, 0] + 0.587 * Ref_image[:, :, 1] + 0.114 * Ref_image[:, :, 2]
    Ypro = 0.299 * Pro_image[:, :, 0] + 0.587 * Pro_image[:, :, 1] + 0.114 * Pro_image[:, :, 2]
    DL, _ = MU.SSIM(Yref, Ypro, mode='same')
    dl = 0.
    # SE = np.ones((2, 2))
    YCbCr_ref = CAM.ycbcr(Ref_image)
    YCbCr_pro = CAM.ycbcr(Pro_image)
    ECbCr = np.sqrt(np.power(YCbCr_ref[:, :, 1] - YCbCr_pro[:, :, 1], 2) + \
                    np.power(YCbCr_ref[:, :, 2] - YCbCr_pro[:, :, 2], 2))
    LBP, _ = CT.lbp(Ref_image, th=th, r=r)
    list_of_patterns = np.unique(LBP)
    cd = 0.
    wpt = 0.
    CD = np.zeros(LBP.shape)
    Np = LBP.size
    for pp in list_of_patterns:
        homo_patches = LBP == pp
        homo_patches = ndimage.morphology.binary_closing(homo_patches, SE)
        patch_index, num_patches = ndimage.measurements.label(homo_patches)
        for ii in range(1, num_patches):
            idx = np.where(patch_index == ii)
            if idx[0].size > min_num_pixels:
                if sq:
                    chroma_spread = np.std(np.power(ECbCr[idx],2))
                    p = np.sort(np.array(np.power(ECbCr[idx],2)).ravel())[::-1]
                    chroma_extreme = np.mean(p[0:np.int_(np.ceil(p.size * 0.01))]) - p[np.int_(np.ceil(p.size * 0.01)) - 1]
                else:
                    chroma_spread = np.std(ECbCr[idx])
                    p = np.sort(np.array(ECbCr[idx]).ravel())[::-1]
                    chroma_extreme = np.mean(p[0:np.int_(np.ceil(p.size * 0.01))]) - p[np.int_(np.ceil(p.size * 0.01)) - 1]
                temp = 0.0192 * chroma_spread + 0.0076 * chroma_extreme
                templ = np.mean((1.-DL[idx])/2.)
                CD[idx] = 0.7*temp + 0.3*templ
                if np.isnan(temp):
                    pass
                wp = (1. * idx[0].size) / Np
                cd += wp * temp
                dl += wp * templ
                wpt += 1  # wp
    return CD, (0.7*cd + 0.3*dl)



def conf19_sprext_patches(Ref_image, Pro_image, th=0.025, r=3., min_num_pixels = 4, sq = False):
    SE = np.ones((np.int_(2 * r + 1), np.int_(2 * r + 1)))
    Yref = 0.299 * Ref_image[:, :, 0] + 0.587 * Ref_image[:, :, 1] + 0.114 * Ref_image[:, :, 2]
    Ypro = 0.299 * Pro_image[:, :, 0] + 0.587 * Pro_image[:, :, 1] + 0.114 * Pro_image[:, :, 2]
    DL, _ = MU.SSIM(Yref, Ypro, mode='same')
    dl = 0.
    # SE = np.ones((2, 2))
    YCbCr_ref = CAM.ycbcr(Ref_image)
    YCbCr_pro = CAM.ycbcr(Pro_image)
    ECbCr = np.sqrt(np.power(YCbCr_ref[:, :, 1] - YCbCr_pro[:, :, 1], 2) + \
                    np.power(YCbCr_ref[:, :, 2] - YCbCr_pro[:, :, 2], 2))
    LBP, _ = CT.lbp(Ref_image, th=th, r=r)
    list_of_patterns = np.unique(LBP)
    cd = 0.
    wpt = 0.
    CD = np.zeros(LBP.shape)
    Np = LBP.size
    for pp in list_of_patterns:
        homo_patches = LBP == pp
        homo_patches = ndimage.morphology.binary_closing(homo_patches, SE)
        patch_index, num_patches = ndimage.measurements.label(homo_patches)
        for ii in range(1, num_patches):
            idx = np.where(patch_index == ii)
            if idx[0].size > min_num_pixels:
                if sq:
                    chroma_spread = np.std(np.power(ECbCr[idx],2))
                    p = np.sort(np.array(np.power(ECbCr[idx],2)).ravel())[::-1]
                    chroma_extreme = np.mean(p[0:np.int_(np.ceil(p.size * 0.01))]) - p[np.int_(np.ceil(p.size * 0.01)) - 1]
                else:
                    chroma_spread = np.std(ECbCr[idx])
                    p = np.sort(np.array(ECbCr[idx]).ravel())[::-1]
                    chroma_extreme = np.mean(p[0:np.int_(np.ceil(p.size * 0.01))]) - p[np.int_(np.ceil(p.size * 0.01)) - 1]
                temp = 0.0192 * chroma_spread + 0.0076 * chroma_extreme
                templ = np.mean((1.-DL[idx])/2.)
                CD[idx] = 0.7*temp + 0.3*templ
                if np.isnan(temp):
                    pass
                wp = (1. * idx[0].size) / Np
                cd += wp * temp
                dl += wp * templ
                wpt += 1  # wp
    return CD, (0.7*cd + 0.3*dl)


def conf20_sprext_patches(Ref_image, Pro_image, th=0.03, r=1., min_num_pixels = 4, sq = False):
    SE = np.ones((np.int_(2 * r + 1), np.int_(2 * r + 1)))
    Yref = 0.299 * Ref_image[:, :, 0] + 0.587 * Ref_image[:, :, 1] + 0.114 * Ref_image[:, :, 2]
    Ypro = 0.299 * Pro_image[:, :, 0] + 0.587 * Pro_image[:, :, 1] + 0.114 * Pro_image[:, :, 2]
    DL, _ = MU.SSIM(Yref, Ypro, mode='same')
    dl = 0.
    # SE = np.ones((2, 2))
    YCbCr_ref = CAM.ycbcr(Ref_image)
    YCbCr_pro = CAM.ycbcr(Pro_image)
    ECbCr = np.sqrt(np.power(YCbCr_ref[:, :, 1] - YCbCr_pro[:, :, 1], 2) + \
                    np.power(YCbCr_ref[:, :, 2] - YCbCr_pro[:, :, 2], 2))
    LBP, _ = CT.lbp(Ref_image, th=th, r=r)
    list_of_patterns = np.unique(LBP)
    cd = 0.
    wpt = 0.
    CD = np.zeros(LBP.shape)
    Np = LBP.size
    for pp in list_of_patterns:
        homo_patches = LBP == pp
        homo_patches = ndimage.morphology.binary_closing(homo_patches, SE)
        patch_index, num_patches = ndimage.measurements.label(homo_patches)
        for ii in range(1, num_patches):
            idx = np.where(patch_index == ii)
            if idx[0].size > min_num_pixels:
                if sq:
                    chroma_spread = np.std(np.power(ECbCr[idx],2))
                    p = np.sort(np.array(np.power(ECbCr[idx],2)).ravel())[::-1]
                    chroma_extreme = np.mean(p[0:np.int_(np.ceil(p.size * 0.01))]) - p[np.int_(np.ceil(p.size * 0.01)) - 1]
                else:
                    chroma_spread = np.std(ECbCr[idx])
                    p = np.sort(np.array(ECbCr[idx]).ravel())[::-1]
                    chroma_extreme = np.mean(p[0:np.int_(np.ceil(p.size * 0.01))]) - p[np.int_(np.ceil(p.size * 0.01)) - 1]
                temp = 0.0192 * chroma_spread + 0.0076 * chroma_extreme
                templ = np.mean((1.-DL[idx])/2.)
                CD[idx] = 0.7*temp + 0.3*templ
                if np.isnan(temp):
                    pass
                wp = (1. * idx[0].size) / Np
                cd += wp * temp
                dl += wp * templ
                wpt += 1  # wp
    return CD, (0.7*cd + 0.3*dl)



def conf21_sprext_patches(Ref_image, Pro_image, th=0.03, r=1.5, min_num_pixels = 4, sq = False):
    SE = np.ones((np.int_(2 * r + 1), np.int_(2 * r + 1)))
    Yref = 0.299 * Ref_image[:, :, 0] + 0.587 * Ref_image[:, :, 1] + 0.114 * Ref_image[:, :, 2]
    Ypro = 0.299 * Pro_image[:, :, 0] + 0.587 * Pro_image[:, :, 1] + 0.114 * Pro_image[:, :, 2]
    DL, _ = MU.SSIM(Yref, Ypro, mode='same')
    dl = 0.
    # SE = np.ones((2, 2))
    YCbCr_ref = CAM.ycbcr(Ref_image)
    YCbCr_pro = CAM.ycbcr(Pro_image)
    ECbCr = np.sqrt(np.power(YCbCr_ref[:, :, 1] - YCbCr_pro[:, :, 1], 2) + \
                    np.power(YCbCr_ref[:, :, 2] - YCbCr_pro[:, :, 2], 2))
    LBP, _ = CT.lbp(Ref_image, th=th, r=r)
    list_of_patterns = np.unique(LBP)
    cd = 0.
    wpt = 0.
    CD = np.zeros(LBP.shape)
    Np = LBP.size
    for pp in list_of_patterns:
        homo_patches = LBP == pp
        homo_patches = ndimage.morphology.binary_closing(homo_patches, SE)
        patch_index, num_patches = ndimage.measurements.label(homo_patches)
        for ii in range(1, num_patches):
            idx = np.where(patch_index == ii)
            if idx[0].size > min_num_pixels:
                if sq:
                    chroma_spread = np.std(np.power(ECbCr[idx],2))
                    p = np.sort(np.array(np.power(ECbCr[idx],2)).ravel())[::-1]
                    chroma_extreme = np.mean(p[0:np.int_(np.ceil(p.size * 0.01))]) - p[np.int_(np.ceil(p.size * 0.01)) - 1]
                else:
                    chroma_spread = np.std(ECbCr[idx])
                    p = np.sort(np.array(ECbCr[idx]).ravel())[::-1]
                    chroma_extreme = np.mean(p[0:np.int_(np.ceil(p.size * 0.01))]) - p[np.int_(np.ceil(p.size * 0.01)) - 1]
                temp = 0.0192 * chroma_spread + 0.0076 * chroma_extreme
                templ = np.mean((1.-DL[idx])/2.)
                CD[idx] = 0.7*temp + 0.3*templ
                if np.isnan(temp):
                    pass
                wp = (1. * idx[0].size) / Np
                cd += wp * temp
                dl += wp * templ
                wpt += 1  # wp
    return CD, (0.7*cd + 0.3*dl)



def conf22_sprext_patches(Ref_image, Pro_image, th=0.03, r=2., min_num_pixels = 4, sq = False):
    SE = np.ones((np.int_(2 * r + 1), np.int_(2 * r + 1)))
    Yref = 0.299 * Ref_image[:, :, 0] + 0.587 * Ref_image[:, :, 1] + 0.114 * Ref_image[:, :, 2]
    Ypro = 0.299 * Pro_image[:, :, 0] + 0.587 * Pro_image[:, :, 1] + 0.114 * Pro_image[:, :, 2]
    DL, _ = MU.SSIM(Yref, Ypro, mode='same')
    dl = 0.
    # SE = np.ones((2, 2))
    YCbCr_ref = CAM.ycbcr(Ref_image)
    YCbCr_pro = CAM.ycbcr(Pro_image)
    ECbCr = np.sqrt(np.power(YCbCr_ref[:, :, 1] - YCbCr_pro[:, :, 1], 2) + \
                    np.power(YCbCr_ref[:, :, 2] - YCbCr_pro[:, :, 2], 2))
    LBP, _ = CT.lbp(Ref_image, th=th, r=r)
    list_of_patterns = np.unique(LBP)
    cd = 0.
    wpt = 0.
    CD = np.zeros(LBP.shape)
    Np = LBP.size
    for pp in list_of_patterns:
        homo_patches = LBP == pp
        homo_patches = ndimage.morphology.binary_closing(homo_patches, SE)
        patch_index, num_patches = ndimage.measurements.label(homo_patches)
        for ii in range(1, num_patches):
            idx = np.where(patch_index == ii)
            if idx[0].size > min_num_pixels:
                if sq:
                    chroma_spread = np.std(np.power(ECbCr[idx],2))
                    p = np.sort(np.array(np.power(ECbCr[idx],2)).ravel())[::-1]
                    chroma_extreme = np.mean(p[0:np.int_(np.ceil(p.size * 0.01))]) - p[np.int_(np.ceil(p.size * 0.01)) - 1]
                else:
                    chroma_spread = np.std(ECbCr[idx])
                    p = np.sort(np.array(ECbCr[idx]).ravel())[::-1]
                    chroma_extreme = np.mean(p[0:np.int_(np.ceil(p.size * 0.01))]) - p[np.int_(np.ceil(p.size * 0.01)) - 1]
                temp = 0.0192 * chroma_spread + 0.0076 * chroma_extreme
                templ = np.mean((1.-DL[idx])/2.)
                CD[idx] = 0.7*temp + 0.3*templ
                if np.isnan(temp):
                    pass
                wp = (1. * idx[0].size) / Np
                cd += wp * temp
                dl += wp * templ
                wpt += 1  # wp
    return CD, (0.7*cd + 0.3*dl)



def conf23_sprext_patches(Ref_image, Pro_image, th=0.03, r=3., min_num_pixels = 4, sq = False):
    SE = np.ones((np.int_(2 * r + 1), np.int_(2 * r + 1)))
    Yref = 0.299 * Ref_image[:, :, 0] + 0.587 * Ref_image[:, :, 1] + 0.114 * Ref_image[:, :, 2]
    Ypro = 0.299 * Pro_image[:, :, 0] + 0.587 * Pro_image[:, :, 1] + 0.114 * Pro_image[:, :, 2]
    DL, _ = MU.SSIM(Yref, Ypro, mode='same')
    dl = 0.
    # SE = np.ones((2, 2))
    YCbCr_ref = CAM.ycbcr(Ref_image)
    YCbCr_pro = CAM.ycbcr(Pro_image)
    ECbCr = np.sqrt(np.power(YCbCr_ref[:, :, 1] - YCbCr_pro[:, :, 1], 2) + \
                    np.power(YCbCr_ref[:, :, 2] - YCbCr_pro[:, :, 2], 2))
    LBP, _ = CT.lbp(Ref_image, th=th, r=r)
    list_of_patterns = np.unique(LBP)
    cd = 0.
    wpt = 0.
    CD = np.zeros(LBP.shape)
    Np = LBP.size
    for pp in list_of_patterns:
        homo_patches = LBP == pp
        homo_patches = ndimage.morphology.binary_closing(homo_patches, SE)
        patch_index, num_patches = ndimage.measurements.label(homo_patches)
        for ii in range(1, num_patches):
            idx = np.where(patch_index == ii)
            if idx[0].size > min_num_pixels:
                if sq:
                    chroma_spread = np.std(np.power(ECbCr[idx],2))
                    p = np.sort(np.array(np.power(ECbCr[idx],2)).ravel())[::-1]
                    chroma_extreme = np.mean(p[0:np.int_(np.ceil(p.size * 0.01))]) - p[np.int_(np.ceil(p.size * 0.01)) - 1]
                else:
                    chroma_spread = np.std(ECbCr[idx])
                    p = np.sort(np.array(ECbCr[idx]).ravel())[::-1]
                    chroma_extreme = np.mean(p[0:np.int_(np.ceil(p.size * 0.01))]) - p[np.int_(np.ceil(p.size * 0.01)) - 1]
                temp = 0.0192 * chroma_spread + 0.0076 * chroma_extreme
                templ = np.mean((1.-DL[idx])/2.)
                CD[idx] = 0.7*temp + 0.3*templ
                if np.isnan(temp):
                    pass
                wp = (1. * idx[0].size) / Np
                cd += wp * temp
                dl += wp * templ
                wpt += 1  # wp
    return CD, (0.7*cd + 0.3*dl)



def conf24_sprext_patches(Ref_image, Pro_image, th=0.035, r=1., min_num_pixels = 4, sq = False):
    SE = np.ones((np.int_(2 * r + 1), np.int_(2 * r + 1)))
    Yref = 0.299 * Ref_image[:, :, 0] + 0.587 * Ref_image[:, :, 1] + 0.114 * Ref_image[:, :, 2]
    Ypro = 0.299 * Pro_image[:, :, 0] + 0.587 * Pro_image[:, :, 1] + 0.114 * Pro_image[:, :, 2]
    DL, _ = MU.SSIM(Yref, Ypro, mode='same')
    dl = 0.
    # SE = np.ones((2, 2))
    YCbCr_ref = CAM.ycbcr(Ref_image)
    YCbCr_pro = CAM.ycbcr(Pro_image)
    ECbCr = np.sqrt(np.power(YCbCr_ref[:, :, 1] - YCbCr_pro[:, :, 1], 2) + \
                    np.power(YCbCr_ref[:, :, 2] - YCbCr_pro[:, :, 2], 2))
    LBP, _ = CT.lbp(Ref_image, th=th, r=r)
    list_of_patterns = np.unique(LBP)
    cd = 0.
    wpt = 0.
    CD = np.zeros(LBP.shape)
    Np = LBP.size
    for pp in list_of_patterns:
        homo_patches = LBP == pp
        homo_patches = ndimage.morphology.binary_closing(homo_patches, SE)
        patch_index, num_patches = ndimage.measurements.label(homo_patches)
        for ii in range(1, num_patches):
            idx = np.where(patch_index == ii)
            if idx[0].size > min_num_pixels:
                if sq:
                    chroma_spread = np.std(np.power(ECbCr[idx],2))
                    p = np.sort(np.array(np.power(ECbCr[idx],2)).ravel())[::-1]
                    chroma_extreme = np.mean(p[0:np.int_(np.ceil(p.size * 0.01))]) - p[np.int_(np.ceil(p.size * 0.01)) - 1]
                else:
                    chroma_spread = np.std(ECbCr[idx])
                    p = np.sort(np.array(ECbCr[idx]).ravel())[::-1]
                    chroma_extreme = np.mean(p[0:np.int_(np.ceil(p.size * 0.01))]) - p[np.int_(np.ceil(p.size * 0.01)) - 1]
                temp = 0.0192 * chroma_spread + 0.0076 * chroma_extreme
                templ = np.mean((1.-DL[idx])/2.)
                CD[idx] = 0.7*temp + 0.3*templ
                if np.isnan(temp):
                    pass
                wp = (1. * idx[0].size) / Np
                cd += wp * temp
                dl += wp * templ
                wpt += 1  # wp
    return CD, (0.7*cd + 0.3*dl)



def conf25_sprext_patches(Ref_image, Pro_image, th=0.035, r=1.5, min_num_pixels = 4, sq = False):
    SE = np.ones((np.int_(2 * r + 1), np.int_(2 * r + 1)))
    Yref = 0.299 * Ref_image[:, :, 0] + 0.587 * Ref_image[:, :, 1] + 0.114 * Ref_image[:, :, 2]
    Ypro = 0.299 * Pro_image[:, :, 0] + 0.587 * Pro_image[:, :, 1] + 0.114 * Pro_image[:, :, 2]
    DL, _ = MU.SSIM(Yref, Ypro, mode='same')
    dl = 0.
    # SE = np.ones((2, 2))
    YCbCr_ref = CAM.ycbcr(Ref_image)
    YCbCr_pro = CAM.ycbcr(Pro_image)
    ECbCr = np.sqrt(np.power(YCbCr_ref[:, :, 1] - YCbCr_pro[:, :, 1], 2) + \
                    np.power(YCbCr_ref[:, :, 2] - YCbCr_pro[:, :, 2], 2))
    LBP, _ = CT.lbp(Ref_image, th=th, r=r)
    list_of_patterns = np.unique(LBP)
    cd = 0.
    wpt = 0.
    CD = np.zeros(LBP.shape)
    Np = LBP.size
    for pp in list_of_patterns:
        homo_patches = LBP == pp
        homo_patches = ndimage.morphology.binary_closing(homo_patches, SE)
        patch_index, num_patches = ndimage.measurements.label(homo_patches)
        for ii in range(1, num_patches):
            idx = np.where(patch_index == ii)
            if idx[0].size > min_num_pixels:
                if sq:
                    chroma_spread = np.std(np.power(ECbCr[idx],2))
                    p = np.sort(np.array(np.power(ECbCr[idx],2)).ravel())[::-1]
                    chroma_extreme = np.mean(p[0:np.int_(np.ceil(p.size * 0.01))]) - p[np.int_(np.ceil(p.size * 0.01)) - 1]
                else:
                    chroma_spread = np.std(ECbCr[idx])
                    p = np.sort(np.array(ECbCr[idx]).ravel())[::-1]
                    chroma_extreme = np.mean(p[0:np.int_(np.ceil(p.size * 0.01))]) - p[np.int_(np.ceil(p.size * 0.01)) - 1]
                temp = 0.0192 * chroma_spread + 0.0076 * chroma_extreme
                templ = np.mean((1.-DL[idx])/2.)
                CD[idx] = 0.7*temp + 0.3*templ
                if np.isnan(temp):
                    pass
                wp = (1. * idx[0].size) / Np
                cd += wp * temp
                dl += wp * templ
                wpt += 1  # wp
    return CD, (0.7*cd + 0.3*dl)



def conf26_sprext_patches(Ref_image, Pro_image, th=0.035, r=2., min_num_pixels = 4, sq = False):
    SE = np.ones((np.int_(2 * r + 1), np.int_(2 * r + 1)))
    Yref = 0.299 * Ref_image[:, :, 0] + 0.587 * Ref_image[:, :, 1] + 0.114 * Ref_image[:, :, 2]
    Ypro = 0.299 * Pro_image[:, :, 0] + 0.587 * Pro_image[:, :, 1] + 0.114 * Pro_image[:, :, 2]
    DL, _ = MU.SSIM(Yref, Ypro, mode='same')
    dl = 0.
    # SE = np.ones((2, 2))
    YCbCr_ref = CAM.ycbcr(Ref_image)
    YCbCr_pro = CAM.ycbcr(Pro_image)
    ECbCr = np.sqrt(np.power(YCbCr_ref[:, :, 1] - YCbCr_pro[:, :, 1], 2) + \
                    np.power(YCbCr_ref[:, :, 2] - YCbCr_pro[:, :, 2], 2))
    LBP, _ = CT.lbp(Ref_image, th=th, r=r)
    list_of_patterns = np.unique(LBP)
    cd = 0.
    wpt = 0.
    CD = np.zeros(LBP.shape)
    Np = LBP.size
    for pp in list_of_patterns:
        homo_patches = LBP == pp
        homo_patches = ndimage.morphology.binary_closing(homo_patches, SE)
        patch_index, num_patches = ndimage.measurements.label(homo_patches)
        for ii in range(1, num_patches):
            idx = np.where(patch_index == ii)
            if idx[0].size > min_num_pixels:
                if sq:
                    chroma_spread = np.std(np.power(ECbCr[idx],2))
                    p = np.sort(np.array(np.power(ECbCr[idx],2)).ravel())[::-1]
                    chroma_extreme = np.mean(p[0:np.int_(np.ceil(p.size * 0.01))]) - p[np.int_(np.ceil(p.size * 0.01)) - 1]
                else:
                    chroma_spread = np.std(ECbCr[idx])
                    p = np.sort(np.array(ECbCr[idx]).ravel())[::-1]
                    chroma_extreme = np.mean(p[0:np.int_(np.ceil(p.size * 0.01))]) - p[np.int_(np.ceil(p.size * 0.01)) - 1]
                temp = 0.0192 * chroma_spread + 0.0076 * chroma_extreme
                templ = np.mean((1.-DL[idx])/2.)
                CD[idx] = 0.7*temp + 0.3*templ
                if np.isnan(temp):
                    pass
                wp = (1. * idx[0].size) / Np
                cd += wp * temp
                dl += wp * templ
                wpt += 1  # wp
    return CD, (0.7*cd + 0.3*dl)



def conf27_sprext_patches(Ref_image, Pro_image, th=0.035, r=3., min_num_pixels = 4, sq = False):
    SE = np.ones((np.int_(2 * r + 1), np.int_(2 * r + 1)))
    Yref = 0.299 * Ref_image[:, :, 0] + 0.587 * Ref_image[:, :, 1] + 0.114 * Ref_image[:, :, 2]
    Ypro = 0.299 * Pro_image[:, :, 0] + 0.587 * Pro_image[:, :, 1] + 0.114 * Pro_image[:, :, 2]
    DL, _ = MU.SSIM(Yref, Ypro, mode='same')
    dl = 0.
    # SE = np.ones((2, 2))
    YCbCr_ref = CAM.ycbcr(Ref_image)
    YCbCr_pro = CAM.ycbcr(Pro_image)
    ECbCr = np.sqrt(np.power(YCbCr_ref[:, :, 1] - YCbCr_pro[:, :, 1], 2) + \
                    np.power(YCbCr_ref[:, :, 2] - YCbCr_pro[:, :, 2], 2))
    LBP, _ = CT.lbp(Ref_image, th=th, r=r)
    list_of_patterns = np.unique(LBP)
    cd = 0.
    wpt = 0.
    CD = np.zeros(LBP.shape)
    Np = LBP.size
    for pp in list_of_patterns:
        homo_patches = LBP == pp
        homo_patches = ndimage.morphology.binary_closing(homo_patches, SE)
        patch_index, num_patches = ndimage.measurements.label(homo_patches)
        for ii in range(1, num_patches):
            idx = np.where(patch_index == ii)
            if idx[0].size > min_num_pixels:
                if sq:
                    chroma_spread = np.std(np.power(ECbCr[idx],2))
                    p = np.sort(np.array(np.power(ECbCr[idx],2)).ravel())[::-1]
                    chroma_extreme = np.mean(p[0:np.int_(np.ceil(p.size * 0.01))]) - p[np.int_(np.ceil(p.size * 0.01)) - 1]
                else:
                    chroma_spread = np.std(ECbCr[idx])
                    p = np.sort(np.array(ECbCr[idx]).ravel())[::-1]
                    chroma_extreme = np.mean(p[0:np.int_(np.ceil(p.size * 0.01))]) - p[np.int_(np.ceil(p.size * 0.01)) - 1]
                temp = 0.0192 * chroma_spread + 0.0076 * chroma_extreme
                templ = np.mean((1.-DL[idx])/2.)
                CD[idx] = 0.7*temp + 0.3*templ
                if np.isnan(temp):
                    pass
                wp = (1. * idx[0].size) / Np
                cd += wp * temp
                dl += wp * templ
                wpt += 1  # wp
    return CD, (0.7*cd + 0.3*dl)



def conf28_sprext_patches(Ref_image, Pro_image, th=0.04, r=1., min_num_pixels = 4, sq = False):
    SE = np.ones((np.int_(2 * r + 1), np.int_(2 * r + 1)))
    Yref = 0.299 * Ref_image[:, :, 0] + 0.587 * Ref_image[:, :, 1] + 0.114 * Ref_image[:, :, 2]
    Ypro = 0.299 * Pro_image[:, :, 0] + 0.587 * Pro_image[:, :, 1] + 0.114 * Pro_image[:, :, 2]
    DL, _ = MU.SSIM(Yref, Ypro, mode='same')
    dl = 0.
    # SE = np.ones((2, 2))
    YCbCr_ref = CAM.ycbcr(Ref_image)
    YCbCr_pro = CAM.ycbcr(Pro_image)
    ECbCr = np.sqrt(np.power(YCbCr_ref[:, :, 1] - YCbCr_pro[:, :, 1], 2) + \
                    np.power(YCbCr_ref[:, :, 2] - YCbCr_pro[:, :, 2], 2))
    LBP, _ = CT.lbp(Ref_image, th=th, r=r)
    list_of_patterns = np.unique(LBP)
    cd = 0.
    wpt = 0.
    CD = np.zeros(LBP.shape)
    Np = LBP.size
    for pp in list_of_patterns:
        homo_patches = LBP == pp
        homo_patches = ndimage.morphology.binary_closing(homo_patches, SE)
        patch_index, num_patches = ndimage.measurements.label(homo_patches)
        for ii in range(1, num_patches):
            idx = np.where(patch_index == ii)
            if idx[0].size > min_num_pixels:
                if sq:
                    chroma_spread = np.std(np.power(ECbCr[idx],2))
                    p = np.sort(np.array(np.power(ECbCr[idx],2)).ravel())[::-1]
                    chroma_extreme = np.mean(p[0:np.int_(np.ceil(p.size * 0.01))]) - p[np.int_(np.ceil(p.size * 0.01)) - 1]
                else:
                    chroma_spread = np.std(ECbCr[idx])
                    p = np.sort(np.array(ECbCr[idx]).ravel())[::-1]
                    chroma_extreme = np.mean(p[0:np.int_(np.ceil(p.size * 0.01))]) - p[np.int_(np.ceil(p.size * 0.01)) - 1]
                temp = 0.0192 * chroma_spread + 0.0076 * chroma_extreme
                templ = np.mean((1.-DL[idx])/2.)
                CD[idx] = 0.7*temp + 0.3*templ
                if np.isnan(temp):
                    pass
                wp = (1. * idx[0].size) / Np
                cd += wp * temp
                dl += wp * templ
                wpt += 1  # wp
    return CD, (0.7*cd + 0.3*dl)



def conf29_sprext_patches(Ref_image, Pro_image, th=0.04, r=1.5, min_num_pixels = 4, sq = False):
    SE = np.ones((np.int_(2 * r + 1), np.int_(2 * r + 1)))
    Yref = 0.299 * Ref_image[:, :, 0] + 0.587 * Ref_image[:, :, 1] + 0.114 * Ref_image[:, :, 2]
    Ypro = 0.299 * Pro_image[:, :, 0] + 0.587 * Pro_image[:, :, 1] + 0.114 * Pro_image[:, :, 2]
    DL, _ = MU.SSIM(Yref, Ypro, mode='same')
    dl = 0.
    # SE = np.ones((2, 2))
    YCbCr_ref = CAM.ycbcr(Ref_image)
    YCbCr_pro = CAM.ycbcr(Pro_image)
    ECbCr = np.sqrt(np.power(YCbCr_ref[:, :, 1] - YCbCr_pro[:, :, 1], 2) + \
                    np.power(YCbCr_ref[:, :, 2] - YCbCr_pro[:, :, 2], 2))
    LBP, _ = CT.lbp(Ref_image, th=th, r=r)
    list_of_patterns = np.unique(LBP)
    cd = 0.
    wpt = 0.
    CD = np.zeros(LBP.shape)
    Np = LBP.size
    for pp in list_of_patterns:
        homo_patches = LBP == pp
        homo_patches = ndimage.morphology.binary_closing(homo_patches, SE)
        patch_index, num_patches = ndimage.measurements.label(homo_patches)
        for ii in range(1, num_patches):
            idx = np.where(patch_index == ii)
            if idx[0].size > min_num_pixels:
                if sq:
                    chroma_spread = np.std(np.power(ECbCr[idx],2))
                    p = np.sort(np.array(np.power(ECbCr[idx],2)).ravel())[::-1]
                    chroma_extreme = np.mean(p[0:np.int_(np.ceil(p.size * 0.01))]) - p[np.int_(np.ceil(p.size * 0.01)) - 1]
                else:
                    chroma_spread = np.std(ECbCr[idx])
                    p = np.sort(np.array(ECbCr[idx]).ravel())[::-1]
                    chroma_extreme = np.mean(p[0:np.int_(np.ceil(p.size * 0.01))]) - p[np.int_(np.ceil(p.size * 0.01)) - 1]
                temp = 0.0192 * chroma_spread + 0.0076 * chroma_extreme
                templ = np.mean((1.-DL[idx])/2.)
                CD[idx] = 0.7*temp + 0.3*templ
                if np.isnan(temp):
                    pass
                wp = (1. * idx[0].size) / Np
                cd += wp * temp
                dl += wp * templ
                wpt += 1  # wp
    return CD, (0.7*cd + 0.3*dl)



def conf30_sprext_patches(Ref_image, Pro_image, th=0.04, r=2., min_num_pixels = 4, sq = False):
    SE = np.ones((np.int_(2 * r + 1), np.int_(2 * r + 1)))
    Yref = 0.299 * Ref_image[:, :, 0] + 0.587 * Ref_image[:, :, 1] + 0.114 * Ref_image[:, :, 2]
    Ypro = 0.299 * Pro_image[:, :, 0] + 0.587 * Pro_image[:, :, 1] + 0.114 * Pro_image[:, :, 2]
    DL, _ = MU.SSIM(Yref, Ypro, mode='same')
    dl = 0.
    # SE = np.ones((2, 2))
    YCbCr_ref = CAM.ycbcr(Ref_image)
    YCbCr_pro = CAM.ycbcr(Pro_image)
    ECbCr = np.sqrt(np.power(YCbCr_ref[:, :, 1] - YCbCr_pro[:, :, 1], 2) + \
                    np.power(YCbCr_ref[:, :, 2] - YCbCr_pro[:, :, 2], 2))
    LBP, _ = CT.lbp(Ref_image, th=th, r=r)
    list_of_patterns = np.unique(LBP)
    cd = 0.
    wpt = 0.
    CD = np.zeros(LBP.shape)
    Np = LBP.size
    for pp in list_of_patterns:
        homo_patches = LBP == pp
        homo_patches = ndimage.morphology.binary_closing(homo_patches, SE)
        patch_index, num_patches = ndimage.measurements.label(homo_patches)
        for ii in range(1, num_patches):
            idx = np.where(patch_index == ii)
            if idx[0].size > min_num_pixels:
                if sq:
                    chroma_spread = np.std(np.power(ECbCr[idx],2))
                    p = np.sort(np.array(np.power(ECbCr[idx],2)).ravel())[::-1]
                    chroma_extreme = np.mean(p[0:np.int_(np.ceil(p.size * 0.01))]) - p[np.int_(np.ceil(p.size * 0.01)) - 1]
                else:
                    chroma_spread = np.std(ECbCr[idx])
                    p = np.sort(np.array(ECbCr[idx]).ravel())[::-1]
                    chroma_extreme = np.mean(p[0:np.int_(np.ceil(p.size * 0.01))]) - p[np.int_(np.ceil(p.size * 0.01)) - 1]
                temp = 0.0192 * chroma_spread + 0.0076 * chroma_extreme
                templ = np.mean((1.-DL[idx])/2.)
                CD[idx] = 0.7*temp + 0.3*templ
                if np.isnan(temp):
                    pass
                wp = (1. * idx[0].size) / Np
                cd += wp * temp
                dl += wp * templ
                wpt += 1  # wp
    return CD, (0.7*cd + 0.3*dl)



def conf31_sprext_patches(Ref_image, Pro_image, th=0.04, r=3., min_num_pixels = 4, sq = False):
    SE = np.ones((np.int_(2 * r + 1), np.int_(2 * r + 1)))
    Yref = 0.299 * Ref_image[:, :, 0] + 0.587 * Ref_image[:, :, 1] + 0.114 * Ref_image[:, :, 2]
    Ypro = 0.299 * Pro_image[:, :, 0] + 0.587 * Pro_image[:, :, 1] + 0.114 * Pro_image[:, :, 2]
    DL, _ = MU.SSIM(Yref, Ypro, mode='same')
    dl = 0.
    # SE = np.ones((2, 2))
    YCbCr_ref = CAM.ycbcr(Ref_image)
    YCbCr_pro = CAM.ycbcr(Pro_image)
    ECbCr = np.sqrt(np.power(YCbCr_ref[:, :, 1] - YCbCr_pro[:, :, 1], 2) + \
                    np.power(YCbCr_ref[:, :, 2] - YCbCr_pro[:, :, 2], 2))
    LBP, _ = CT.lbp(Ref_image, th=th, r=r)
    list_of_patterns = np.unique(LBP)
    cd = 0.
    wpt = 0.
    CD = np.zeros(LBP.shape)
    Np = LBP.size
    for pp in list_of_patterns:
        homo_patches = LBP == pp
        homo_patches = ndimage.morphology.binary_closing(homo_patches, SE)
        patch_index, num_patches = ndimage.measurements.label(homo_patches)
        for ii in range(1, num_patches):
            idx = np.where(patch_index == ii)
            if idx[0].size > min_num_pixels:
                if sq:
                    chroma_spread = np.std(np.power(ECbCr[idx],2))
                    p = np.sort(np.array(np.power(ECbCr[idx],2)).ravel())[::-1]
                    chroma_extreme = np.mean(p[0:np.int_(np.ceil(p.size * 0.01))]) - p[np.int_(np.ceil(p.size * 0.01)) - 1]
                else:
                    chroma_spread = np.std(ECbCr[idx])
                    p = np.sort(np.array(ECbCr[idx]).ravel())[::-1]
                    chroma_extreme = np.mean(p[0:np.int_(np.ceil(p.size * 0.01))]) - p[np.int_(np.ceil(p.size * 0.01)) - 1]
                temp = 0.0192 * chroma_spread + 0.0076 * chroma_extreme
                templ = np.mean((1.-DL[idx])/2.)
                CD[idx] = 0.7*temp + 0.3*templ
                if np.isnan(temp):
                    pass
                wp = (1. * idx[0].size) / Np
                cd += wp * temp
                dl += wp * templ
                wpt += 1  # wp
    return CD, (0.7*cd + 0.3*dl)



def conf32_sprext_patches(Ref_image, Pro_image, th=0.045, r=1., min_num_pixels = 4, sq = False):
    SE = np.ones((np.int_(2 * r + 1), np.int_(2 * r + 1)))
    Yref = 0.299 * Ref_image[:, :, 0] + 0.587 * Ref_image[:, :, 1] + 0.114 * Ref_image[:, :, 2]
    Ypro = 0.299 * Pro_image[:, :, 0] + 0.587 * Pro_image[:, :, 1] + 0.114 * Pro_image[:, :, 2]
    DL, _ = MU.SSIM(Yref, Ypro, mode='same')
    dl = 0.
    # SE = np.ones((2, 2))
    YCbCr_ref = CAM.ycbcr(Ref_image)
    YCbCr_pro = CAM.ycbcr(Pro_image)
    ECbCr = np.sqrt(np.power(YCbCr_ref[:, :, 1] - YCbCr_pro[:, :, 1], 2) + \
                    np.power(YCbCr_ref[:, :, 2] - YCbCr_pro[:, :, 2], 2))
    LBP, _ = CT.lbp(Ref_image, th=th, r=r)
    list_of_patterns = np.unique(LBP)
    cd = 0.
    wpt = 0.
    CD = np.zeros(LBP.shape)
    Np = LBP.size
    for pp in list_of_patterns:
        homo_patches = LBP == pp
        homo_patches = ndimage.morphology.binary_closing(homo_patches, SE)
        patch_index, num_patches = ndimage.measurements.label(homo_patches)
        for ii in range(1, num_patches):
            idx = np.where(patch_index == ii)
            if idx[0].size > min_num_pixels:
                if sq:
                    chroma_spread = np.std(np.power(ECbCr[idx],2))
                    p = np.sort(np.array(np.power(ECbCr[idx],2)).ravel())[::-1]
                    chroma_extreme = np.mean(p[0:np.int_(np.ceil(p.size * 0.01))]) - p[np.int_(np.ceil(p.size * 0.01)) - 1]
                else:
                    chroma_spread = np.std(ECbCr[idx])
                    p = np.sort(np.array(ECbCr[idx]).ravel())[::-1]
                    chroma_extreme = np.mean(p[0:np.int_(np.ceil(p.size * 0.01))]) - p[np.int_(np.ceil(p.size * 0.01)) - 1]
                temp = 0.0192 * chroma_spread + 0.0076 * chroma_extreme
                templ = np.mean((1.-DL[idx])/2.)
                CD[idx] = 0.7*temp + 0.3*templ
                if np.isnan(temp):
                    pass
                wp = (1. * idx[0].size) / Np
                cd += wp * temp
                dl += wp * templ
                wpt += 1  # wp
    return CD, (0.7*cd + 0.3*dl)



def conf33_sprext_patches(Ref_image, Pro_image, th=0.045, r=1.5, min_num_pixels = 4, sq = False):
    SE = np.ones((np.int_(2 * r + 1), np.int_(2 * r + 1)))
    Yref = 0.299 * Ref_image[:, :, 0] + 0.587 * Ref_image[:, :, 1] + 0.114 * Ref_image[:, :, 2]
    Ypro = 0.299 * Pro_image[:, :, 0] + 0.587 * Pro_image[:, :, 1] + 0.114 * Pro_image[:, :, 2]
    DL, _ = MU.SSIM(Yref, Ypro, mode='same')
    dl = 0.
    # SE = np.ones((2, 2))
    YCbCr_ref = CAM.ycbcr(Ref_image)
    YCbCr_pro = CAM.ycbcr(Pro_image)
    ECbCr = np.sqrt(np.power(YCbCr_ref[:, :, 1] - YCbCr_pro[:, :, 1], 2) + \
                    np.power(YCbCr_ref[:, :, 2] - YCbCr_pro[:, :, 2], 2))
    LBP, _ = CT.lbp(Ref_image, th=th, r=r)
    list_of_patterns = np.unique(LBP)
    cd = 0.
    wpt = 0.
    CD = np.zeros(LBP.shape)
    Np = LBP.size
    for pp in list_of_patterns:
        homo_patches = LBP == pp
        homo_patches = ndimage.morphology.binary_closing(homo_patches, SE)
        patch_index, num_patches = ndimage.measurements.label(homo_patches)
        for ii in range(1, num_patches):
            idx = np.where(patch_index == ii)
            if idx[0].size > min_num_pixels:
                if sq:
                    chroma_spread = np.std(np.power(ECbCr[idx],2))
                    p = np.sort(np.array(np.power(ECbCr[idx],2)).ravel())[::-1]
                    chroma_extreme = np.mean(p[0:np.int_(np.ceil(p.size * 0.01))]) - p[np.int_(np.ceil(p.size * 0.01)) - 1]
                else:
                    chroma_spread = np.std(ECbCr[idx])
                    p = np.sort(np.array(ECbCr[idx]).ravel())[::-1]
                    chroma_extreme = np.mean(p[0:np.int_(np.ceil(p.size * 0.01))]) - p[np.int_(np.ceil(p.size * 0.01)) - 1]
                temp = 0.0192 * chroma_spread + 0.0076 * chroma_extreme
                templ = np.mean((1.-DL[idx])/2.)
                CD[idx] = 0.7*temp + 0.3*templ
                if np.isnan(temp):
                    pass
                wp = (1. * idx[0].size) / Np
                cd += wp * temp
                dl += wp * templ
                wpt += 1  # wp
    return CD, (0.7*cd + 0.3*dl)



def conf34_sprext_patches(Ref_image, Pro_image, th=0.045, r=2., min_num_pixels = 4, sq = False):
    SE = np.ones((np.int_(2 * r + 1), np.int_(2 * r + 1)))
    Yref = 0.299 * Ref_image[:, :, 0] + 0.587 * Ref_image[:, :, 1] + 0.114 * Ref_image[:, :, 2]
    Ypro = 0.299 * Pro_image[:, :, 0] + 0.587 * Pro_image[:, :, 1] + 0.114 * Pro_image[:, :, 2]
    DL, _ = MU.SSIM(Yref, Ypro, mode='same')
    dl = 0.
    # SE = np.ones((2, 2))
    YCbCr_ref = CAM.ycbcr(Ref_image)
    YCbCr_pro = CAM.ycbcr(Pro_image)
    ECbCr = np.sqrt(np.power(YCbCr_ref[:, :, 1] - YCbCr_pro[:, :, 1], 2) + \
                    np.power(YCbCr_ref[:, :, 2] - YCbCr_pro[:, :, 2], 2))
    LBP, _ = CT.lbp(Ref_image, th=th, r=r)
    list_of_patterns = np.unique(LBP)
    cd = 0.
    wpt = 0.
    CD = np.zeros(LBP.shape)
    Np = LBP.size
    for pp in list_of_patterns:
        homo_patches = LBP == pp
        homo_patches = ndimage.morphology.binary_closing(homo_patches, SE)
        patch_index, num_patches = ndimage.measurements.label(homo_patches)
        for ii in range(1, num_patches):
            idx = np.where(patch_index == ii)
            if idx[0].size > min_num_pixels:
                if sq:
                    chroma_spread = np.std(np.power(ECbCr[idx],2))
                    p = np.sort(np.array(np.power(ECbCr[idx],2)).ravel())[::-1]
                    chroma_extreme = np.mean(p[0:np.int_(np.ceil(p.size * 0.01))]) - p[np.int_(np.ceil(p.size * 0.01)) - 1]
                else:
                    chroma_spread = np.std(ECbCr[idx])
                    p = np.sort(np.array(ECbCr[idx]).ravel())[::-1]
                    chroma_extreme = np.mean(p[0:np.int_(np.ceil(p.size * 0.01))]) - p[np.int_(np.ceil(p.size * 0.01)) - 1]
                temp = 0.0192 * chroma_spread + 0.0076 * chroma_extreme
                templ = np.mean((1.-DL[idx])/2.)
                CD[idx] = 0.7*temp + 0.3*templ
                if np.isnan(temp):
                    pass
                wp = (1. * idx[0].size) / Np
                cd += wp * temp
                dl += wp * templ
                wpt += 1  # wp
    return CD, (0.7*cd + 0.3*dl)



def conf35_sprext_patches(Ref_image, Pro_image, th=0.045, r=3., min_num_pixels = 4, sq = False):
    SE = np.ones((np.int_(2 * r + 1), np.int_(2 * r + 1)))
    Yref = 0.299 * Ref_image[:, :, 0] + 0.587 * Ref_image[:, :, 1] + 0.114 * Ref_image[:, :, 2]
    Ypro = 0.299 * Pro_image[:, :, 0] + 0.587 * Pro_image[:, :, 1] + 0.114 * Pro_image[:, :, 2]
    DL, _ = MU.SSIM(Yref, Ypro, mode='same')
    dl = 0.
    # SE = np.ones((2, 2))
    YCbCr_ref = CAM.ycbcr(Ref_image)
    YCbCr_pro = CAM.ycbcr(Pro_image)
    ECbCr = np.sqrt(np.power(YCbCr_ref[:, :, 1] - YCbCr_pro[:, :, 1], 2) + \
                    np.power(YCbCr_ref[:, :, 2] - YCbCr_pro[:, :, 2], 2))
    LBP, _ = CT.lbp(Ref_image, th=th, r=r)
    list_of_patterns = np.unique(LBP)
    cd = 0.
    wpt = 0.
    CD = np.zeros(LBP.shape)
    Np = LBP.size
    for pp in list_of_patterns:
        homo_patches = LBP == pp
        homo_patches = ndimage.morphology.binary_closing(homo_patches, SE)
        patch_index, num_patches = ndimage.measurements.label(homo_patches)
        for ii in range(1, num_patches):
            idx = np.where(patch_index == ii)
            if idx[0].size > min_num_pixels:
                if sq:
                    chroma_spread = np.std(np.power(ECbCr[idx],2))
                    p = np.sort(np.array(np.power(ECbCr[idx],2)).ravel())[::-1]
                    chroma_extreme = np.mean(p[0:np.int_(np.ceil(p.size * 0.01))]) - p[np.int_(np.ceil(p.size * 0.01)) - 1]
                else:
                    chroma_spread = np.std(ECbCr[idx])
                    p = np.sort(np.array(ECbCr[idx]).ravel())[::-1]
                    chroma_extreme = np.mean(p[0:np.int_(np.ceil(p.size * 0.01))]) - p[np.int_(np.ceil(p.size * 0.01)) - 1]
                temp = 0.0192 * chroma_spread + 0.0076 * chroma_extreme
                templ = np.mean((1.-DL[idx])/2.)
                CD[idx] = 0.7*temp + 0.3*templ
                if np.isnan(temp):
                    pass
                wp = (1. * idx[0].size) / Np
                cd += wp * temp
                dl += wp * templ
                wpt += 1  # wp
    return CD, (0.7*cd + 0.3*dl)




def conf36_sprext_patches(Ref_image, Pro_image, th=0.05, r=1., min_num_pixels = 4, sq = False):
    SE = np.ones((np.int_(2 * r + 1), np.int_(2 * r + 1)))
    Yref = 0.299 * Ref_image[:, :, 0] + 0.587 * Ref_image[:, :, 1] + 0.114 * Ref_image[:, :, 2]
    Ypro = 0.299 * Pro_image[:, :, 0] + 0.587 * Pro_image[:, :, 1] + 0.114 * Pro_image[:, :, 2]
    DL, _ = MU.SSIM(Yref, Ypro, mode='same')
    dl = 0.
    # SE = np.ones((2, 2))
    YCbCr_ref = CAM.ycbcr(Ref_image)
    YCbCr_pro = CAM.ycbcr(Pro_image)
    ECbCr = np.sqrt(np.power(YCbCr_ref[:, :, 1] - YCbCr_pro[:, :, 1], 2) + \
                    np.power(YCbCr_ref[:, :, 2] - YCbCr_pro[:, :, 2], 2))
    LBP, _ = CT.lbp(Ref_image, th=th, r=r)
    list_of_patterns = np.unique(LBP)
    cd = 0.
    wpt = 0.
    CD = np.zeros(LBP.shape)
    Np = LBP.size
    for pp in list_of_patterns:
        homo_patches = LBP == pp
        homo_patches = ndimage.morphology.binary_closing(homo_patches, SE)
        patch_index, num_patches = ndimage.measurements.label(homo_patches)
        for ii in range(1, num_patches):
            idx = np.where(patch_index == ii)
            if idx[0].size > min_num_pixels:
                if sq:
                    chroma_spread = np.std(np.power(ECbCr[idx],2))
                    p = np.sort(np.array(np.power(ECbCr[idx],2)).ravel())[::-1]
                    chroma_extreme = np.mean(p[0:np.int_(np.ceil(p.size * 0.01))]) - p[np.int_(np.ceil(p.size * 0.01)) - 1]
                else:
                    chroma_spread = np.std(ECbCr[idx])
                    p = np.sort(np.array(ECbCr[idx]).ravel())[::-1]
                    chroma_extreme = np.mean(p[0:np.int_(np.ceil(p.size * 0.01))]) - p[np.int_(np.ceil(p.size * 0.01)) - 1]
                temp = 0.0192 * chroma_spread + 0.0076 * chroma_extreme
                templ = np.mean((1.-DL[idx])/2.)
                CD[idx] = 0.7*temp + 0.3*templ
                if np.isnan(temp):
                    pass
                wp = (1. * idx[0].size) / Np
                cd += wp * temp
                dl += wp * templ
                wpt += 1  # wp
    return CD, (0.7*cd + 0.3*dl)



def conf37_sprext_patches(Ref_image, Pro_image, th=0.05, r=1.5, min_num_pixels = 4, sq = False):
    SE = np.ones((np.int_(2 * r + 1), np.int_(2 * r + 1)))
    Yref = 0.299 * Ref_image[:, :, 0] + 0.587 * Ref_image[:, :, 1] + 0.114 * Ref_image[:, :, 2]
    Ypro = 0.299 * Pro_image[:, :, 0] + 0.587 * Pro_image[:, :, 1] + 0.114 * Pro_image[:, :, 2]
    DL, _ = MU.SSIM(Yref, Ypro, mode='same')
    dl = 0.
    # SE = np.ones((2, 2))
    YCbCr_ref = CAM.ycbcr(Ref_image)
    YCbCr_pro = CAM.ycbcr(Pro_image)
    ECbCr = np.sqrt(np.power(YCbCr_ref[:, :, 1] - YCbCr_pro[:, :, 1], 2) + \
                    np.power(YCbCr_ref[:, :, 2] - YCbCr_pro[:, :, 2], 2))
    LBP, _ = CT.lbp(Ref_image, th=th, r=r)
    list_of_patterns = np.unique(LBP)
    cd = 0.
    wpt = 0.
    CD = np.zeros(LBP.shape)
    Np = LBP.size
    for pp in list_of_patterns:
        homo_patches = LBP == pp
        homo_patches = ndimage.morphology.binary_closing(homo_patches, SE)
        patch_index, num_patches = ndimage.measurements.label(homo_patches)
        for ii in range(1, num_patches):
            idx = np.where(patch_index == ii)
            if idx[0].size > min_num_pixels:
                if sq:
                    chroma_spread = np.std(np.power(ECbCr[idx],2))
                    p = np.sort(np.array(np.power(ECbCr[idx],2)).ravel())[::-1]
                    chroma_extreme = np.mean(p[0:np.int_(np.ceil(p.size * 0.01))]) - p[np.int_(np.ceil(p.size * 0.01)) - 1]
                else:
                    chroma_spread = np.std(ECbCr[idx])
                    p = np.sort(np.array(ECbCr[idx]).ravel())[::-1]
                    chroma_extreme = np.mean(p[0:np.int_(np.ceil(p.size * 0.01))]) - p[np.int_(np.ceil(p.size * 0.01)) - 1]
                temp = 0.0192 * chroma_spread + 0.0076 * chroma_extreme
                templ = np.mean((1.-DL[idx])/2.)
                CD[idx] = 0.7*temp + 0.3*templ
                if np.isnan(temp):
                    pass
                wp = (1. * idx[0].size) / Np
                cd += wp * temp
                dl += wp * templ
                wpt += 1  # wp
    return CD, (0.7*cd + 0.3*dl)



def conf38_sprext_patches(Ref_image, Pro_image, th=0.05, r=2., min_num_pixels = 4, sq = False):
    SE = np.ones((np.int_(2 * r + 1), np.int_(2 * r + 1)))
    Yref = 0.299 * Ref_image[:, :, 0] + 0.587 * Ref_image[:, :, 1] + 0.114 * Ref_image[:, :, 2]
    Ypro = 0.299 * Pro_image[:, :, 0] + 0.587 * Pro_image[:, :, 1] + 0.114 * Pro_image[:, :, 2]
    DL, _ = MU.SSIM(Yref, Ypro, mode='same')
    dl = 0.
    # SE = np.ones((2, 2))
    YCbCr_ref = CAM.ycbcr(Ref_image)
    YCbCr_pro = CAM.ycbcr(Pro_image)
    ECbCr = np.sqrt(np.power(YCbCr_ref[:, :, 1] - YCbCr_pro[:, :, 1], 2) + \
                    np.power(YCbCr_ref[:, :, 2] - YCbCr_pro[:, :, 2], 2))
    LBP, _ = CT.lbp(Ref_image, th=th, r=r)
    list_of_patterns = np.unique(LBP)
    cd = 0.
    wpt = 0.
    CD = np.zeros(LBP.shape)
    Np = LBP.size
    for pp in list_of_patterns:
        homo_patches = LBP == pp
        homo_patches = ndimage.morphology.binary_closing(homo_patches, SE)
        patch_index, num_patches = ndimage.measurements.label(homo_patches)
        for ii in range(1, num_patches):
            idx = np.where(patch_index == ii)
            if idx[0].size > min_num_pixels:
                if sq:
                    chroma_spread = np.std(np.power(ECbCr[idx],2))
                    p = np.sort(np.array(np.power(ECbCr[idx],2)).ravel())[::-1]
                    chroma_extreme = np.mean(p[0:np.int_(np.ceil(p.size * 0.01))]) - p[np.int_(np.ceil(p.size * 0.01)) - 1]
                else:
                    chroma_spread = np.std(ECbCr[idx])
                    p = np.sort(np.array(ECbCr[idx]).ravel())[::-1]
                    chroma_extreme = np.mean(p[0:np.int_(np.ceil(p.size * 0.01))]) - p[np.int_(np.ceil(p.size * 0.01)) - 1]
                temp = 0.0192 * chroma_spread + 0.0076 * chroma_extreme
                templ = np.mean((1.-DL[idx])/2.)
                CD[idx] = 0.7*temp + 0.3*templ
                if np.isnan(temp):
                    pass
                wp = (1. * idx[0].size) / Np
                cd += wp * temp
                dl += wp * templ
                wpt += 1  # wp
    return CD, (0.7*cd + 0.3*dl)



def conf39_sprext_patches(Ref_image, Pro_image, th=0.05, r=3., min_num_pixels = 4, sq = False):
    SE = np.ones((np.int_(2 * r + 1), np.int_(2 * r + 1)))
    Yref = 0.299 * Ref_image[:, :, 0] + 0.587 * Ref_image[:, :, 1] + 0.114 * Ref_image[:, :, 2]
    Ypro = 0.299 * Pro_image[:, :, 0] + 0.587 * Pro_image[:, :, 1] + 0.114 * Pro_image[:, :, 2]
    DL, _ = MU.SSIM(Yref, Ypro, mode='same')
    dl = 0.
    # SE = np.ones((2, 2))
    YCbCr_ref = CAM.ycbcr(Ref_image)
    YCbCr_pro = CAM.ycbcr(Pro_image)
    ECbCr = np.sqrt(np.power(YCbCr_ref[:, :, 1] - YCbCr_pro[:, :, 1], 2) + \
                    np.power(YCbCr_ref[:, :, 2] - YCbCr_pro[:, :, 2], 2))
    LBP, _ = CT.lbp(Ref_image, th=th, r=r)
    list_of_patterns = np.unique(LBP)
    cd = 0.
    wpt = 0.
    CD = np.zeros(LBP.shape)
    Np = LBP.size
    for pp in list_of_patterns:
        homo_patches = LBP == pp
        homo_patches = ndimage.morphology.binary_closing(homo_patches, SE)
        patch_index, num_patches = ndimage.measurements.label(homo_patches)
        for ii in range(1, num_patches):
            idx = np.where(patch_index == ii)
            if idx[0].size > min_num_pixels:
                if sq:
                    chroma_spread = np.std(np.power(ECbCr[idx],2))
                    p = np.sort(np.array(np.power(ECbCr[idx],2)).ravel())[::-1]
                    chroma_extreme = np.mean(p[0:np.int_(np.ceil(p.size * 0.01))]) - p[np.int_(np.ceil(p.size * 0.01)) - 1]
                else:
                    chroma_spread = np.std(ECbCr[idx])
                    p = np.sort(np.array(ECbCr[idx]).ravel())[::-1]
                    chroma_extreme = np.mean(p[0:np.int_(np.ceil(p.size * 0.01))]) - p[np.int_(np.ceil(p.size * 0.01)) - 1]
                temp = 0.0192 * chroma_spread + 0.0076 * chroma_extreme
                templ = np.mean((1.-DL[idx])/2.)
                CD[idx] = 0.7*temp + 0.3*templ
                if np.isnan(temp):
                    pass
                wp = (1. * idx[0].size) / Np
                cd += wp * temp
                dl += wp * templ
                wpt += 1  # wp
    return CD, (0.7*cd + 0.3*dl)
