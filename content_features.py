#!/usr/bin/env python2.7
# Importing necessary packages
import my_utilities as MU
from scipy import signal
import numpy as np
from skimage import feature
from skimage import color
import compute_texture as CT
from scipy import ndimage

def spatial_activity(img):
    tempHS = np.array([-0.0052625, -0.0173445, -0.0427401, -0.0768961, -0.0957739, -0.0696751, \
                       0, 0.0696751, 0.0957739, 0.0768961, 0.0427401, 0.0173445, 0.0052625])
    HS = np.tile(tempHS, (tempHS.size, 1))
    Yref = MU.checkifRGB(img)
    Gx = signal.convolve2d(Yref, np.rot90(HS, 2), mode='same')
    Gy = signal.convolve2d(Yref, np.transpose(np.rot90(HS, 2)), mode='same')
    SI13 = np.sqrt(Gx*Gx + Gy*Gy)
    return np.mean(np.asarray(SI13).ravel())

def texture_extent(img):
    Yref = MU.checkifRGB(img)
    GLCM = feature.greycomatrix(np.int8(0.25 * Yref), [1], [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4], levels=64)
    GLCM_ = np.sum(GLCM, 3)
    GLCM_ = np.sum(GLCM_, 2) / 4.
    for ii in range(GLCM.shape[3]):
        for jj in range(GLCM.shape[2]):
            GLCM[:, :, jj, ii] = GLCM_
    feature_temp = feature.greycoprops(GLCM, prop='contrast')
    return feature_temp[0, 0] / np.sum(GLCM_)


def principal_color(img, th=0.02, r=1., min_num_pixels = 4):
    SE = np.ones((np.int_(2 * r + 1), np.int_(2 * r + 1)))
    Lab_ref = color.rgb2lab(img)
    L_ref = Lab_ref[:, :, 0]
    a_ref = Lab_ref[:, :, 1]
    b_ref = Lab_ref[:, :, 2]
    LBP, _ = CT.lbp(img, th=th, r=r)
    list_of_patterns = np.unique(LBP)
    L_avg = np.zeros(LBP.shape)
    a_avg = np.zeros(LBP.shape)
    b_avg = np.zeros(LBP.shape)
    WD = np.zeros(LBP.shape)
    Np = LBP.size
    for pp in list_of_patterns:
        homo_patches = LBP == pp
        homo_patches = ndimage.morphology.binary_closing(homo_patches, SE)
        patch_index, num_patches = ndimage.measurements.label(homo_patches)
        for ii in range(1, num_patches):
            idx = np.where(patch_index == ii)
            if idx[0].size > min_num_pixels:
                L_avg[idx] = np.mean(L_ref[idx])
                a_avg[idx] = np.mean(a_ref[idx])
                b_avg[idx] = np.mean(b_ref[idx])
                wp = (1. * idx[0].size) / Np
                WD[idx] = wp
    Lab_avg = np.dstack((L_avg, a_avg, b_avg))
    H_1, edges = MU.colorhistogramplusedeges(Lab_avg, minc=np.array([np.min(L_avg), np.min(a_avg), np.min(b_avg)]),\
                         maxc=np.array([np.max(L_avg), np.max(a_avg), np.max(b_avg)]), nbins=8)
    H_1 = H_1/np.sum(H_1)
    ab_hist = np.reshape(np.mean(H_1, axis=0), (8, 8))
    img_avg = color.lab2rgb(Lab_avg)
    prixx, priyy, prizz = np.unravel_index(np.argmax(H_1), H_1.shape)
    for ii in range(3):
        edges[ii] = (edges[ii][:-1]+edges[ii][1:])/2.
    avg_color = color.lab2rgb(np.dstack((edges[0][prixx], edges[1][priyy], edges[2][prizz])))
    #The color features
    avg_cielch = color.lab2lch(np.dstack((edges[0][prixx], edges[1][priyy], edges[2][prizz])))
    E3d = MU.entropy3d(H_1, edges)
    TV = np.var(L_avg) + np.var(a_avg) + np.var(b_avg)
    return avg_color



if __name__ == "__main__":
    img = ndimage.imread('/media/bortiz/Data/PhD_Thesis_files/Experiments/Chapt6-color/Tid2013Color/i15.bmp')
    principal_color(img, th=0.02, r=1., min_num_pixels=9)
