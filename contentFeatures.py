#!/usr/bin/env python2.7
# Importing necessary packages
from scipy import signal
import numpy as np
from skimage import feature
from skimage import color
from scipy import ndimage
import myUtilities
import computeTexture



def spatialActivity(img):
    tempHS = np.array([-0.0052625, -0.0173445, -0.0427401, -0.0768961, -0.0957739, -0.0696751, \
                       0, 0.0696751, 0.0957739, 0.0768961, 0.0427401, 0.0173445, 0.0052625])
    HS = np.tile(tempHS, (tempHS.size, 1))
    grayImg = myUtilities.checkifRGB(img)
    Gx = signal.convolve2d(grayImg, np.rot90(HS, 2), mode='same')
    Gy = signal.convolve2d(grayImg, np.transpose(np.rot90(HS, 2)), mode='same')
    SI13 = np.sqrt(Gx * Gx + Gy * Gy)
    return np.mean(np.asarray(SI13).ravel())


def textureExtent(img):
    grayImg = myUtilities.checkifRGB(img)
    GLCM = feature.greycomatrix(np.int8(0.25 * grayImg), [1], [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4], levels=64)
    GLCM_ = np.sum(GLCM, 3)
    GLCM_ = np.sum(GLCM_, 2) / 4.
    for ii in range(GLCM.shape[3]):
        for jj in range(GLCM.shape[2]):
            GLCM[:, :, jj, ii] = GLCM_
    feature_temp = feature.greycoprops(GLCM, prop='contrast')
    return feature_temp[0, 0] / np.sum(GLCM_)


def TVColor(img, th=0.02, r=1., min_num_pixels = 4):
    SE = np.ones((np.int_(2 * r + 1), np.int_(2 * r + 1)))
    Lab_ref = color.rgb2lab(img)
    L_ref = Lab_ref[:, :, 0]
    a_ref = Lab_ref[:, :, 1]
    b_ref = Lab_ref[:, :, 2]
    LBP, _ = computeTexture.lbp(img, th=th, r=r)
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
    H_1, edges = myUtilities.colorhistogramplusedeges(Lab_avg, nbins=8,\
                                                      minc=np.array([np.min(L_avg), np.min(a_avg), np.min(b_avg)]),
                                                      maxc=np.array([np.max(L_avg), np.max(a_avg), np.max(b_avg)]))
    H_1 = H_1 / np.sum(H_1)
    ab_hist = np.reshape(np.mean(H_1, axis=0), (8, 8))
    img_avg = color.lab2rgb(Lab_avg)
    prixx, priyy, prizz = np.unravel_index(np.argmax(H_1), H_1.shape)
    for ii in range(3):
        edges[ii] = (edges[ii][:-1]+edges[ii][1:])/2.
    avg_color = color.lab2rgb(np.dstack((edges[0][prixx], edges[1][priyy], edges[2][prizz])))
    #The color features only the avrege color is returned by iFAS
    avg_cielch = color.lab2lch(np.dstack((edges[0][prixx], edges[1][priyy], edges[2][prizz])))
    E3d = myUtilities.entropy3d(H_1, edges)
    TV = np.var(L_avg) + np.var(a_avg) + np.var(b_avg)
    return TV


def colorfulness(RGB, sizewin=3, type='Hasler'): #type = 'Hasler', 'Gao', or 'Panetta'
    K = 2
    window = np.ones((sizewin, sizewin))
    RGB = np.double(RGB)
    alpha = RGB[:, :, 0] - RGB[:, :, 1]
    beta = 0.5 * (RGB[:, :, 0] + RGB[:, :, 1]) - RGB[:, :, 2]
    mu_alpha = np.mean(alpha)
    mu_beta = np.mean(beta)
    sigma_alpha_sq = np.var(alpha)
    sigma_beta_sq = np.var(beta)
    Mu_alpha = signal.convolve2d(alpha, np.rot90(window/window.size,2), mode='valid')
    Mu_beta = signal.convolve2d(beta, np.rot90(window/window.size,2), mode='valid')
    Mu_alpha_sq = np.power(Mu_alpha, 2)
    Mu_beta_sq = np.power(Mu_beta, 2)
    Sigma_alpha_sq = signal.convolve2d(alpha*alpha, np.rot90(window/window.size,2), mode='valid') - Mu_alpha_sq
    Sigma_beta_sq = signal.convolve2d(beta*beta, np.rot90(window/window.size,2), mode='valid') - Mu_beta_sq
    if type == 'Gao':
        c = 0.02 * np.log(sigma_alpha_sq / (np.power(np.abs(mu_alpha), 0.2) + K) + K) *\
            np.log(sigma_beta_sq / (np.power(np.abs(mu_beta), 0.2) + K) + K)
        C = 0.02 * np.log(Sigma_alpha_sq / (np.power(np.abs(Mu_alpha), 0.2) + K) + K) *\
            np.log(Sigma_beta_sq / (np.power(np.abs(Mu_beta), 0.2) + K) + K)
    elif type == 'Panetta':
            mu_gamma = (mu_alpha + mu_beta) / 2
            mu_gamma_sq = mu_gamma ** 2
            sigma_gamma_sq = (np.mean(np.power(alpha, 2) - mu_gamma_sq) + np.mean(np.power(beta, 2) - mu_gamma_sq)) / 2
            Mu_gamma = (Mu_alpha + Mu_beta) / 2
            Mu_gamma_sq = np.power(Mu_gamma, 2)
            Sigma_alpha_sq_c = signal.convolve2d(alpha * alpha, np.rot90(window/window.size, 2), mode='valid') -\
                               Mu_gamma_sq
            Sigma_beta_sq_c = signal.convolve2d(beta * beta, np.rot90(window/window.size, 2), mode='valid') -\
                              Mu_gamma_sq
            Sigma_gamma_sq = (Sigma_alpha_sq_c + Sigma_beta_sq_c) / 2
            c = 0.02 * ((np.log(sigma_alpha_sq + K) * np.log(sigma_beta_sq + K)) / np.log(sigma_gamma_sq + K)) *\
                ((np.log(mu_alpha ** 2 + K) * np.log(mu_beta ** 2 + K)) / np.log(mu_gamma_sq + K))
            C = 0.02 * ((np.log(Sigma_alpha_sq + K) * np.log(Sigma_beta_sq + K)) / np.log(Sigma_gamma_sq + K)) *\
                ((np.log(Mu_alpha_sq + K) * np.log(Mu_beta_sq + K)) / np.log(Mu_gamma_sq + K))
    else:
        c = np.sqrt(sigma_alpha_sq + sigma_beta_sq) + 0.3 * np.sqrt(mu_alpha ** 2 + mu_beta ** 2)
        C = np.sqrt(np.abs(Sigma_alpha_sq) + np.abs(Sigma_beta_sq)) + 0.3 * np.sqrt(Mu_alpha_sq + Mu_beta_sq)
    C[np.isnan(C)] = 0
    C[np.isinf(C)] = 0
    return C, c
