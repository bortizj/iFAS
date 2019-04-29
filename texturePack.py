#!/usr/bin/env python2.7
import numpy as np
import computeTexture
import myUtilities


def td00_ar2d(Ref_image, Pro_image):
    Iar_ref, ar_ref = computeTexture.ar2d(Ref_image,r=1.5)
    Iar_pro, ar_pro = computeTexture.ar2d(Pro_image,r=1.5)
    return Iar_ref-Iar_pro, np.sqrt(np.sum(np.power(ar_ref-ar_pro,2)))


def td01_autocorr2d(Ref_image, Pro_image):
    Iar_ref, ar_ref = computeTexture.autocorr2d(Ref_image)
    Iar_pro, ar_pro = computeTexture.autocorr2d(Pro_image)
    return Iar_ref-Iar_pro, np.sqrt(np.sum(np.power(ar_ref-ar_pro,2)))


def td02_coomatrix(Ref_image, Pro_image):
    Iar_ref, ar_ref = computeTexture.coomatrix(Ref_image)
    Iar_pro, ar_pro = computeTexture.coomatrix(Pro_image)
    return Iar_ref-Iar_pro, np.sqrt(np.sum(np.power(ar_ref-ar_pro,2)))


def td03_dwtenergy(Ref_image, Pro_image):
    Iar_ref, ar_ref = computeTexture.dwtenergy(Ref_image)
    Iar_pro, ar_pro = computeTexture.dwtenergy(Pro_image)
    return Iar_ref-Iar_pro, np.sqrt(np.sum(np.power(ar_ref-ar_pro,2)))


def td04_eigenfilter(Ref_image, Pro_image):
    Iar_ref, ar_ref = computeTexture.eigenfilter(Ref_image)
    Iar_pro, ar_pro = computeTexture.eigenfilter(Pro_image)
    return np.sum(np.abs(Iar_ref-Iar_pro),2), np.sqrt(np.sum(np.power(ar_ref-ar_pro,2)))


def td05_gabor(Ref_image, Pro_image):
    YG_ref, hist_ref = computeTexture.gabor_features(Ref_image)
    YG_pro, hist_pro = computeTexture.gabor_features(Pro_image)
    flag = False
    count = 0
    for ii in YG_ref:
        for jj in YG_ref[ii]:
            if not flag:
                Diff = np.zeros_like(YG_ref[ii][jj])
                flag = True
            Diff += np.abs(YG_ref[ii][jj] - YG_pro[ii][jj])
            count+=1
    Diff /= count
    return Diff, myUtilities.KullbackLeiblerDivergence(hist_ref,hist_pro)


def td06_gmrf(Ref_image, Pro_image):
    Igmrf_ref, alpha_ref = computeTexture.gmrf(Ref_image)
    Igmrf_pro, alpha_pro = computeTexture.gmrf(Pro_image)
    return Igmrf_ref-Igmrf_pro, np.sqrt(np.sum(np.power(alpha_ref-alpha_pro,2)))


def td07_granulometrymoments(Ref_image, Pro_image):
    YG_ref, alpha_ref = computeTexture.granulometrymoments(Ref_image)
    YG_pro, alpha_pro = computeTexture.granulometrymoments(Pro_image)
    Diff = np.zeros_like(YG_ref[0][:,:,0])
    for ii in YG_ref:
        Diff += np.abs(YG_ref[ii][:,:,0]-YG_pro[ii][:,:,0])+np.abs(YG_ref[ii][:,:,1]-YG_pro[ii][:,:,1])
    Diff /= len(YG_ref)
    return Diff, np.sqrt(np.sum(np.power(alpha_ref-alpha_pro,2)))


def td08_laplacianpyramid(Ref_image, Pro_image):
    YG_ref, alpha_ref = computeTexture.laplacianpyramid(Ref_image)
    YG_pro, alpha_pro = computeTexture.laplacianpyramid(Pro_image)
    return YG_ref[0]-YG_pro[0], np.sqrt(np.sum(np.power(alpha_ref-alpha_pro,2)))


def td09_lawsoperators(Ref_image, Pro_image):
    YG_ref, alpha_ref = computeTexture.lawsoperators(Ref_image)
    YG_pro, alpha_pro = computeTexture.lawsoperators(Pro_image)
    return np.sum(np.abs(YG_ref-YG_pro),2), np.sqrt(np.sum(np.power(alpha_ref-alpha_pro,2)))


def td10_lbp(Ref_image, Pro_image):
    Ilbp_ref, hist_ref = computeTexture.lbp(Ref_image, r=1.5, color_=False,th=0.01)
    Ilbp_pro, hist_pro = computeTexture.lbp(Pro_image, r=1.5, color_=False,th=0.01)
    return np.double(Ilbp_ref)-Ilbp_pro, myUtilities.KullbackLeiblerDivergence(hist_ref,hist_pro)


def td11_steerablepyramid(Ref_image, Pro_image):
    YG_ref, alpha_ref = computeTexture.steerablepyramid(Ref_image)
    YG_pro, alpha_pro = computeTexture.steerablepyramid(Pro_image)
    return YG_ref[0][0]-YG_pro[0][0], np.sqrt(np.sum(np.power(alpha_ref-alpha_pro,2)))


def td12_power_spectrum_fft(Ref_image, Pro_image):
    Ipsfft_ref, psfft_ref = computeTexture.power_spectrum_fft(Ref_image)
    Ipsfft_pro, psfft_pro = computeTexture.power_spectrum_fft(Pro_image)
    flag = False
    for ii in range(len(Ipsfft_ref)):
        if not flag:
            Diff = np.zeros_like(Ipsfft_ref[ii])
            flag = True
        Diff += np.abs(Ipsfft_ref[ii]-Ipsfft_pro[ii])
    Diff /= len(Ipsfft_ref)
    return Diff, np.sqrt(np.sum(np.power(psfft_ref-psfft_pro,2)))


def td13_wigner_distribution(Ref_image, Pro_image):
    PWD_ref, hist_ref = computeTexture.wigner_distribution(Ref_image)
    PWD_pro, hist_pro = computeTexture.wigner_distribution(Pro_image)
    flag = False
    for ii in range(PWD_ref.shape[2]):
        for jj in range(PWD_ref.shape[3]):
            if not flag:
                Diff = np.zeros_like(PWD_ref[:,:,ii,jj])
                flag = True
            Diff += np.abs(PWD_ref[:,:,ii,jj] - PWD_pro[:,:,ii,jj])
    Diff /= (PWD_ref.shape[2]*PWD_ref.shape[3])
    return Diff, myUtilities.KullbackLeiblerDivergence(hist_ref,hist_pro)
