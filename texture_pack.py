#!/usr/bin/env python2.7
import my_utilities as MU
import numpy as np
from scipy import ndimage
import compute_texture as CT

def ar2d_difference(Ref_image, Pro_image):
    Iar_ref, ar_ref = CT.ar2d(Ref_image,r=1.5)
    Iar_pro, ar_pro = CT.ar2d(Pro_image,r=1.5)
    return Iar_ref-Iar_pro, np.sqrt(np.sum(np.power(ar_ref-ar_pro,2)))


def autocorr2d_difference(Ref_image, Pro_image):
    Iar_ref, ar_ref = CT.autocorr2d(Ref_image)
    Iar_pro, ar_pro = CT.autocorr2d(Pro_image)
    return Iar_ref-Iar_pro, np.sqrt(np.sum(np.power(ar_ref-ar_pro,2)))


def coomatrix_difference(Ref_image, Pro_image):
    Iar_ref, ar_ref = CT.coomatrix(Ref_image)
    Iar_pro, ar_pro = CT.coomatrix(Pro_image)
    return Iar_ref-Iar_pro, np.sqrt(np.sum(np.power(ar_ref-ar_pro,2)))


def dwtenergy_difference(Ref_image, Pro_image):
    Iar_ref, ar_ref = CT.dwtenergy(Ref_image)
    Iar_pro, ar_pro = CT.dwtenergy(Pro_image)
    return Iar_ref-Iar_pro, np.sqrt(np.sum(np.power(ar_ref-ar_pro,2)))


def eigenfilter_difference(Ref_image, Pro_image):
    Iar_ref, ar_ref = CT.eigenfilter(Ref_image)
    Iar_pro, ar_pro = CT.eigenfilter(Pro_image)
    return np.sum(np.abs(Iar_ref-Iar_pro),2), np.sqrt(np.sum(np.power(ar_ref-ar_pro,2)))


def gabor_features_difference(Ref_image, Pro_image):
    YG_ref, hist_ref = CT.gabor_features(Ref_image)
    YG_pro, hist_pro = CT.gabor_features(Pro_image)
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
    return Diff, MU.KullbackLeiblerDivergence(hist_ref,hist_pro)


def gmrf_difference(Ref_image, Pro_image):
    Igmrf_ref, alpha_ref = CT.gmrf(Ref_image)
    Igmrf_pro, alpha_pro = CT.gmrf(Pro_image)
    return Igmrf_ref-Igmrf_pro, np.sqrt(np.sum(np.power(alpha_ref-alpha_pro,2)))


def granulometrymoments_difference(Ref_image, Pro_image):
    YG_ref, alpha_ref = CT.granulometrymoments(Ref_image)
    YG_pro, alpha_pro = CT.granulometrymoments(Pro_image)
    Diff = np.zeros_like(YG_ref[0][:,:,0])
    for ii in YG_ref:
        Diff += np.abs(YG_ref[ii][:,:,0]-YG_pro[ii][:,:,0])+np.abs(YG_ref[ii][:,:,1]-YG_pro[ii][:,:,1])
    Diff /= len(YG_ref)
    return Diff, np.sqrt(np.sum(np.power(alpha_ref-alpha_pro,2)))


def laplacianpyramid_difference(Ref_image, Pro_image):
    YG_ref, alpha_ref = CT.laplacianpyramid(Ref_image)
    YG_pro, alpha_pro = CT.laplacianpyramid(Pro_image)
    return YG_ref[0]-YG_pro[0], np.sqrt(np.sum(np.power(alpha_ref-alpha_pro,2)))


def lawsoperators_difference(Ref_image, Pro_image):
    YG_ref, alpha_ref = CT.lawsoperators(Ref_image)
    YG_pro, alpha_pro = CT.lawsoperators(Pro_image)
    return np.sum(np.abs(YG_ref-YG_pro),2), np.sqrt(np.sum(np.power(alpha_ref-alpha_pro,2)))


def lbp_difference_gray(Ref_image, Pro_image):
    Ilbp_ref, hist_ref = CT.lbp(Ref_image, r=1.5, color_=False,th=0.01)
    Ilbp_pro, hist_pro = CT.lbp(Pro_image, r=1.5, color_=False,th=0.01)
    return np.double(Ilbp_ref)-Ilbp_pro, MU.KullbackLeiblerDivergence(hist_ref,hist_pro)


def steerablepyramid_difference(Ref_image, Pro_image):
    YG_ref, alpha_ref = CT.steerablepyramid(Ref_image)
    YG_pro, alpha_pro = CT.steerablepyramid(Pro_image)
    return YG_ref[0][0]-YG_pro[0][0], np.sqrt(np.sum(np.power(alpha_ref-alpha_pro,2)))


def power_spectrum_fft_difference(Ref_image, Pro_image):
    Ipsfft_ref, psfft_ref = CT.power_spectrum_fft(Ref_image)
    Ipsfft_pro, psfft_pro = CT.power_spectrum_fft(Pro_image)
    flag = False
    for ii in range(len(Ipsfft_ref)):
        if not flag:
            Diff = np.zeros_like(Ipsfft_ref[ii])
            flag = True
        Diff += np.abs(Ipsfft_ref[ii]-Ipsfft_pro[ii])
    Diff /= len(Ipsfft_ref)
    return Diff, np.sqrt(np.sum(np.power(psfft_ref-psfft_pro,2)))


def wigner_distribution_difference(Ref_image, Pro_image):
    PWD_ref, hist_ref = CT.wigner_distribution(Ref_image)
    PWD_pro, hist_pro = CT.wigner_distribution(Pro_image)
    flag = False
    for ii in range(PWD_ref.shape[2]):
        for jj in range(PWD_ref.shape[3]):
            if not flag:
                Diff = np.zeros_like(PWD_ref[:,:,ii,jj])
                flag = True
            Diff += np.abs(PWD_ref[:,:,ii,jj] - PWD_pro[:,:,ii,jj])
    Diff /= (PWD_ref.shape[2]*PWD_ref.shape[3])
    return Diff, MU.KullbackLeiblerDivergence(hist_ref,hist_pro)


# def lbp_difference_color(Ref_image, Pro_image):
#     Ilbp_ref, hist_ref = CT.lbp(Ref_image, r=1.5, color_=True,th=3)
#     Ilbp_pro, hist_pro = CT.lbp(Pro_image, r=1.5, color_=True,th=3)
#     return np.double(Ilbp_ref)-Ilbp_pro, MU.KullbackLeiblerDivergence(hist_ref,hist_pro)

if __name__ == "__main__":
    MUltimedia_file_ref = './sample_images/test_ref_0.bmp'
    image_ref = ndimage.imread(MUltimedia_file_ref)
    MUltimedia_file_pro = './sample_images/test_pro_0.bmp'
    image_pro = ndimage.imread(MUltimedia_file_pro)
    Ydiff, c = granulometrymoments_difference(image_ref, image_pro)
    print c
    plt.imshow(Ydiff)
    plt.colorbar()
    plt.show()
