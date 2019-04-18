#!/usr/bin/env python2.7
import my_utilities as MU
import numpy as np
import cython_functions
import matplotlib.pyplot as plt

def SME_difference(Ref_image, Pro_image):
    C_ref, sme_ref = MU.SME(Ref_image)
    C_pro, sme_pro = MU.SME(Pro_image)
    return C_ref-C_pro, sme_ref-sme_pro#np.abs(sme_ref-sme_pro)


def WME_difference(Ref_image, Pro_image):
    C_ref, wme_ref = MU.WME(Ref_image)
    C_pro, wme_pro = MU.WME(Pro_image)
    return C_ref-C_pro, wme_ref-wme_pro#np.abs(wme_ref-wme_pro)


def MME_difference(Ref_image, Pro_image):
    C_ref, mme_ref = MU.MME(Ref_image)
    C_pro, mme_pro = MU.MME(Pro_image)
    return C_ref-C_pro, mme_ref-mme_pro#np.abs(mme_ref-mme_pro)


def RMS_difference(Ref_image, Pro_image):
    C_ref, rme_ref = MU.RME(Ref_image)
    C_pro, rme_pro = MU.RME(Pro_image)
    return C_ref-C_pro, rme_ref-rme_pro#np.abs(rme_ref-rme_pro)


def PWC_contrast_difference(Ref_image, Pro_image):
    C_ref, rme_ref = MU.contrast_peli(Ref_image)
    C_pro, rme_pro = MU.contrast_peli(Pro_image)
    return C_ref-C_pro, rme_ref-rme_pro#np.abs(rme_ref-rme_pro)


def MSRMS_contrast_difference(Ref_image, Pro_image):
    C_ref, rme_ref = MU.contrast_rizzi(Ref_image)
    C_pro, rme_pro = MU.contrast_rizzi(Pro_image)
    return C_ref-C_pro, rme_ref-rme_pro#np.abs(rme_ref-rme_pro)


def CMMC_michelsonharmonicp75w15o02_michelsonlocal_contrast_bcontent_measure_difference(Ref_image, Pro_image, sizeblk = 15, min_fb = 3, overlap = 3):
    Yref = MU.checkifRGB(Ref_image)
    C_ref = cython_functions.local_contrast_bcontent_measure_michelson(Yref,np.array([sizeblk,sizeblk]),min_fb,overlap)
    C_ref_temp = C_ref[sizeblk//2:C_ref.shape[0]-sizeblk//2:overlap,sizeblk//2:C_ref.shape[1]-sizeblk//2:overlap]
    maxCref = np.max(C_ref_temp)
    e_ref = 1./np.mean(1./C_ref_temp[C_ref_temp>0.75*maxCref])
    if np.isnan(e_ref):
        e_ref = np.mean(C_ref_temp[C_ref_temp>-1])
    Ypro = MU.checkifRGB(Pro_image)
    C_pro = cython_functions.local_contrast_bcontent_measure_michelson(Ypro,np.array([sizeblk,sizeblk]),min_fb,overlap)
    C_pro_temp = C_pro[sizeblk//2:C_pro.shape[0]-sizeblk//2:overlap,sizeblk//2:C_pro.shape[1]-sizeblk//2:overlap]
    maxCpro = np.max(C_pro_temp)
    e_pro = 1./np.mean(1./C_pro_temp[C_pro_temp>0.75*maxCpro])
    if np.isnan(e_pro):
        e_pro = np.mean(C_pro_temp[C_pro_temp>-1])
    C_diff = C_ref - C_pro
    return C_diff, e_ref-e_pro


def CWMC_weberharmonicp75w15o02_weberlocal_contrast_bcontent_measure_difference(Ref_image, Pro_image, sizeblk = 15, min_fb = 3, overlap = 3):
    Yref = MU.checkifRGB(Ref_image)
    C_ref = cython_functions.local_contrast_bcontent_measure_weber(Yref,np.array([sizeblk,sizeblk]),min_fb,overlap)
    C_ref_temp = C_ref[sizeblk//2:C_ref.shape[0]-sizeblk//2:overlap,sizeblk//2:C_ref.shape[1]-sizeblk//2:overlap]
    maxCref = np.max(C_ref_temp)
    e_ref = 1./np.mean(1./C_ref_temp[C_ref_temp>0.75*maxCref])
    if np.isnan(e_ref):
        e_ref = np.mean(C_ref_temp[C_ref_temp>-1])
    Ypro = MU.checkifRGB(Pro_image)
    C_pro = cython_functions.local_contrast_bcontent_measure_weber(Ypro,np.array([sizeblk,sizeblk]),min_fb,overlap)
    C_pro_temp = C_pro[sizeblk//2:C_pro.shape[0]-sizeblk//2:overlap,sizeblk//2:C_pro.shape[1]-sizeblk//2:overlap]
    maxCpro = np.max(C_pro_temp)
    e_pro = 1./np.mean(1./C_pro_temp[C_pro_temp>0.75*maxCpro])
    if np.isnan(e_pro):
        e_pro = np.mean(C_pro_temp[C_pro_temp>-1])
    C_diff = C_ref - C_pro
    return C_diff, e_ref-e_pro
