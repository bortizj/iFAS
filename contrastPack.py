#!/usr/bin/env python2.7
import matplotlib.pyplot as plt
import numpy as np
import cythonFunctions
import computeContrast
import myUtilities


def SME_difference(Ref_image, Pro_image):
    C_ref, sme_ref = computeContrast.SME(Ref_image)
    C_pro, sme_pro = computeContrast.SME(Pro_image)
    return C_ref-C_pro, sme_ref-sme_pro#np.abs(sme_ref-sme_pro)


def WME_difference(Ref_image, Pro_image):
    C_ref, wme_ref = computeContrast.WME(Ref_image)
    C_pro, wme_pro = computeContrast.WME(Pro_image)
    return C_ref-C_pro, wme_ref-wme_pro#np.abs(wme_ref-wme_pro)


def MME_difference(Ref_image, Pro_image):
    C_ref, mme_ref = computeContrast.MME(Ref_image)
    C_pro, mme_pro = computeContrast.MME(Pro_image)
    return C_ref-C_pro, mme_ref-mme_pro#np.abs(mme_ref-mme_pro)


def RMS_difference(Ref_image, Pro_image):
    C_ref, rme_ref = computeContrast.RME(Ref_image)
    C_pro, rme_pro = computeContrast.RME(Pro_image)
    return C_ref-C_pro, rme_ref-rme_pro#np.abs(rme_ref-rme_pro)


def PWC_difference(Ref_image, Pro_image):
    C_ref, rme_ref = computeContrast.contrast_peli(Ref_image)
    C_pro, rme_pro = computeContrast.contrast_peli(Pro_image)
    return C_ref-C_pro, rme_ref-rme_pro#np.abs(rme_ref-rme_pro)


def MSRMS_difference(Ref_image, Pro_image):
    C_ref, rme_ref = computeContrast.contrast_rizzi(Ref_image)
    C_pro, rme_pro = computeContrast.contrast_rizzi(Pro_image)
    return C_ref-C_pro, rme_ref-rme_pro#np.abs(rme_ref-rme_pro)


def CMMC_difference(Ref_image, Pro_image, sizeblk=15, min_fb=3, overlap=3):
    Yref = myUtilities.checkifRGB(Ref_image)
    C_ref = cythonFunctions.local_contrast_bcontent_measure_michelson(Yref,np.array([sizeblk,sizeblk]),min_fb,overlap)
    C_ref_temp = C_ref[sizeblk//2:C_ref.shape[0]-sizeblk//2:overlap,sizeblk//2:C_ref.shape[1]-sizeblk//2:overlap]
    maxCref = np.max(C_ref_temp)
    e_ref = 1./np.mean(1./C_ref_temp[C_ref_temp>0.75*maxCref])
    if np.isnan(e_ref):
        e_ref = np.mean(C_ref_temp[C_ref_temp>-1])
    Ypro = myUtilities.checkifRGB(Pro_image)
    C_pro = cythonFunctions.local_contrast_bcontent_measure_michelson(Ypro,np.array([sizeblk,sizeblk]),min_fb,overlap)
    C_pro_temp = C_pro[sizeblk//2:C_pro.shape[0]-sizeblk//2:overlap,sizeblk//2:C_pro.shape[1]-sizeblk//2:overlap]
    maxCpro = np.max(C_pro_temp)
    e_pro = 1./np.mean(1./C_pro_temp[C_pro_temp>0.75*maxCpro])
    if np.isnan(e_pro):
        e_pro = np.mean(C_pro_temp[C_pro_temp>-1])
    C_diff = C_ref - C_pro
    return C_diff, e_ref-e_pro


def CWMC_difference(Ref_image, Pro_image, sizeblk=15, min_fb=3, overlap=3):
    Yref = myUtilities.checkifRGB(Ref_image)
    C_ref = cythonFunctions.local_contrast_bcontent_measure_weber(Yref,np.array([sizeblk,sizeblk]),min_fb,overlap)
    C_ref_temp = C_ref[sizeblk//2:C_ref.shape[0]-sizeblk//2:overlap,sizeblk//2:C_ref.shape[1]-sizeblk//2:overlap]
    maxCref = np.max(C_ref_temp)
    e_ref = 1./np.mean(1./C_ref_temp[C_ref_temp>0.75*maxCref])
    if np.isnan(e_ref):
        e_ref = np.mean(C_ref_temp[C_ref_temp>-1])
    Ypro = myUtilities.checkifRGB(Pro_image)
    C_pro = cythonFunctions.local_contrast_bcontent_measure_weber(Ypro,np.array([sizeblk,sizeblk]),min_fb,overlap)
    C_pro_temp = C_pro[sizeblk//2:C_pro.shape[0]-sizeblk//2:overlap,sizeblk//2:C_pro.shape[1]-sizeblk//2:overlap]
    maxCpro = np.max(C_pro_temp)
    e_pro = 1./np.mean(1./C_pro_temp[C_pro_temp>0.75*maxCpro])
    if np.isnan(e_pro):
        e_pro = np.mean(C_pro_temp[C_pro_temp>-1])
    C_diff = C_ref - C_pro
    return C_diff, e_ref-e_pro

