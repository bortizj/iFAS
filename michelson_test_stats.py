import numpy as np
import cython_functions
from scipy import stats
import my_utilities as MU
import matplotlib.pyplot as plt
import time


def harmonicp75w13o01_michelsonlocal_contrast_bcontent_measure_difference(Ref_image, Pro_image, sizeblk = 13, min_fb = 3, overlap = 1):
    Yref = MU.checkifRGB(Ref_image)
    # Yref = 0.299 * Ref_image[:, :, 0] + 0.587 * Ref_image[:, :, 1] + 0.114 * Ref_image[:, :, 2]
    start_time = time.clock()
    C_ref = cython_functions.local_contrast_bcontent_measure_michelson(Yref,np.array([sizeblk,sizeblk]),min_fb,overlap)
    print time.clock() - start_time, overlap
    C_ref_temp = C_ref[sizeblk//2:C_ref.shape[0]-sizeblk//2:overlap,sizeblk//2:C_ref.shape[1]-sizeblk//2:overlap]
    maxCref = np.max(C_ref_temp)
    e_ref = 1./np.mean(1./C_ref_temp[C_ref_temp>0.75*maxCref])
    if np.isnan(e_ref):
        e_ref = np.mean(C_ref_temp[C_ref_temp>-1])
    Ypro = MU.checkifRGB(Pro_image)
    # Ypro = 0.299 * Pro_image[:, :, 0] + 0.587 * Pro_image[:, :, 1] + 0.114 * Pro_image[:, :, 2]
    C_pro = cython_functions.local_contrast_bcontent_measure_michelson(Ypro,np.array([sizeblk,sizeblk]),min_fb,overlap)
    C_pro_temp = C_pro[sizeblk//2:C_pro.shape[0]-sizeblk//2:overlap,sizeblk//2:C_pro.shape[1]-sizeblk//2:overlap]
    maxCpro = np.max(C_pro_temp)
    e_pro = 1./np.mean(1./C_pro_temp[C_pro_temp>0.75*maxCpro])
    if np.isnan(e_pro):
        e_pro = np.mean(C_pro_temp[C_pro_temp>-1])
    C_diff = C_ref - C_pro
    return C_diff, e_ref-e_pro


def harmonicp75w13o02_michelsonlocal_contrast_bcontent_measure_difference(Ref_image, Pro_image, sizeblk = 13, min_fb = 3, overlap = 2):
    Yref = MU.checkifRGB(Ref_image)
    # Yref = 0.299 * Ref_image[:, :, 0] + 0.587 * Ref_image[:, :, 1] + 0.114 * Ref_image[:, :, 2]
    start_time = time.clock()
    C_ref = cython_functions.local_contrast_bcontent_measure_michelson(Yref,np.array([sizeblk,sizeblk]),min_fb,overlap)
    print time.clock() - start_time, overlap
    C_ref_temp = C_ref[sizeblk//2:C_ref.shape[0]-sizeblk//2:overlap,sizeblk//2:C_ref.shape[1]-sizeblk//2:overlap]
    maxCref = np.max(C_ref_temp)
    e_ref = 1./np.mean(1./C_ref_temp[C_ref_temp>0.75*maxCref])
    if np.isnan(e_ref):
        e_ref = np.mean(C_ref_temp[C_ref_temp>-1])
    Ypro = MU.checkifRGB(Pro_image)
    # Ypro = 0.299 * Pro_image[:, :, 0] + 0.587 * Pro_image[:, :, 1] + 0.114 * Pro_image[:, :, 2]
    C_pro = cython_functions.local_contrast_bcontent_measure_michelson(Ypro,np.array([sizeblk,sizeblk]),min_fb,overlap)
    C_pro_temp = C_pro[sizeblk//2:C_pro.shape[0]-sizeblk//2:overlap,sizeblk//2:C_pro.shape[1]-sizeblk//2:overlap]
    maxCpro = np.max(C_pro_temp)
    e_pro = 1./np.mean(1./C_pro_temp[C_pro_temp>0.75*maxCpro])
    if np.isnan(e_pro):
        e_pro = np.mean(C_pro_temp[C_pro_temp>-1])
    C_diff = C_ref - C_pro
    return C_diff, e_ref-e_pro


def harmonicp75w13o03_michelsonlocal_contrast_bcontent_measure_difference(Ref_image, Pro_image, sizeblk = 13, min_fb = 3, overlap = 3):
    Yref = MU.checkifRGB(Ref_image)
    # Yref = 0.299 * Ref_image[:, :, 0] + 0.587 * Ref_image[:, :, 1] + 0.114 * Ref_image[:, :, 2]
    start_time = time.clock()
    C_ref = cython_functions.local_contrast_bcontent_measure_michelson(Yref,np.array([sizeblk,sizeblk]),min_fb,overlap)
    print time.clock() - start_time, overlap
    C_ref_temp = C_ref[sizeblk//2:C_ref.shape[0]-sizeblk//2:overlap,sizeblk//2:C_ref.shape[1]-sizeblk//2:overlap]
    maxCref = np.max(C_ref_temp)
    e_ref = 1./np.mean(1./C_ref_temp[C_ref_temp>0.75*maxCref])
    if np.isnan(e_ref):
        e_ref = np.mean(C_ref_temp[C_ref_temp>-1])
    Ypro = MU.checkifRGB(Pro_image)
    # Ypro = 0.299 * Pro_image[:, :, 0] + 0.587 * Pro_image[:, :, 1] + 0.114 * Pro_image[:, :, 2]
    C_pro = cython_functions.local_contrast_bcontent_measure_michelson(Ypro,np.array([sizeblk,sizeblk]),min_fb,overlap)
    C_pro_temp = C_pro[sizeblk//2:C_pro.shape[0]-sizeblk//2:overlap,sizeblk//2:C_pro.shape[1]-sizeblk//2:overlap]
    maxCpro = np.max(C_pro_temp)
    e_pro = 1./np.mean(1./C_pro_temp[C_pro_temp>0.75*maxCpro])
    if np.isnan(e_pro):
        e_pro = np.mean(C_pro_temp[C_pro_temp>-1])
    C_diff = C_ref - C_pro
    return C_diff, e_ref-e_pro



def harmonicp75w13o04_michelsonlocal_contrast_bcontent_measure_difference(Ref_image, Pro_image, sizeblk = 13, min_fb = 3, overlap = 4):
    Yref = MU.checkifRGB(Ref_image)
    # Yref = 0.299 * Ref_image[:, :, 0] + 0.587 * Ref_image[:, :, 1] + 0.114 * Ref_image[:, :, 2]
    start_time = time.clock()
    C_ref = cython_functions.local_contrast_bcontent_measure_michelson(Yref,np.array([sizeblk,sizeblk]),min_fb,overlap)
    print time.clock() - start_time, overlap
    C_ref_temp = C_ref[sizeblk//2:C_ref.shape[0]-sizeblk//2:overlap,sizeblk//2:C_ref.shape[1]-sizeblk//2:overlap]
    maxCref = np.max(C_ref_temp)
    e_ref = 1./np.mean(1./C_ref_temp[C_ref_temp>0.75*maxCref])
    if np.isnan(e_ref):
        e_ref = np.mean(C_ref_temp[C_ref_temp>-1])
    Ypro = MU.checkifRGB(Pro_image)
    # Ypro = 0.299 * Pro_image[:, :, 0] + 0.587 * Pro_image[:, :, 1] + 0.114 * Pro_image[:, :, 2]
    C_pro = cython_functions.local_contrast_bcontent_measure_michelson(Ypro,np.array([sizeblk,sizeblk]),min_fb,overlap)
    C_pro_temp = C_pro[sizeblk//2:C_pro.shape[0]-sizeblk//2:overlap,sizeblk//2:C_pro.shape[1]-sizeblk//2:overlap]
    maxCpro = np.max(C_pro_temp)
    e_pro = 1./np.mean(1./C_pro_temp[C_pro_temp>0.75*maxCpro])
    if np.isnan(e_pro):
        e_pro = np.mean(C_pro_temp[C_pro_temp>-1])
    C_diff = C_ref - C_pro
    return C_diff, e_ref-e_pro


def harmonicp75w13o05_michelsonlocal_contrast_bcontent_measure_difference(Ref_image, Pro_image, sizeblk = 13, min_fb = 3, overlap = 5):
    Yref = MU.checkifRGB(Ref_image)
    # Yref = 0.299 * Ref_image[:, :, 0] + 0.587 * Ref_image[:, :, 1] + 0.114 * Ref_image[:, :, 2]
    start_time = time.clock()
    C_ref = cython_functions.local_contrast_bcontent_measure_michelson(Yref,np.array([sizeblk,sizeblk]),min_fb,overlap)
    print time.clock() - start_time, overlap
    C_ref_temp = C_ref[sizeblk//2:C_ref.shape[0]-sizeblk//2:overlap,sizeblk//2:C_ref.shape[1]-sizeblk//2:overlap]
    maxCref = np.max(C_ref_temp)
    e_ref = 1./np.mean(1./C_ref_temp[C_ref_temp>0.75*maxCref])
    if np.isnan(e_ref):
        e_ref = np.mean(C_ref_temp[C_ref_temp>-1])
    Ypro = MU.checkifRGB(Pro_image)
    # Ypro = 0.299 * Pro_image[:, :, 0] + 0.587 * Pro_image[:, :, 1] + 0.114 * Pro_image[:, :, 2]
    C_pro = cython_functions.local_contrast_bcontent_measure_michelson(Ypro,np.array([sizeblk,sizeblk]),min_fb,overlap)
    C_pro_temp = C_pro[sizeblk//2:C_pro.shape[0]-sizeblk//2:overlap,sizeblk//2:C_pro.shape[1]-sizeblk//2:overlap]
    maxCpro = np.max(C_pro_temp)
    e_pro = 1./np.mean(1./C_pro_temp[C_pro_temp>0.75*maxCpro])
    if np.isnan(e_pro):
        e_pro = np.mean(C_pro_temp[C_pro_temp>-1])
    C_diff = C_ref - C_pro
    return C_diff, e_ref-e_pro



def harmonicp75w13o06_michelsonlocal_contrast_bcontent_measure_difference(Ref_image, Pro_image, sizeblk = 13, min_fb = 3, overlap = 6):
    Yref = MU.checkifRGB(Ref_image)
    # Yref = 0.299 * Ref_image[:, :, 0] + 0.587 * Ref_image[:, :, 1] + 0.114 * Ref_image[:, :, 2]
    start_time = time.clock()
    C_ref = cython_functions.local_contrast_bcontent_measure_michelson(Yref,np.array([sizeblk,sizeblk]),min_fb,overlap)
    print time.clock() - start_time, overlap
    C_ref_temp = C_ref[sizeblk//2:C_ref.shape[0]-sizeblk//2:overlap,sizeblk//2:C_ref.shape[1]-sizeblk//2:overlap]
    maxCref = np.max(C_ref_temp)
    e_ref = 1./np.mean(1./C_ref_temp[C_ref_temp>0.75*maxCref])
    if np.isnan(e_ref):
        e_ref = np.mean(C_ref_temp[C_ref_temp>-1])
    Ypro = MU.checkifRGB(Pro_image)
    # Ypro = 0.299 * Pro_image[:, :, 0] + 0.587 * Pro_image[:, :, 1] + 0.114 * Pro_image[:, :, 2]
    C_pro = cython_functions.local_contrast_bcontent_measure_michelson(Ypro,np.array([sizeblk,sizeblk]),min_fb,overlap)
    C_pro_temp = C_pro[sizeblk//2:C_pro.shape[0]-sizeblk//2:overlap,sizeblk//2:C_pro.shape[1]-sizeblk//2:overlap]
    maxCpro = np.max(C_pro_temp)
    e_pro = 1./np.mean(1./C_pro_temp[C_pro_temp>0.75*maxCpro])
    if np.isnan(e_pro):
        e_pro = np.mean(C_pro_temp[C_pro_temp>-1])
    C_diff = C_ref - C_pro
    return C_diff, e_ref-e_pro



def harmonicp75w13o07_michelsonlocal_contrast_bcontent_measure_difference(Ref_image, Pro_image, sizeblk = 13, min_fb = 3, overlap = 7):
    Yref = MU.checkifRGB(Ref_image)
    # Yref = 0.299 * Ref_image[:, :, 0] + 0.587 * Ref_image[:, :, 1] + 0.114 * Ref_image[:, :, 2]
    start_time = time.clock()
    C_ref = cython_functions.local_contrast_bcontent_measure_michelson(Yref,np.array([sizeblk,sizeblk]),min_fb,overlap)
    print time.clock() - start_time, overlap
    C_ref_temp = C_ref[sizeblk//2:C_ref.shape[0]-sizeblk//2:overlap,sizeblk//2:C_ref.shape[1]-sizeblk//2:overlap]
    maxCref = np.max(C_ref_temp)
    e_ref = 1./np.mean(1./C_ref_temp[C_ref_temp>0.75*maxCref])
    if np.isnan(e_ref):
        e_ref = np.mean(C_ref_temp[C_ref_temp>-1])
    Ypro = MU.checkifRGB(Pro_image)
    # Ypro = 0.299 * Pro_image[:, :, 0] + 0.587 * Pro_image[:, :, 1] + 0.114 * Pro_image[:, :, 2]
    C_pro = cython_functions.local_contrast_bcontent_measure_michelson(Ypro,np.array([sizeblk,sizeblk]),min_fb,overlap)
    C_pro_temp = C_pro[sizeblk//2:C_pro.shape[0]-sizeblk//2:overlap,sizeblk//2:C_pro.shape[1]-sizeblk//2:overlap]
    maxCpro = np.max(C_pro_temp)
    e_pro = 1./np.mean(1./C_pro_temp[C_pro_temp>0.75*maxCpro])
    if np.isnan(e_pro):
        e_pro = np.mean(C_pro_temp[C_pro_temp>-1])
    C_diff = C_ref - C_pro
    return C_diff, e_ref-e_pro


def harmonicp75w13o08_michelsonlocal_contrast_bcontent_measure_difference(Ref_image, Pro_image, sizeblk = 13, min_fb = 3, overlap = 8):
    Yref = MU.checkifRGB(Ref_image)
    # Yref = 0.299 * Ref_image[:, :, 0] + 0.587 * Ref_image[:, :, 1] + 0.114 * Ref_image[:, :, 2]
    start_time = time.clock()
    C_ref = cython_functions.local_contrast_bcontent_measure_michelson(Yref,np.array([sizeblk,sizeblk]),min_fb,overlap)
    print time.clock() - start_time, overlap
    C_ref_temp = C_ref[sizeblk//2:C_ref.shape[0]-sizeblk//2:overlap,sizeblk//2:C_ref.shape[1]-sizeblk//2:overlap]
    maxCref = np.max(C_ref_temp)
    e_ref = 1./np.mean(1./C_ref_temp[C_ref_temp>0.75*maxCref])
    if np.isnan(e_ref):
        e_ref = np.mean(C_ref_temp[C_ref_temp>-1])
    Ypro = MU.checkifRGB(Pro_image)
    # Ypro = 0.299 * Pro_image[:, :, 0] + 0.587 * Pro_image[:, :, 1] + 0.114 * Pro_image[:, :, 2]
    C_pro = cython_functions.local_contrast_bcontent_measure_michelson(Ypro,np.array([sizeblk,sizeblk]),min_fb,overlap)
    C_pro_temp = C_pro[sizeblk//2:C_pro.shape[0]-sizeblk//2:overlap,sizeblk//2:C_pro.shape[1]-sizeblk//2:overlap]
    maxCpro = np.max(C_pro_temp)
    e_pro = 1./np.mean(1./C_pro_temp[C_pro_temp>0.75*maxCpro])
    if np.isnan(e_pro):
        e_pro = np.mean(C_pro_temp[C_pro_temp>-1])
    C_diff = C_ref - C_pro
    return C_diff, e_ref-e_pro


def harmonicp75w13o09_michelsonlocal_contrast_bcontent_measure_difference(Ref_image, Pro_image, sizeblk = 13, min_fb = 3, overlap = 9):
    Yref = MU.checkifRGB(Ref_image)
    # Yref = 0.299 * Ref_image[:, :, 0] + 0.587 * Ref_image[:, :, 1] + 0.114 * Ref_image[:, :, 2]
    start_time = time.clock()
    C_ref = cython_functions.local_contrast_bcontent_measure_michelson(Yref,np.array([sizeblk,sizeblk]),min_fb,overlap)
    print time.clock() - start_time, overlap
    C_ref_temp = C_ref[sizeblk//2:C_ref.shape[0]-sizeblk//2:overlap,sizeblk//2:C_ref.shape[1]-sizeblk//2:overlap]
    maxCref = np.max(C_ref_temp)
    e_ref = 1./np.mean(1./C_ref_temp[C_ref_temp>0.75*maxCref])
    if np.isnan(e_ref):
        e_ref = np.mean(C_ref_temp[C_ref_temp>-1])
    Ypro = MU.checkifRGB(Pro_image)
    # Ypro = 0.299 * Pro_image[:, :, 0] + 0.587 * Pro_image[:, :, 1] + 0.114 * Pro_image[:, :, 2]
    C_pro = cython_functions.local_contrast_bcontent_measure_michelson(Ypro,np.array([sizeblk,sizeblk]),min_fb,overlap)
    C_pro_temp = C_pro[sizeblk//2:C_pro.shape[0]-sizeblk//2:overlap,sizeblk//2:C_pro.shape[1]-sizeblk//2:overlap]
    maxCpro = np.max(C_pro_temp)
    e_pro = 1./np.mean(1./C_pro_temp[C_pro_temp>0.75*maxCpro])
    if np.isnan(e_pro):
        e_pro = np.mean(C_pro_temp[C_pro_temp>-1])
    C_diff = C_ref - C_pro
    return C_diff, e_ref-e_pro


def harmonicp75w13o10_michelsonlocal_contrast_bcontent_measure_difference(Ref_image, Pro_image, sizeblk = 13, min_fb = 3, overlap = 10):
    Yref = MU.checkifRGB(Ref_image)
    # Yref = 0.299 * Ref_image[:, :, 0] + 0.587 * Ref_image[:, :, 1] + 0.114 * Ref_image[:, :, 2]
    start_time = time.clock()
    C_ref = cython_functions.local_contrast_bcontent_measure_michelson(Yref,np.array([sizeblk,sizeblk]),min_fb,overlap)
    print time.clock() - start_time, overlap
    C_ref_temp = C_ref[sizeblk//2:C_ref.shape[0]-sizeblk//2:overlap,sizeblk//2:C_ref.shape[1]-sizeblk//2:overlap]
    maxCref = np.max(C_ref_temp)
    e_ref = 1./np.mean(1./C_ref_temp[C_ref_temp>0.75*maxCref])
    if np.isnan(e_ref):
        e_ref = np.mean(C_ref_temp[C_ref_temp>-1])
    Ypro = MU.checkifRGB(Pro_image)
    # Ypro = 0.299 * Pro_image[:, :, 0] + 0.587 * Pro_image[:, :, 1] + 0.114 * Pro_image[:, :, 2]
    C_pro = cython_functions.local_contrast_bcontent_measure_michelson(Ypro,np.array([sizeblk,sizeblk]),min_fb,overlap)
    C_pro_temp = C_pro[sizeblk//2:C_pro.shape[0]-sizeblk//2:overlap,sizeblk//2:C_pro.shape[1]-sizeblk//2:overlap]
    maxCpro = np.max(C_pro_temp)
    e_pro = 1./np.mean(1./C_pro_temp[C_pro_temp>0.75*maxCpro])
    if np.isnan(e_pro):
        e_pro = np.mean(C_pro_temp[C_pro_temp>-1])
    C_diff = C_ref - C_pro
    return C_diff, e_ref-e_pro


def harmonicp75w13o11_michelsonlocal_contrast_bcontent_measure_difference(Ref_image, Pro_image, sizeblk = 13, min_fb = 3, overlap = 11):
    Yref = MU.checkifRGB(Ref_image)
    # Yref = 0.299 * Ref_image[:, :, 0] + 0.587 * Ref_image[:, :, 1] + 0.114 * Ref_image[:, :, 2]
    start_time = time.clock()
    C_ref = cython_functions.local_contrast_bcontent_measure_michelson(Yref,np.array([sizeblk,sizeblk]),min_fb,overlap)
    print time.clock() - start_time, overlap
    C_ref_temp = C_ref[sizeblk//2:C_ref.shape[0]-sizeblk//2:overlap,sizeblk//2:C_ref.shape[1]-sizeblk//2:overlap]
    maxCref = np.max(C_ref_temp)
    e_ref = 1./np.mean(1./C_ref_temp[C_ref_temp>0.75*maxCref])
    if np.isnan(e_ref):
        e_ref = np.mean(C_ref_temp[C_ref_temp>-1])
    Ypro = MU.checkifRGB(Pro_image)
    # Ypro = 0.299 * Pro_image[:, :, 0] + 0.587 * Pro_image[:, :, 1] + 0.114 * Pro_image[:, :, 2]
    C_pro = cython_functions.local_contrast_bcontent_measure_michelson(Ypro,np.array([sizeblk,sizeblk]),min_fb,overlap)
    C_pro_temp = C_pro[sizeblk//2:C_pro.shape[0]-sizeblk//2:overlap,sizeblk//2:C_pro.shape[1]-sizeblk//2:overlap]
    maxCpro = np.max(C_pro_temp)
    e_pro = 1./np.mean(1./C_pro_temp[C_pro_temp>0.75*maxCpro])
    if np.isnan(e_pro):
        e_pro = np.mean(C_pro_temp[C_pro_temp>-1])
    C_diff = C_ref - C_pro
    return C_diff, e_ref-e_pro


def harmonicp75w13o12_michelsonlocal_contrast_bcontent_measure_difference(Ref_image, Pro_image, sizeblk = 13, min_fb = 3, overlap = 12):
    Yref = MU.checkifRGB(Ref_image)
    # Yref = 0.299 * Ref_image[:, :, 0] + 0.587 * Ref_image[:, :, 1] + 0.114 * Ref_image[:, :, 2]
    start_time = time.clock()
    C_ref = cython_functions.local_contrast_bcontent_measure_michelson(Yref,np.array([sizeblk,sizeblk]),min_fb,overlap)
    print time.clock() - start_time, overlap
    C_ref_temp = C_ref[sizeblk//2:C_ref.shape[0]-sizeblk//2:overlap,sizeblk//2:C_ref.shape[1]-sizeblk//2:overlap]
    maxCref = np.max(C_ref_temp)
    e_ref = 1./np.mean(1./C_ref_temp[C_ref_temp>0.75*maxCref])
    if np.isnan(e_ref):
        e_ref = np.mean(C_ref_temp[C_ref_temp>-1])
    Ypro = MU.checkifRGB(Pro_image)
    # Ypro = 0.299 * Pro_image[:, :, 0] + 0.587 * Pro_image[:, :, 1] + 0.114 * Pro_image[:, :, 2]
    C_pro = cython_functions.local_contrast_bcontent_measure_michelson(Ypro,np.array([sizeblk,sizeblk]),min_fb,overlap)
    C_pro_temp = C_pro[sizeblk//2:C_pro.shape[0]-sizeblk//2:overlap,sizeblk//2:C_pro.shape[1]-sizeblk//2:overlap]
    maxCpro = np.max(C_pro_temp)
    e_pro = 1./np.mean(1./C_pro_temp[C_pro_temp>0.75*maxCpro])
    if np.isnan(e_pro):
        e_pro = np.mean(C_pro_temp[C_pro_temp>-1])
    C_diff = C_ref - C_pro
    return C_diff, e_ref-e_pro


def harmonicp75w13o13_michelsonlocal_contrast_bcontent_measure_difference(Ref_image, Pro_image, sizeblk = 13, min_fb = 3, overlap = 13):
    Yref = MU.checkifRGB(Ref_image)
    # Yref = 0.299 * Ref_image[:, :, 0] + 0.587 * Ref_image[:, :, 1] + 0.114 * Ref_image[:, :, 2]
    start_time = time.clock()
    C_ref = cython_functions.local_contrast_bcontent_measure_michelson(Yref,np.array([sizeblk,sizeblk]),min_fb,overlap)
    print time.clock() - start_time, overlap
    C_ref_temp = C_ref[sizeblk//2:C_ref.shape[0]-sizeblk//2:overlap,sizeblk//2:C_ref.shape[1]-sizeblk//2:overlap]
    maxCref = np.max(C_ref_temp)
    e_ref = 1./np.mean(1./C_ref_temp[C_ref_temp>0.75*maxCref])
    if np.isnan(e_ref):
        e_ref = np.mean(C_ref_temp[C_ref_temp>-1])
    Ypro = MU.checkifRGB(Pro_image)
    # Ypro = 0.299 * Pro_image[:, :, 0] + 0.587 * Pro_image[:, :, 1] + 0.114 * Pro_image[:, :, 2]
    C_pro = cython_functions.local_contrast_bcontent_measure_michelson(Ypro,np.array([sizeblk,sizeblk]),min_fb,overlap)
    C_pro_temp = C_pro[sizeblk//2:C_pro.shape[0]-sizeblk//2:overlap,sizeblk//2:C_pro.shape[1]-sizeblk//2:overlap]
    maxCpro = np.max(C_pro_temp)
    e_pro = 1./np.mean(1./C_pro_temp[C_pro_temp>0.75*maxCpro])
    if np.isnan(e_pro):
        e_pro = np.mean(C_pro_temp[C_pro_temp>-1])
    C_diff = C_ref - C_pro
    return C_diff, e_ref-e_pro