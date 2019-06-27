import numpy as np
from scipy import misc
import pywt
import cythonFunctions


def SME(img, blk_size=np.array([3, 3])):
    if len(img.shape) == 3:
        Yref = 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]
    else:
        Yref = 1. * img
    max_y = np.max(Yref)
    if max_y <= 1.:
        Yref = 255. * Yref
    n = 0
    sme = 0.
    M, N = Yref.shape
    C = np.zeros_like(Yref)
    for ii in xrange(0, M - blk_size[0], blk_size[0]):
        for jj in xrange(0, N - blk_size[1], blk_size[1]):
            values = Yref[ii:ii + blk_size[0], jj:jj + blk_size[1]]
            Imax = np.maximum(1., np.max(values))
            Imin = np.maximum(1., np.min(values))
            c = np.log(Imax / Imin)
            C[ii:ii + blk_size[0], jj:jj + blk_size[1]] = c
            sme += c
            n += 1
    sme = 20. * sme / (blk_size[0] * blk_size[1])
    return C, sme / n


def WME(img, blk_size=np.array([3, 3])):
    if len(img.shape) == 3:
        Yref = 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]
    else:
        Yref = 1. * img
    max_y = np.max(Yref)
    if max_y <= 1:
        Yref = 255. * Yref
    n = 0
    wme = 0.
    M, N = Yref.shape
    C = np.zeros_like(Yref)
    for ii in xrange(0, M - blk_size[0], blk_size[0]):
        for jj in xrange(0, N - blk_size[1], blk_size[1]):
            values = Yref[ii:ii + blk_size[0], jj:jj + blk_size[1]]
            Imax = np.maximum(0., np.max(values))
            Imin = np.maximum(1., np.min(values))
            c = np.log((np.abs(Imax - Imin) / (Imin)) + 1)
            C[ii:ii + blk_size[0], jj:jj + blk_size[1]] = c
            wme += c
            n += 1
    wme = 20. * wme / (blk_size[0] * blk_size[1])
    return C, wme / n


def MME(img, blk_size=np.array([3, 3])):
    if len(img.shape) == 3:
        Yref = 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]
    else:
        Yref = 1. * img
    max_y = np.max(Yref)
    if max_y <= 1:
        Yref = 255. * Yref
    n = 0
    mme = 0.
    M, N = Yref.shape
    C = np.zeros_like(Yref)
    for ii in xrange(0, M - blk_size[0], blk_size[0]):
        for jj in xrange(0, N - blk_size[1], blk_size[1]):
            values = Yref[ii:ii + blk_size[0], jj:jj + blk_size[1]]
            Imax = np.maximum(1., np.max(values))
            Imin = np.min(values)
            c = np.log(((Imax - Imin) / (Imax + Imin)) + 1)
            mme += c
            n += 1
            C[ii:ii + blk_size[0], jj:jj + blk_size[1]] = c
    mme = 20. * mme / (blk_size[0] * blk_size[1])
    return C, mme / n


def RME(img, blk_size=np.array([3, 3])):
    if len(img.shape) == 3:
        Yref = 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]
    else:
        Yref = 1. * img
    max_y = np.max(Yref)
    if max_y <= 1:
        Yref = 255. * Yref
    n = 0
    rme = 0.
    M, N = Yref.shape
    C = np.zeros_like(Yref)
    for ii in xrange(0, M - blk_size[0], blk_size[0]):
        for jj in xrange(0, N - blk_size[1], blk_size[1]):
            values = Yref[ii:ii + blk_size[0], jj:jj + blk_size[1]]
            values_center = values[np.int(np.floor(blk_size[0] / 2.)), np.int(np.floor(blk_size[0] / 2.))]
            mean_values = np.mean(values)
            Imin = np.maximum(1., np.abs(values_center - mean_values))
            Imax = np.maximum(1., np.abs(values_center + mean_values))
            if Imax == 1:
                c = 0
            else:
                c = np.abs(np.log(Imin) / np.log(Imax))
                rme += np.sqrt(c)
                n += 1
            C[ii:ii + blk_size[0], jj:jj + blk_size[1]] = np.sqrt(c)
    rme = rme / (blk_size[0] * blk_size[1])
    return C, rme / n


def contrast_peli(img, wname='db1'):
    alpha = 0.1
    if len(img.shape) == 3:
        Yref = 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]
    else:
        Yref = 1. * img
    cA = Yref
    CA = Yref
    C = np.zeros(Yref.shape)
    levels = pywt.dwt_max_level(np.int_(np.min(Yref.shape)), pywt.Wavelet(wname).dec_len)
    c = 0.
    for kk in range(levels-1):
        (cA, (cH, cV, cD)) = pywt.dwt2(cA, wname, 'sym')
        (CA, (CH, CV, CD)) = pywt.swt2(CA, wname, level=1)[0]
        app = alpha * np.mean(cA) + (1 - alpha) * cA
        App = alpha * np.mean(CA) + (1 - alpha) * CA
        th = np.max(cH) / 10.
        cH[np.where(cH<th)] = 0.
        tv = np.max(cV) / 10.
        cV[np.where(cV < tv)] = 0.
        td = np.max(cD) / 10.
        cD[np.where(cD < td)] = 0.
        # Contrast value with decimated wavelet
        contrastH = np.abs(cH) / np.abs(app)
        contrastV = np.abs(cV) / np.abs(app)
        contrastD = np.abs(cD) / np.abs(app)
        m0 = np.mean(contrastH[np.where(contrastH > 0.)])
        m1 = np.mean(contrastV[np.where(contrastV > 0.)])
        m2 = np.mean(contrastD[np.where(contrastD > 0.)])
        if np.isnan(m0):
            m0 = 0.
        if np.isnan(m1):
            m1 = 0.
        if np.isnan(m2):
            m2 = 0.
        c += (m0 + m1 + m2) / 3.
        # Contrast map computed using non decimated wavelet
        Th = np.max(CH) / 10.
        CH[np.where(CH < Th)] = 0.
        Tv = np.max(CV) / 10.
        CV[np.where(CV < Tv)] = 0.
        Td = np.max(CD) / 10.
        CD[np.where(CD < Td)] = 0.
        ContrastH = np.abs(CH) / np.abs(App)
        ContrastV = np.abs(CV) / np.abs(App)
        ContrastD = np.abs(CD) / np.abs(App)
        C += (ContrastH + ContrastV + ContrastD) / 3.
    c = c / levels
    C = C / levels
    return C, c


def contrast_rizzi(img):
    Numfilters = 3
    nstds = 3
    if len(img.shape) == 3:
        Yref = 0.299 * img[:, :, 0] + 0.587 * img[:, :, 1] + 0.114 * img[:, :, 2]
    else:
        Yref = 1. * img
    C = {}
    C[0] = cythonFunctions.crm_convolution(Yref, 3)
    cglobal = np.mean(C[0])
    A = Yref
    C_total = C[0]
    for ii in range(1, Numfilters + 1):
        A = A[0::2, 0::2]
        C[ii] = cythonFunctions.crm_convolution(A, 3)
        cglobal += np.mean(C[ii])
        C_total += misc.imresize(C[ii], Yref.shape)
    return C_total / (Numfilters + 1.), cglobal / (Numfilters + 1.)
