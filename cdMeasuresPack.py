#!/usr/bin/env python2.7
# Importing necessary packages
import numpy as np
from scipy import ndimage
from scipy import signal
from skimage import color
import contentFeatures
import miselaneusPack
import computeTexture
import myUtilities
import colorSpaces


def cd00_deltaE2000(Ref_image, Pro_image, KLCH=np.array([1, 1, 1]), rcompo=False):
    kl = KLCH[0]
    kc = KLCH[1]
    kh = KLCH[2]
    Lab_ref = color.rgb2lab(Ref_image)
    Lab_pro = color.rgb2lab(Pro_image)
    Lstd = Lab_ref[:, :, 0]
    astd = Lab_ref[:, :, 1]
    bstd = Lab_ref[:, :, 2]
    Cabstd = np.sqrt(np.power(astd, 2) + np.power(bstd, 2))
    Lsample = Lab_pro[:, :, 0]
    asample = Lab_pro[:, :, 1]
    bsample = Lab_pro[:, :, 2]
    Cabsample = np.sqrt(np.power(asample, 2) + np.power(bsample, 2))
    Cabarithmean = (Cabstd + Cabsample) / 2.
    G = 0.5 * (1 - np.sqrt(np.power(Cabarithmean, 7) / (np.power(Cabarithmean, 7) + 25. ** 7)))
    apstd = (1 + G) * astd
    apsample = (1 + G) * asample
    Cpsample = np.sqrt(np.power(apsample, 2) + np.power(bsample, 2))
    Cpstd = np.sqrt(np.power(apstd, 2) + np.power(bstd, 2))
    Cpprod = (Cpsample * Cpstd)
    zcidx = np.where(Cpprod == 0)
    hpstd = np.arctan2(bstd, apstd)
    hpstd = hpstd + 2 * np.pi * (hpstd < 0)
    hpstd[(np.abs(apstd) + np.abs(bstd)) == 0] = 0
    hpsample = np.arctan2(bsample, apsample)
    hpsample = hpsample + 2 * np.pi * (hpsample < 0)
    hpsample[(np.abs(apsample) + np.abs(bsample)) == 0] = 0
    dL = (Lsample - Lstd)
    dC = (Cpsample - Cpstd)
    dhp = (hpsample - hpstd)
    dhp = dhp - 2 * np.pi * (dhp > np.pi)
    dhp = dhp + 2 * np.pi * (dhp < (-np.pi))
    dhp[zcidx] = 0
    dH = 2 * np.sqrt(Cpprod) * np.sin(dhp / 2)
    Lp = (Lsample + Lstd) / 2
    Cp = (Cpstd + Cpsample) / 2
    hp = (hpstd + hpsample) / 2
    hp = hp - (np.abs(hpstd - hpsample) > np.pi) * np.pi
    hp = hp + (hp < 0) * 2 * np.pi
    hp[zcidx] = hpsample[zcidx] + hpstd[zcidx]
    Lpm502 = np.power((Lp - 50), 2)
    Sl = 1 + 0.015 * Lpm502 / np.sqrt(20 + Lpm502)
    Sc = 1 + 0.045 * Cp
    T = 1 - 0.17 * np.cos(hp - np.pi / 6) + 0.24 * np.cos(2 * hp) + 0.32 * np.cos(3 * hp + np.pi / 30) - 0.20 * np.cos(
        4 * hp - 63 * np.pi / 180)
    Sh = 1 + 0.015 * Cp * T
    delthetarad = (30 * np.pi / 180) * np.exp(- np.power((180 / np.pi * hp - 275) / 25, 2))
    Rc = 2 * np.sqrt(np.power(Cp, 7) / (np.power(Cp, 7) + 25 ** 7))
    RT = - np.sin(2 * delthetarad) * Rc
    klSl = kl * Sl
    kcSc = kc * Sc
    khSh = kh * Sh
    DE00 = np.sqrt(
        np.power((dL / klSl), 2) + np.power((dC / kcSc), 2) + np.power((dH / khSh), 2) + RT * (dC / kcSc) * (dH / khSh))
    de00 = np.mean(DE00[:])
    if rcompo is False:
        return DE00, de00
    else:
        return DE00, de00, hpsample


def cd01_SdeltaE2000(Ref_image, Pro_image):
    wi = np.array([[0.921, 0.105, -0.108], [0.531, 0.330, 0], [0.488, 0.371, 0]])
    si = np.array([[0.0283, 0.133, 4.336], [0.0392, 0.494, 0], [0.0536, 0.386, 0]])
    xx, yy = np.meshgrid(np.arange(-11, 12), np.arange(-11, 12))
    h1 = wi[0, 0] * myUtilities.gaussian(xx, yy, si[0, 0]) + wi[0, 1] * myUtilities.gaussian(xx, yy, si[0, 1]) + wi[0, 2] * myUtilities.gaussian(
        xx, yy, si[0, 2])
    h1 = h1 / np.sum(h1[:])
    h2 = wi[1, 0] * myUtilities.gaussian(xx, yy, si[1, 0]) + wi[1, 1] * myUtilities.gaussian(xx, yy, si[1, 1])
    h2 = h2 / np.sum(h2[:])
    h3 = wi[2, 0] * myUtilities.gaussian(xx, yy, si[2, 0]) + wi[2, 1] * myUtilities.gaussian(xx, yy, si[2, 1])
    h3 = h3 / np.sum(h3[:])
    XYZ_ref = colorSpaces.RGB2XYZ(Ref_image / 255.)
    XYZ_pro = colorSpaces.RGB2XYZ(Pro_image / 255.)
    O123_ref = colorSpaces.XYZ2O1O2O3(XYZ_ref)
    O123_pro = colorSpaces.XYZ2O1O2O3(XYZ_pro)
    O1_ref = signal.convolve2d(O123_ref[:, :, 0], np.rot90(h1, 2), mode='valid')
    O2_ref = signal.convolve2d(O123_ref[:, :, 1], np.rot90(h2, 2), mode='valid')
    O3_ref = signal.convolve2d(O123_ref[:, :, 2], np.rot90(h3, 2), mode='valid')
    O1_pro = signal.convolve2d(O123_pro[:, :, 0], np.rot90(h1, 2), mode='valid')
    O2_pro = signal.convolve2d(O123_pro[:, :, 1], np.rot90(h2, 2), mode='valid')
    O3_pro = signal.convolve2d(O123_pro[:, :, 2], np.rot90(h3, 2), mode='valid')
    XYZ_ref = colorSpaces.O1O2O32XYZ(np.dstack((O1_ref, O2_ref, O3_ref)))
    XYZ_pro = colorSpaces.O1O2O32XYZ(np.dstack((O1_pro, O2_pro, O3_pro)))
    RGB_ref_back = 255 * colorSpaces.XYZ2RGB(XYZ_ref)
    RGB_ref_back[RGB_ref_back > 255] = 255
    RGB_ref_back[RGB_ref_back < 0] = 0
    RGB_ref_back = np.uint8(RGB_ref_back)
    RGB_pro_back = 255 * colorSpaces.XYZ2RGB(XYZ_pro)
    RGB_pro_back[RGB_pro_back > 255] = 255
    RGB_pro_back[RGB_pro_back < 0] = 0
    RGB_pro_back = np.uint8(RGB_pro_back)
    DE00, de00 = cd00_deltaE2000(RGB_ref_back, RGB_pro_back)
    return DE00, de00


def cd02_mahalanobis(Ref_image, Pro_image):
    Lab_ref = color.rgb2lab(Ref_image)
    Lab_pro = color.rgb2lab(Pro_image)
    Lstd = Lab_ref[:, :, 0]
    astd = Lab_ref[:, :, 1]
    bstd = Lab_ref[:, :, 2]
    Cabstd = np.sqrt(np.power(astd, 2) + np.power(bstd, 2))
    Lsample = Lab_pro[:, :, 0]
    asample = Lab_pro[:, :, 1]
    bsample = Lab_pro[:, :, 2]
    Cabsample = np.sqrt(np.power(asample, 2) + np.power(bsample, 2))
    Cabarithmean = (Cabstd + Cabsample) / 2.
    G = 0.5 * (1 - np.sqrt(np.power(Cabarithmean, 7) / (np.power(Cabarithmean, 7) + 25. ** 7)))
    apstd = (1 + G) * astd
    apsample = (1 + G) * asample
    hpstd = np.arctan2(bstd, apstd)
    hpstd = hpstd + 2 * np.pi * (hpstd < 0)
    hpstd[(np.abs(apstd) + np.abs(bstd)) == 0] = 0
    hpsample = np.arctan2(bsample, apsample)
    hpsample = hpsample + 2 * np.pi * (hpsample < 0)
    hpsample[(np.abs(apsample) + np.abs(bsample)) == 0] = 0
    num_1 = Lstd.size - 1
    muL = np.mean(Lstd[:])
    muC = np.mean(Cabstd[:])
    muh = np.mean(hpstd[:])
    sigmaLL = np.sum((Lstd[:] - muL) * (Lstd[:] - muL)) / num_1
    sigmaLC = np.sum((Lstd[:] - muL) * (Cabstd[:] - muC)) / num_1
    sigmaLh = np.sum((Lstd[:] - muL) * (hpstd[:] - muh)) / num_1
    sigmaCC = np.sum((Cabstd[:] - muC) * (Cabstd[:] - muC)) / num_1
    sigmaCh = np.sum((Cabstd[:] - muC) * (hpstd[:] - muh)) / num_1
    sigmahh = np.sum((hpstd[:] - muh) * (hpstd[:] - muh)) / num_1
    A = np.array([[sigmaLL, sigmaLC, sigmaLh], [sigmaLC, sigmaCC, sigmaCh], [sigmaLh, sigmaCh, sigmahh]])
    A = np.linalg.inv(A)
    deltaL = Lstd - Lsample
    deltaC = Cabstd - Cabsample
    deltaH = hpstd - hpsample
    CDM = A[0, 0] * np.power(deltaL, 2) + A[1, 1] * np.power(deltaC, 2) + A[2, 2] * np.power(deltaH, 2)
    CDM += 2 * A[0, 1] * (deltaL * deltaC) + 2 * A[0, 2] * (deltaL * deltaH) + 2 * A[1, 2] * (deltaC * deltaH)
    CDM = np.sqrt(CDM)
    cdm = np.mean(CDM[:])
    return CDM, cdm


def cd03_colorfulness_difference(Ref_image, Pro_image, sizewin=3, type='Gao'):
    C_ref, c_ref = contentFeatures.colorfulness(Ref_image, sizewin, type)
    C_pro, c_pro = contentFeatures.colorfulness(Pro_image, sizewin, type)
    return np.abs(C_ref - C_pro), np.abs(c_ref - c_pro)


def cd04_colorSSIM(Ref_image, Pro_image):
    wL = 3.05
    walpha = 1.1
    wbeta = 0.85
    L_alpha_beta_ref = colorSpaces.RGB2LAlphaBeta(Ref_image)
    L_alpha_beta_pro = colorSpaces.RGB2LAlphaBeta(Pro_image)
    ssim_map_L, mssim_L = miselaneusPack.ssim(L_alpha_beta_ref[:, :, 0], L_alpha_beta_pro[:, :, 0])
    ssim_map_alpha, mssim_alpha = miselaneusPack.ssim(L_alpha_beta_ref[:, :, 1], L_alpha_beta_pro[:, :, 1])
    ssim_map_beta, mssim_beta = miselaneusPack.ssim(L_alpha_beta_ref[:, :, 2], L_alpha_beta_pro[:, :, 2])
    cssim = np.sqrt(wL * mssim_L ** 2 + walpha * mssim_alpha ** 2 + wbeta * mssim_beta ** 2)
    CSSIM = np.sqrt(wL * np.power(ssim_map_L, 2) + walpha * np.power(ssim_map_alpha, 2)\
                    + wbeta * np.power(ssim_map_beta, 2))
    return CSSIM, cssim


def cd05_chroma_spread_extreme(Ref_image, Pro_image):
    YCbCr_ref = colorSpaces.RGB2YCbCr(Ref_image)
    YCbCr_pro = colorSpaces.RGB2YCbCr(Pro_image)
    mCb_ref, mCr_ref, MuCb_ref, MuCr_ref = myUtilities.meanblock(YCbCr_ref[:, :, 1], YCbCr_ref[:, :, 2])
    mCb_pro, mCr_pro, MuCb_pro, MuCr_pro = myUtilities.meanblock(YCbCr_pro[:, :, 1], YCbCr_pro[:, :, 2])
    eCbCr = np.sqrt(np.power(mCb_ref - mCb_pro, 2) + np.power(mCr_ref - mCr_pro, 2))
    ECbCr = np.sqrt(np.power(MuCb_ref - MuCb_pro, 2) + np.power(MuCr_ref - MuCr_pro, 2))
    chroma_spread = np.std(eCbCr)
    p = np.sort(np.array(eCbCr).ravel())[::-1]
    chroma_extreme = np.mean(p[0:np.int_(p.size * 0.01)]) - p[np.int_(p.size * 0.01) - 1]
    return ECbCr, 0.0192 * chroma_spread + 0.0076 * chroma_extreme


def cd06_colorhist_diff(Ref_image, Pro_image,blk_size=16):
    Lab_ref = color.rgb2lab(Ref_image / 255.)
    Lab_pro = color.rgb2lab(Pro_image / 255.)
    Diff = myUtilities.block_proc(Lab_ref, blk_size, C=Lab_pro, funname='IhistogramintersectionLab')
    H_ref = myUtilities.colorhistogram(Lab_ref, minc=np.array([0, -128, -128]), maxc=np.array([100, 127, 127]), nbins=8)
    H_pro = myUtilities.colorhistogram(Lab_pro, minc=np.array([0, -128, -128]), maxc=np.array([100, 127, 127]), nbins=8)
    if np.sum(H_ref[:]) != 0:
        H_ref = H_ref / np.sum(H_ref[:])
    if np.sum(H_pro[:]) != 0:
        H_pro = H_pro / np.sum(H_pro[:])
    di = np.sum(np.minimum(H_ref[:], H_pro[:]))
    return Diff, di


def cd07_weighted_deltaE(Ref_image, Pro_image):
    DE00, de00, hpsample = cd00_deltaE2000(Ref_image, Pro_image, rcompo=True)
    h_edges = np.linspace(0, 360, 181)
    hist_c, _ = np.histogram(180.*myUtilities.convert_vec(hpsample)/np.pi, h_edges)
    ind = np.digitize(180.*hpsample/np.pi, h_edges)
    hist_c = 1.*hist_c/np.sum(hist_c[:])
    index_sort = np.argsort(hist_c)
    hists_sort = hist_c[index_sort]
    hists_sort = np.cumsum(hists_sort)
    k_25 = np.where(hists_sort > 0.25)[0][0]
    k_50 = np.where(hists_sort > 0.50)[0][0]
    k_75 = np.where(hists_sort > 0.75)[0][0]
    hist_c[index_sort[0:k_25]] = hist_c[index_sort[0:k_25]]/4
    hist_c[index_sort[k_25:k_50]] = hist_c[index_sort[k_25:k_50]]/2
    hist_c[index_sort[k_50:k_75]] = hist_c[index_sort[k_50:k_75]]
    hist_c[index_sort[k_75:]] = 2.25*hist_c[index_sort[k_75:]]
    h_dis = np.zeros_like(hist_c)
    DE00w = np.zeros_like(DE00)
    for ii in range(0, h_dis.size):
        if np.sum(ind == ii) != 0:
            h_dis[ii] = np.mean(DE00[ind == ii])
            DE00w[ind == ii] = DE00[ind == ii]*hist_c[ii]
    wdE = np.nansum(np.power(h_dis,2)*hist_c)/4
    return DE00w, wdE


def cd08_cid_appearance(Ref_image, Pro_image):
    iCAM_ref = colorSpaces.RGB2iCAM(Ref_image)#ipt_image
    iCAM_pro = colorSpaces.RGB2iCAM(Pro_image)
    CID = np.nansum(np.power(iCAM_ref - iCAM_pro,2), 2)
    return CID, np.nanmean(CID)


def cd09_jncd_deltaE(Ref_image, Pro_image):
    w = 49
    JNCDLab = 2.3
    window = np.ones((w,w))/(w*w)
    Lab_ref = color.rgb2lab(Ref_image)
    temp = signal.convolve2d(Lab_ref[:,:,0], np.rot90(window,2), mode='same', boundary='symm')
    sigmaL_sq = signal.convolve2d(Lab_ref[:,:,0]*Lab_ref[:,:,0], np.rot90(window,2), mode='same', boundary='symm')\
                - temp*temp
    temp = signal.convolve2d(Lab_ref[:, :, 1], np.rot90(window, 2), mode='same', boundary='symm')
    sigmaa_sq = signal.convolve2d(Lab_ref[:,:,1]*Lab_ref[:,:,1], np.rot90(window, 2), mode='same', boundary='symm')\
                - temp*temp
    temp = signal.convolve2d(Lab_ref[:, :, 2], np.rot90(window, 2), mode='same', boundary='symm')
    sigmab_sq = signal.convolve2d(Lab_ref[:,:,2]*Lab_ref[:,:,2], np.rot90(window, 2), mode='same', boundary='symm')\
                - temp*temp
    vLab = (sigmaL_sq + sigmaa_sq + sigmab_sq)/3.
    alpha = (vLab/150)+1
    alpha[alpha>4.3] = 4.3
    Sc = 1+0.045*np.sqrt(Lab_ref[:,:,1]*Lab_ref[:,:,1]+Lab_ref[:,:,2]*Lab_ref[:,:,2])
    YCBCR = 0.299*Ref_image[:,:,0]+0.587*Ref_image[:,:,1]+0.114*Ref_image[:,:,2]
    gxY, gyY = np.gradient(YCBCR)
    GmY = np.sqrt(gxY*gxY+gyY*gyY)
    muY = signal.convolve2d(YCBCR, np.rot90(window, 2), mode='same', boundary='symm')
    rho_muY = myUtilities.rho_cd09(muY)
    beta = rho_muY*GmY+1
    VJNCD = JNCDLab*alpha*beta*Sc
    Lab_pro = color.rgb2lab(Pro_image)
    delta = np.abs(Lab_ref-Lab_pro)
    dPE = np.zeros_like(VJNCD)
    for ii in xrange(0, 3):
        temp = np.zeros_like(VJNCD)
        temp[delta[:,:,ii]>VJNCD] = 1
        dPE += temp*np.power(delta[:,:,ii]-VJNCD,2)
    dpE = np.sqrt(np.nanmean(dPE[:]/3.))
    return dPE/3., dpE


def cd10_deltaHS(Ref_image, Pro_image, sizewin=3):
    HSI_ref = color.rgb2hsv(Ref_image)
    HSI_pro = color.rgb2hsv(Pro_image)
    window = np.ones((sizewin,sizewin))
    w1 = 0.3
    w2 = 0.1
    deltaH = np.abs(np.mean(HSI_ref[:,:,0])-np.mean(HSI_pro[:,:,0]))
    deltaS = np.abs(np.mean(HSI_ref[:,:,1])-np.mean(HSI_pro[:,:,1]))
    deltaHS = w1*deltaH+w2*deltaS
    mu_H_ref = signal.convolve2d(HSI_ref[:,:,0], np.rot90(window, 2), mode='valid')
    mu_S_ref = signal.convolve2d(HSI_ref[:,:,1], np.rot90(window, 2), mode='valid')
    mu_H_pro = signal.convolve2d(HSI_pro[:,:,0], np.rot90(window, 2), mode='valid')
    mu_S_pro = signal.convolve2d(HSI_pro[:,:,1], np.rot90(window, 2), mode='valid')
    DeltaH = np.abs(mu_H_ref-mu_H_pro)
    DeltaS = np.abs(mu_S_ref-mu_S_pro)
    DeltaHS = w1*DeltaH+w2*DeltaS
    return DeltaHS, deltaHS


def cd11_delta_ascd(Ref_image, Pro_image, sizewin=3):
    M, N, K = Ref_image.shape
    DeltaASCD = np.zeros((M, N))
    half_blk = int(np.floor(sizewin / 2))
    samples = 3 * sizewin * sizewin
    A = np.zeros((samples, 6))
    RGB_vec_pro = np.zeros((samples, 1))
    WA = 0.1 * np.eye(6)
    WA[5, 5] = 0.5
    for jj in range(half_blk,N-half_blk):
        for ii in range(half_blk,M-half_blk):
            temp_ref = Ref_image[ii - half_blk:ii + half_blk+1, jj - half_blk:jj + half_blk+1,:]
            temp_pro = Pro_image[ii - half_blk:ii + half_blk+1, jj - half_blk:jj + half_blk+1,:]
            a1 = temp_ref[:,:, 0]
            A[0:samples:3, 0] = a1.ravel()
            a2 = temp_ref[:,:, 1]
            A[1:samples:3, 1] = a2.ravel()
            a3 = temp_ref[:,:, 2]
            A[1:samples:3, 2] = a3.ravel()
            RGB_vec_ref = A[:, 0] + A[:, 1] + A[:, 2]
            RGB_vec_pro[0:samples:3, 0] = temp_pro[:,:, 0].ravel()
            RGB_vec_pro[1:samples:3, 0] = temp_pro[:,:, 1].ravel()
            RGB_vec_pro[2:samples:3, 0] = temp_pro[:,:, 2].ravel()
            delta_rgb = RGB_vec_ref - RGB_vec_pro
            l = np.sum(temp_ref, 2) / 3
            A[0:samples:3, 3] = l.ravel()
            A[1:samples:3, 3] = l.ravel()
            A[2:samples:3, 3] = l.ravel()
            A[:, 4] = RGB_vec_ref - A[:, 3]
            A[0:samples:3, 5] = l.ravel()*(a3.ravel()-a2.ravel())
            A[1:samples:3, 5] = l.ravel()*(a1.ravel()-a3.ravel())
            A[2:samples:3, 5] = l.ravel()*(a2.ravel()-a1.ravel())
            An = np.tile(np.sqrt(np.sum(np.power(A, 2), 0)), (A.shape[0], 1))
            idx = np.where(A == 0)
            An[idx] = 1.
            A = A / An
            A[idx] = 0.
            deltaCA = np.linalg.solve((np.power(WA, 2) + np.dot(A.T,A)),(np.dot(A.T, delta_rgb)))
            DeltaASCD[ii, jj] = np.sum(np.power(np.dot(WA, deltaCA), 2)) +\
                                np.sum(np.power(delta_rgb - np.dot(A, deltaCA),2))
    deltaASCD = np.mean(DeltaASCD[half_blk:M - half_blk,half_blk:N - half_blk])
    return DeltaASCD, deltaASCD


def cd12_shameCIELAB(Ref_image, Pro_image):
    wi = np.array([[0.921, 0.105, -0.108], [0.531, 0.330, 0], [0.488, 0.371, 0]])
    si = np.array([[0.0283, 0.133, 4.336], [0.0392, 0.494, 0], [0.0536, 0.386, 0]])
    xx, yy = np.meshgrid(np.arange(-11, 12), np.arange(-11, 12))
    h1 = wi[0, 0] * myUtilities.gaussian(xx, yy, si[0, 0]) + wi[0, 1] * myUtilities.gaussian(xx, yy, si[0, 1]) + wi[0, 2] * myUtilities.gaussian(
        xx, yy, si[0, 2])
    h1 = h1 / np.sum(h1[:])
    h2 = wi[1, 0] * myUtilities.gaussian(xx, yy, si[1, 0]) + wi[1, 1] * myUtilities.gaussian(xx, yy, si[1, 1])
    h2 = h2 / np.sum(h2[:])
    h3 = wi[2, 0] * myUtilities.gaussian(xx, yy, si[2, 0]) + wi[2, 1] * myUtilities.gaussian(xx, yy, si[2, 1])
    h3 = h3 / np.sum(h3[:])
    XYZ_ref = colorSpaces.RGB2XYZ(Ref_image / 255.)
    XYZ_pro = colorSpaces.RGB2XYZ(Pro_image / 255.)
    O123_ref = colorSpaces.XYZ2O1O2O3(XYZ_ref)
    O123_pro = colorSpaces.XYZ2O1O2O3(XYZ_pro)
    O1_ref = signal.convolve2d(O123_ref[:, :, 0], np.rot90(h1, 2), mode='valid')
    O2_ref = signal.convolve2d(O123_ref[:, :, 1], np.rot90(h2, 2), mode='valid')
    O3_ref = signal.convolve2d(O123_ref[:, :, 2], np.rot90(h3, 2), mode='valid')
    O1_pro = signal.convolve2d(O123_pro[:, :, 0], np.rot90(h1, 2), mode='valid')
    O2_pro = signal.convolve2d(O123_pro[:, :, 1], np.rot90(h2, 2), mode='valid')
    O3_pro = signal.convolve2d(O123_pro[:, :, 2], np.rot90(h3, 2), mode='valid')
    XYZ_ref = colorSpaces.O1O2O32XYZ(np.dstack((O1_ref, O2_ref, O3_ref)))
    XYZ_pro = colorSpaces.O1O2O32XYZ(np.dstack((O1_pro, O2_pro, O3_pro)))
    RGB_ref_back = 255 * colorSpaces.XYZ2RGB(XYZ_ref)
    RGB_ref_back[RGB_ref_back > 255] = 255
    RGB_ref_back[RGB_ref_back < 0] = 0
    RGB_ref_back = np.uint8(RGB_ref_back)
    RGB_pro_back = 255 * colorSpaces.XYZ2RGB(XYZ_pro)
    RGB_pro_back[RGB_pro_back > 255] = 255
    RGB_pro_back[RGB_pro_back < 0] = 0
    RGB_pro_back = np.uint8(RGB_pro_back)
    return cd07_weighted_deltaE(RGB_ref_back, RGB_pro_back)


def cd13_cid(Ref_image, Pro_image):
    Img1 = Ref_image
    Img2 = Pro_image
    CyclesPerDegree = 20
    Img1_XYZ = colorSpaces.ImageSRGB2XYZ(Img1)
    Img1_filt = colorSpaces.scielab_simple(2 * CyclesPerDegree, Img1_XYZ)
    Img1_LAB2000HL = colorSpaces.ImageXYZ2LAB2000HL(Img1_filt)
    Img2_XYZ = colorSpaces.ImageSRGB2XYZ(Img2)
    Img2_filt = colorSpaces.scielab_simple(2 * CyclesPerDegree, Img2_XYZ)
    Img2_LAB2000HL = colorSpaces.ImageXYZ2LAB2000HL(Img2_filt)
    Window = myUtilities.matlab_style_gauss2D(shape=(11,11),sigma=1.5)
    img1 = Img1_LAB2000HL
    img2 = Img2_LAB2000HL
    L1 = img1[:,:, 0]
    A1 = img1[:,:, 1]
    B1 = img1[:,:, 2]
    Chr1_sq = np.power(A1, 2) + np.power(B1, 2)
    L2 = img2[:,:, 0]
    A2 = img2[:,:, 1]
    B2 = img2[:,:, 2]
    Chr2_sq = np.power(A2, 2) + np.power(B2, 2)
    muL1 = signal.convolve2d(L1, np.rot90(Window, 2), mode='valid')
    muC1 = signal.convolve2d(np.sqrt(Chr1_sq), np.rot90(Window, 2), mode='valid')
    muL2 = signal.convolve2d(L2, np.rot90(Window, 2), mode='valid')
    muC2 = signal.convolve2d(np.sqrt(Chr2_sq), np.rot90(Window, 2), mode='valid')
    sL1_sq = signal.convolve2d(np.power(L1,2), np.rot90(Window, 2), mode='valid') - np.power(muL1,2)
    sL1_sq[sL1_sq < 0] = 0
    sL1 = np.sqrt(sL1_sq)
    sL2_sq = signal.convolve2d(np.power(L2, 2), np.rot90(Window, 2), mode='valid') - np.power(muL2, 2)
    sL2_sq[sL2_sq < 0] = 0
    sL2 = np.sqrt(sL2_sq)
    dL_sq = np.power(muL1 - muL2,2)
    dC_sq = np.power(muC1 - muC2,2)
    Tem = np.sqrt(np.power(A1 - A2,2) + np.power(B1 - B2,2) -\
                  np.power(np.sqrt(Chr1_sq) - np.sqrt(Chr2_sq),2))
    dH_sq = np.power(signal.convolve2d(Tem, np.rot90(Window, 2), mode='valid'),2)
    sL12 = signal.convolve2d(L1*L2, np.rot90(Window, 2), mode='valid') - muL1*muL2
    IDFConsts = np.array([0.002, 0.1, 0.1, 0.002, 0.008])
    Maps_invL = 1 / (IDFConsts[0] * dL_sq + 1)
    Maps_invLc = (IDFConsts[1] + 2 * sL1 * sL2) / (IDFConsts[1] + sL1_sq + sL2_sq)
    Maps_invLs = (IDFConsts[2] + sL12) / (IDFConsts[2] + sL1 * sL2)
    Maps_invC = 1 / (IDFConsts[3] * dC_sq + 1)
    Maps_invH = 1 / (IDFConsts[4] * dH_sq + 1)
    IDF1 = np.nanmean(Maps_invL)
    IDF2 = np.nanmean(Maps_invLc)
    IDF3 = np.nanmean(Maps_invLs)
    IDF4 = np.nanmean(Maps_invC)
    IDF5 = np.nanmean(Maps_invH)
    prediction = np.real(1 - IDF1 * IDF2 * IDF3 * IDF4 * IDF5)
    Prediction = np.real(1 - Maps_invL * Maps_invLc * Maps_invLs * Maps_invC * Maps_invH)
    return Prediction, prediction


def cd14_circular_hue(Ref_image, Pro_image, mode='valid'):
    wi = np.array([[0.921, 0.105, -0.108], [0.531, 0.330, 0], [0.488, 0.371, 0]])
    si = np.array([[0.0283, 0.133, 4.336], [0.0392, 0.494, 0], [0.0536, 0.386, 0]])
    xx, yy = np.meshgrid(np.arange(-11, 12), np.arange(-11, 12))
    h1 = wi[0, 0] * myUtilities.gaussian(xx, yy, si[0, 0]) + wi[0, 1] * myUtilities.gaussian(xx, yy, si[0, 1]) + wi[0, 2] * myUtilities.gaussian(
        xx, yy, si[0, 2])
    h1 = h1 / np.sum(h1[:])
    h2 = wi[1, 0] * myUtilities.gaussian(xx, yy, si[1, 0]) + wi[1, 1] * myUtilities.gaussian(xx, yy, si[1, 1])
    h2 = h2 / np.sum(h2[:])
    h3 = wi[2, 0] * myUtilities.gaussian(xx, yy, si[2, 0]) + wi[2, 1] * myUtilities.gaussian(xx, yy, si[2, 1])
    h3 = h3 / np.sum(h3[:])
    XYZ_ref = colorSpaces.RGB2XYZ(Ref_image / 255.)
    XYZ_pro = colorSpaces.RGB2XYZ(Pro_image / 255.)
    O123_ref = colorSpaces.XYZ2O1O2O3(XYZ_ref)
    O123_pro = colorSpaces.XYZ2O1O2O3(XYZ_pro)
    O1_ref = signal.convolve2d(O123_ref[:, :, 0], np.rot90(h1, 2), mode=mode)
    O2_ref = signal.convolve2d(O123_ref[:, :, 1], np.rot90(h2, 2), mode=mode)
    O3_ref = signal.convolve2d(O123_ref[:, :, 2], np.rot90(h3, 2), mode=mode)
    O1_pro = signal.convolve2d(O123_pro[:, :, 0], np.rot90(h1, 2), mode=mode)
    O2_pro = signal.convolve2d(O123_pro[:, :, 1], np.rot90(h2, 2), mode=mode)
    O3_pro = signal.convolve2d(O123_pro[:, :, 2], np.rot90(h3, 2), mode=mode)
    XYZ_ref = colorSpaces.O1O2O32XYZ(np.dstack((O1_ref, O2_ref, O3_ref)))
    XYZ_pro = colorSpaces.O1O2O32XYZ(np.dstack((O1_pro, O2_pro, O3_pro)))
    RGB_ref_back = 255 * colorSpaces.XYZ2RGB(XYZ_ref)
    RGB_ref_back[RGB_ref_back > 255] = 255
    RGB_ref_back[RGB_ref_back < 0] = 0
    RGB_ref = np.uint8(RGB_ref_back)
    RGB_pro_back = 255 * colorSpaces.XYZ2RGB(XYZ_pro)
    RGB_pro_back[RGB_pro_back > 255] = 255
    RGB_pro_back[RGB_pro_back < 0] = 0
    RGB_pro = np.uint8(RGB_pro_back)
    Lab_ref = color.rgb2lab(RGB_ref)
    Lab_pro = color.rgb2lab(RGB_pro)
    H_ref = np.arctan2(Lab_ref[:,:, 2], Lab_ref[:,:, 1])
    H_pro = np.arctan2(Lab_pro[:,:, 2], Lab_pro[:,:, 1])
    sizewin = 11
    window = np.ones((sizewin, sizewin))
    Kh = (360*0.01)**2
    Kc = (180*0.01)**2
    H_ref_mean = np.arctan2(signal.convolve2d(np.cos(H_ref), np.rot90(window, 2), mode=mode), \
                          signal.convolve2d(np.sin(H_ref), np.rot90(window, 2), mode=mode))
    H_pro_mean = np.arctan2(signal.convolve2d(np.cos(H_pro), np.rot90(window, 2), mode=mode), \
                          signal.convolve2d(np.sin(H_pro), np.rot90(window, 2), mode=mode))
    DH = (2*H_ref_mean*H_pro_mean+Kh)\
         /(np.power(H_ref_mean,2)+np.power(H_pro_mean,2)+Kh)
    dH = np.mean(DH)
    C_ref = np.sqrt(np.power(Lab_ref[:,:,1],2)+np.power(Lab_ref[:,:,2],2))
    C_ref_mean = signal.convolve2d(C_ref, np.rot90(window, 2), mode=mode)
    C_pro = np.sqrt(np.power(Lab_pro[:,:,1],2)+np.power(Lab_pro[:,:,2],2))
    C_pro_mean = signal.convolve2d(C_pro, np.rot90(window, 2), mode=mode)
    DC = (2*C_ref_mean*C_pro_mean+Kc)\
         /(np.power(C_ref_mean,2)+np.power(C_pro_mean,2)+Kc)
    dC = np.mean(DC)
    DL, dL = miselaneusPack.SSIM(Lab_ref[:,:,0], Lab_pro[:,:,0],mode=mode)
    DE = 1-DH*DC*DL
    dE = 1-dH*dC*dL
    return DE, dE


def cd15_osa_ucsdE(Ref_image, Pro_image):
    Ljg_ref = colorSpaces.RGB2Ljg(Ref_image)
    Ljg_pro = colorSpaces.RGB2Ljg(Pro_image)
    C_ref = np.sqrt(np.power(Ljg_ref[:,:,1],2)+np.power(Ljg_ref[:,:,2],2))
    h_ref = np.arctan2(Ljg_ref[:,:,1],-Ljg_ref[:,:,2])
    C_pro = np.sqrt(np.power(Ljg_pro[:,:,1],2)+np.power(Ljg_pro[:,:,2],2))
    h_pro = np.arctan2(Ljg_pro[:,:,1],-Ljg_pro[:,:,2])
    S_L = 2.499+0.07*(Ljg_ref[:,:,0]+Ljg_pro[:,:,0])/2.
    S_C = 1.235+0.58*(C_ref+C_pro)/2
    S_H = 1.392+0.17*(h_ref+h_pro)/2
    dL = (Ljg_ref[:,:,0]-Ljg_pro[:,:,0])/S_L
    dC = (C_ref-C_pro)/S_C
    dh = (h_ref-h_pro)/S_H
    DE = 10*np.sqrt(np.power(dL,2)+np.power(dC,2)+np.power(dh,2))
    dE = np.nanmean(DE)
    return DE, dE


def cd16_Sosa_ucsdE(Ref_image, Pro_image):
    wi = np.array([[0.921, 0.105, -0.108], [0.531, 0.330, 0], [0.488, 0.371, 0]])
    si = np.array([[0.0283, 0.133, 4.336], [0.0392, 0.494, 0], [0.0536, 0.386, 0]])
    xx, yy = np.meshgrid(np.arange(-11, 12), np.arange(-11, 12))
    h1 = wi[0, 0] * myUtilities.gaussian(xx, yy, si[0, 0]) + wi[0, 1] * myUtilities.gaussian(xx, yy, si[0, 1]) + wi[0, 2] * myUtilities.gaussian(
        xx, yy, si[0, 2])
    h1 = h1 / np.sum(h1[:])
    h2 = wi[1, 0] * myUtilities.gaussian(xx, yy, si[1, 0]) + wi[1, 1] * myUtilities.gaussian(xx, yy, si[1, 1])
    h2 = h2 / np.sum(h2[:])
    h3 = wi[2, 0] * myUtilities.gaussian(xx, yy, si[2, 0]) + wi[2, 1] * myUtilities.gaussian(xx, yy, si[2, 1])
    h3 = h3 / np.sum(h3[:])
    XYZ_ref = colorSpaces.RGB2XYZ(Ref_image / 255.)
    XYZ_pro = colorSpaces.RGB2XYZ(Pro_image / 255.)
    O123_ref = colorSpaces.XYZ2O1O2O3(XYZ_ref)
    O123_pro = colorSpaces.XYZ2O1O2O3(XYZ_pro)
    O1_ref = signal.convolve2d(O123_ref[:, :, 0], np.rot90(h1, 2), mode='valid')
    O2_ref = signal.convolve2d(O123_ref[:, :, 1], np.rot90(h2, 2), mode='valid')
    O3_ref = signal.convolve2d(O123_ref[:, :, 2], np.rot90(h3, 2), mode='valid')
    O1_pro = signal.convolve2d(O123_pro[:, :, 0], np.rot90(h1, 2), mode='valid')
    O2_pro = signal.convolve2d(O123_pro[:, :, 1], np.rot90(h2, 2), mode='valid')
    O3_pro = signal.convolve2d(O123_pro[:, :, 2], np.rot90(h3, 2), mode='valid')
    XYZ_ref = colorSpaces.O1O2O32XYZ(np.dstack((O1_ref, O2_ref, O3_ref)))
    XYZ_pro = colorSpaces.O1O2O32XYZ(np.dstack((O1_pro, O2_pro, O3_pro)))
    RGB_ref_back = 255 * colorSpaces.XYZ2RGB(XYZ_ref)
    RGB_ref_back[RGB_ref_back > 255] = 255
    RGB_ref_back[RGB_ref_back < 0] = 0
    RGB_ref = np.uint8(RGB_ref_back)
    RGB_pro_back = 255 * colorSpaces.XYZ2RGB(XYZ_pro)
    RGB_pro_back[RGB_pro_back > 255] = 255
    RGB_pro_back[RGB_pro_back < 0] = 0
    RGB_pro = np.uint8(RGB_pro_back)
    return cd15_osa_ucsdE(RGB_ref, RGB_pro)


def cd17_localdeltaE2000(Ref_image, Pro_image):
    DE00, _ = cd00_deltaE2000(Ref_image, Pro_image)
    Weight = np.array([[0.5, 1., 0.5], [1., 0., 1.], [0.5, 1., 0.5]])
    Center = np.array([[0., 0., 0.], [0., 1., 0.], [0., 0., 0.]])
    DEW = signal.convolve2d(DE00, np.rot90(Weight, 2), mode='valid')
    DE00 = signal.convolve2d(DE00, np.rot90(Center, 2), mode='valid')
    DE00 = (DE00 + DEW) / 7
    return DE00, np.mean(DE00)


def cd18_sprext_patches(Ref_image, Pro_image, th=0.02, r=1., min_num_pixels=4, sq=False):
    SE = np.ones((np.int_(2 * r + 1), np.int_(2 * r + 1)))
    Yref = 0.299 * Ref_image[:, :, 0] + 0.587 * Ref_image[:, :, 1] + 0.114 * Ref_image[:, :, 2]
    Ypro = 0.299 * Pro_image[:, :, 0] + 0.587 * Pro_image[:, :, 1] + 0.114 * Pro_image[:, :, 2]
    DL, _ = miselaneusPack.ssim(Yref, Ypro)
    dl = 0.
    # SE = np.ones((2, 2))
    YCbCr_ref = colorSpaces.RGB2YCbCr(Ref_image)
    YCbCr_pro = colorSpaces.RGB2YCbCr(Pro_image)
    ECbCr = np.sqrt(np.power(YCbCr_ref[:, :, 1] - YCbCr_pro[:, :, 1], 2) + \
                    np.power(YCbCr_ref[:, :, 2] - YCbCr_pro[:, :, 2], 2))
    LBP, _ = computeTexture.lbp(Ref_image, th=th, r=r)
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


def cd19_SSIMipt(Ref_image, Pro_image, mode='valid'):
    iCAM_ref = colorSpaces.RGB2iCAM(Ref_image)  # ipt_image
    iCAM_pro = colorSpaces.RGB2iCAM(Pro_image)
    iCAM_ref[np.where(np.isnan(iCAM_ref))] = 0
    iCAM_pro[np.where(np.isnan(iCAM_pro))] = 0
    ssim_map_i, mssim_i = miselaneusPack.SSIM(iCAM_ref[:, :, 0], iCAM_pro[:, :, 0])
    ssim_map_p, mssim_p = miselaneusPack.SSIM(iCAM_ref[:, :, 1], iCAM_pro[:, :, 1])
    ssim_map_t, mssim_t = miselaneusPack.SSIM(iCAM_ref[:, :, 2], iCAM_pro[:, :, 2])
    return (ssim_map_i * ssim_map_p * ssim_map_t), (mssim_i * mssim_p * mssim_t)


def cd20_cPSNRHA(Ref_image, Pro_image, wsize=11):
    YCbCr_ref = colorSpaces.RGB2YCbCr(Ref_image)
    YCbCr_pro = colorSpaces.RGB2YCbCr(Pro_image)
    muYref = np.mean(YCbCr_ref[:, :, 0])
    muCbref = np.mean(YCbCr_ref[:, :, 1])
    muCrref = np.mean(YCbCr_ref[:, :, 2])
    muYpro = np.mean(YCbCr_pro[:, :, 0])
    muCbpro = np.mean(YCbCr_pro[:, :, 1])
    muCrpro = np.mean(YCbCr_pro[:, :, 2])
    muDeltaY = muYref - muYpro
    muDeltaCb = muCbref - muCbpro
    muDeltaCr = muCrref - muCrpro
    CYpro = YCbCr_pro[:, :, 0] + muDeltaY
    CCbpro = YCbCr_pro[:, :, 1] + muDeltaCb
    CCrpro = YCbCr_pro[:, :, 2] + muDeltaCr
    muCYpro = np.mean(CYpro)
    muCCbpro = np.mean(CCbpro)
    muCCrpro = np.mean(CCrpro)
    PY = np.sum((YCbCr_ref[:, :, 0]-muYref)*(CYpro-muCYpro))/np.sum(np.power((CYpro-muCYpro), 2))
    PCb = np.sum((YCbCr_ref[:, :, 1] - muCbref) * (CCbpro - muCCbpro))/np.sum(np.power((CCbpro - muCCbpro), 2))
    PCr = np.sum((YCbCr_ref[:, :, 2] - muCrref) * (CCrpro - muCCrpro))/np.sum(np.power((CCrpro - muCCrpro), 2))
    DY = PY * CYpro
    DCb = PCb * CCbpro
    DCr = PCr * CCrpro
    _, p_hvs_m_YCHMA, _, YCHMA = myUtilities.dct_block_mse(YCbCr_ref[:, :, 0], CYpro)
    _, p_hvs_m_CbCHMA, _, CbCHMA = myUtilities.dct_block_mse(YCbCr_ref[:, :, 1], CCbpro)
    _, p_hvs_m_CrCHMA, _, CrCHMA = myUtilities.dct_block_mse(YCbCr_ref[:, :, 2], CCrpro)
    _, p_hvs_m_YDHMA, _, YDHMA = myUtilities.dct_block_mse(YCbCr_ref[:, :, 0], DY)
    _, p_hvs_m_CbDHMA, _, CbDHMA = myUtilities.dct_block_mse(YCbCr_ref[:, :, 1], DCb)
    _, p_hvs_m_CrDHMA, _, CrDHMA = myUtilities.dct_block_mse(YCbCr_ref[:, :, 2], DCr)
    if PY < 1:
        c = 0.002
    else:
        c = 0.25
    if p_hvs_m_YCHMA > p_hvs_m_YDHMA:
        p_hvs_m_YCHMA = p_hvs_m_YDHMA + (p_hvs_m_YCHMA - p_hvs_m_YDHMA) * c
        YCHMA = YDHMA + (YCHMA - YDHMA) * c
    if p_hvs_m_CbCHMA > p_hvs_m_CbDHMA:
        p_hvs_m_CbCHMA = p_hvs_m_CbDHMA + (p_hvs_m_CbCHMA - p_hvs_m_CbDHMA) * c
        CbCHMA = CbDHMA + (CbDHMA - CbCHMA) * c
    if p_hvs_m_CrCHMA > p_hvs_m_CrDHMA:
        p_hvs_m_CrCHMA = p_hvs_m_CrDHMA + (p_hvs_m_CrCHMA - p_hvs_m_CrDHMA) * c
        CrCHMA = CrDHMA + (CrDHMA - CrCHMA) * c
    p_hvs_m_YCHMA += muDeltaY * muDeltaY * 0.04
    p_hvs_m_CbCHMA += muDeltaCb * muDeltaCb * 0.04
    p_hvs_m_CrCHMA += muDeltaCr * muDeltaCr * 0.04
    p_hvs_m_YCHMA = myUtilities.clip_psnr(p_hvs_m_YCHMA)
    p_hvs_m_CbCHMA = myUtilities.clip_psnr(p_hvs_m_CbCHMA)
    p_hvs_m_CrCHMA = myUtilities.clip_psnr(p_hvs_m_CrCHMA)
    cpsnrhma = (p_hvs_m_YCHMA + p_hvs_m_CbCHMA * 0.5 + p_hvs_m_CrCHMA * 0.5) / (1 + 2 * 0.5)
    cPSNRHMA = (YCHMA + CbCHMA * 0.5 + CrCHMA * 0.5) / (1 + 2 * 0.5)
    cPSNRHMA[np.where(np.isnan(cPSNRHMA))] = 100000.
    return cPSNRHMA, cpsnrhma


def cd21_vsi(Ref_image, Pro_image):
    image1 = Ref_image
    image2 = Pro_image
    constForVS = 1.27
    constForGM = 386
    constForChrom = 130
    alpha = 0.40
    lambda_ = 0.020
    sigmaF = 1.34
    omega0 = 0.0210
    sigmaD = 145
    sigmaC = 0.001
    saliencyMap1 = myUtilities.SDSP(image1, sigmaF, omega0, sigmaD, sigmaC)
    saliencyMap2 = myUtilities.SDSP(image2, sigmaF, omega0, sigmaD, sigmaC)

    rows, cols, _ = image1.shape
    L1 = 0.06 * image1[:,:, 0] + 0.63 * image1[:,:, 1] + 0.27 * image1[:,:, 2]
    L2 = 0.06 * image2[:,:, 0] + 0.63 * image2[:,:, 1] + 0.27 * image2[:,:, 2]
    M1 = 0.30 * image1[:,:, 0] + 0.04 * image1[:,:, 1] - 0.35 * image1[:,:, 2]
    M2 = 0.30 * image2[:,:, 0] + 0.04 * image2[:,:, 1] - 0.35 * image2[:,:, 2]
    N1 = 0.34 * image1[:,:, 0] - 0.60 * image1[:,:, 1] + 0.17 * image1[:,:, 2]
    N2 = 0.34 * image2[:,:, 0] - 0.60 * image2[:,:, 1] + 0.17 * image2[:,:, 2]

    minDimension = np.minimum(rows, cols)
    F = np.maximum(1, np.round(minDimension / 256))
    aveKernel = np.ones((F,F))/(F * F)

    aveM1 = signal.convolve2d(M1, np.rot90(aveKernel, 2), mode='same')
    aveM2 = signal.convolve2d(M2, np.rot90(aveKernel, 2), mode='same')

    M1 = aveM1[0:rows:F, 0:cols:F]
    M2 = aveM2[0:rows:F, 0:cols:F]


    aveN1 = signal.convolve2d(N1, np.rot90(aveKernel, 2), mode='same')
    aveN2 = signal.convolve2d(N2, np.rot90(aveKernel, 2), mode='same')
    N1 = aveN1[0:rows:F, 0:cols:F]
    N2 = aveN2[0:rows:F, 0:cols:F]

    aveL1 = signal.convolve2d(L1, np.rot90(aveKernel, 2), mode='same')
    aveL2 = signal.convolve2d(L2, np.rot90(aveKernel, 2), mode='same')
    L1 = aveL1[0:rows:F, 0:cols:F]
    L2 = aveL2[0:rows:F, 0:cols:F]

    aveSM1 = signal.convolve2d(saliencyMap1, np.rot90(aveKernel, 2), mode='same')
    aveSM2 = signal.convolve2d(saliencyMap2, np.rot90(aveKernel, 2), mode='same')
    saliencyMap1 = aveSM1[0:rows:F, 0:cols:F]
    saliencyMap2 = aveSM2[0:rows:F, 0:cols:F]

    dx = np.array([[3, 0, -3], [10, 0, -10], [3, 0, -3]])/16.
    dy = np.array([[3, 10, 3], [0, 0, 0], [-3, -10, -3]])/16.

    IxL1 = signal.convolve2d(L1, np.rot90(dx, 2), mode='same')
    IyL1 = signal.convolve2d(L1, np.rot90(dy, 2), mode='same')
    gradientMap1 = np.sqrt(np.power(IxL1, 2) + np.power(IyL1, 2))

    IxL2 = signal.convolve2d(L2, np.rot90(dx, 2), mode='same')
    IyL2 = signal.convolve2d(L2, np.rot90(dy, 2), mode='same')
    gradientMap2 = np.sqrt(np.power(IxL2, 2) + np.power(IyL2, 2))

    VSSimMatrix = (2 * saliencyMap1 * saliencyMap2 + constForVS)/\
                  (np.power(saliencyMap1, 2) + np.power(saliencyMap2, 2) + constForVS)
    gradientSimMatrix = (2 * gradientMap1 * gradientMap2 + constForGM)/\
                        (np.power(gradientMap1, 2) + np.power(gradientMap2, 2) + constForGM)

    weight = np.maximum(saliencyMap1, saliencyMap2)

    ISimMatrix = (2 * M1 * M2 + constForChrom) / (np.power(M1, 2) + np.power(M2, 2) + constForChrom)
    QSimMatrix = (2 * N1 * N2 + constForChrom) / (np.power(N1, 2) + np.power(N2, 2) + constForChrom)

    prod = ISimMatrix*QSimMatrix
    prod[np.where(prod < 0)] = 0
    SimMatrixC = (np.power(gradientSimMatrix, alpha))*VSSimMatrix*np.real(np.power(prod, lambda_))*weight
    sim = np.nansum(SimMatrixC) / np.nansum(weight)
    SimMatrixC[np.where(np.isnan(SimMatrixC))] = 0
    return SimMatrixC, sim


def cd22_deltaEab(Ref_image, Pro_image):
    Lab_ref = color.rgb2lab(Ref_image)
    Lab_pro = color.rgb2lab(Pro_image)
    Lstd = Lab_ref[:, :, 0]
    astd = Lab_ref[:, :, 1]
    bstd = Lab_ref[:, :, 2]
    Lsample = Lab_pro[:, :, 0]
    asample = Lab_pro[:, :, 1]
    bsample = Lab_pro[:, :, 2]
    DE00 = np.sqrt(np.power(Lstd-Lsample, 2) + np.power(astd-asample, 2) + np.power(bstd-bsample, 2))
    de00 = np.mean(DE00[:])
    return DE00, de00
