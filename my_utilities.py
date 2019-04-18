import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from scipy import ndimage
from scipy import stats
from scipy import spatial
from scipy import io
from scipy import misc
from scipy import fftpack
from skimage import color
import cython_functions
import pywt
from matplotlib.figure import Figure


def format_time(time_seconds):
    h = time_seconds/3600.
    m = 60.*(h-int(h))
    s = 60.*(m-int(m))
    return "H " + str(int(h)) + ": M " + str(int(m)) + ": S " + str(int(s))

def checkifRGB(RGB_image):
    if len(RGB_image.shape) > 2:
        return 0.299 * RGB_image[:, :, 0] + 0.587 * RGB_image[:, :, 1] + 0.114 * RGB_image[:, :, 2]
    else:
        return 1. * RGB_image


def gaussian(x, y, sigma):
    F = np.exp(-((x * x + y * y) / (sigma * sigma)))
    return F / np.sum(F[:])


def number_neighbour(R=1.5):
    if R == 1:
        N = 8
        Scale = 1
    elif R == 1.5:
        N = 12
        Scale = 1 / 2
    elif R == 2:
        N = 16
        Scale = 1 / 4
    elif R == 3:
        N = 24
        Scale = 1 / 8
    var = io.loadmat('LookatTablesLBP.mat')
    Table = var['look_table_' + str(N)]
    return N, Table, Scale


def SSIM(img1, img2, K=np.array([0.01, 0.03]), wsize=11, mode='valid'):
    xx, yy = np.meshgrid(np.arange(-wsize / 2, wsize / 2), np.arange(-wsize / 2, wsize / 2))
    window = gaussian(xx, yy, 1.5)
    C1 = (K[0] * 255) ** 2
    C2 = (K[0] * 255) ** 2
    mu1 = signal.convolve2d(img1, np.rot90(window, 2), mode=mode)
    mu2 = signal.convolve2d(img2, np.rot90(window, 2), mode=mode)
    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = signal.convolve2d(img1 * img1, np.rot90(window, 2), mode=mode) - mu1_sq
    sigma2_sq = signal.convolve2d(img2 * img2, np.rot90(window, 2), mode=mode) - mu2_sq
    sigma12 = signal.convolve2d(img1 * img2, np.rot90(window, 2), mode=mode) - mu1_mu2
    if C1 > 0 and C2 > 0:
        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    else:
        numerator1 = 2 * mu1_mu2 + C1
        numerator2 = 2 * sigma12 + C2
        denominator1 = mu1_sq + mu2_sq + C1
        denominator2 = sigma1_sq + sigma2_sq + C2
        ssim_map = np.ones_like(mu1)
        index = denominator1 * denominator2 > 0
        ssim_map[index] = (numerator1[index] * numerator2[index]) / (denominator1[index] * denominator2[index])
        index = (denominator1 != 0) & (denominator2 == 0)
        ssim_map[index] = numerator1[index] / denominator1[index]
    return ssim_map, np.mean(ssim_map[:])


def meanblock(Cb, Cr):
    window = np.ones((9, 9))
    window[:, 0] = 0
    window[0, :] = 0
    window = window / (8 * 8)
    MuCb = signal.convolve2d(Cb, np.rot90(window, 2), mode='valid')
    mCb = MuCb[0::8, 0::8]
    MuCr = signal.convolve2d(Cr, np.rot90(window, 2), mode='valid')
    mCr = 1.5 * MuCr[0::8, 0::8]
    return mCb, mCr, MuCb, MuCr


def colorhistogram(Image, minc=np.array([0, 0, 0]), maxc=np.array([255, 255, 255]), nbins=8):
    Ivec = np.array([convert_vec(Image[:, :, 0]), convert_vec(Image[:, :, 1]), convert_vec(Image[:, :, 2])])
    xx = np.linspace(minc[0], maxc[0], nbins + 1)
    yy = np.linspace(minc[1], maxc[1], nbins + 1)
    zz = np.linspace(minc[2], maxc[2], nbins + 1)
    H, _ = np.histogramdd(np.transpose(Ivec), (xx, yy, zz))
    return H


def colorhistogramplusedeges(Image, minc=np.array([0, 0, 0]), maxc=np.array([255, 255, 255]), nbins=8):
    Ivec = np.array([convert_vec(Image[:, :, 0]), convert_vec(Image[:, :, 1]), convert_vec(Image[:, :, 2])])
    xx = np.linspace(minc[0], maxc[0], nbins + 1)
    yy = np.linspace(minc[1], maxc[1], nbins + 1)
    zz = np.linspace(minc[2], maxc[2], nbins + 1)
    H, edges = np.histogramdd(np.transpose(Ivec), (xx, yy, zz))
    return H, edges


def IhistogramintersectionLab(I1, I2):
    H_1 = colorhistogram(I1, minc=np.array([0, -128, -128]), maxc=np.array([100, 127, 127]), nbins=8)
    H_2 = colorhistogram(I2, minc=np.array([0, -128, -128]), maxc=np.array([100, 127, 127]), nbins=8)
    H_1 = 1. * H_1 / np.sum(H_1[:])
    H_2 = 1. * H_2 / np.sum(H_2[:])
    return np.sum(np.minimum(H_1[:], H_2[:]))


def convert_vec(M):
    return np.reshape(M, M.size)


def block_proc(A, blk_size, C=None, funname='np.mean'):
    M, N, _ = A.shape
    B = np.zeros((M, N))
    for ii in xrange(0, M, blk_size):
        for jj in xrange(0, N, blk_size):
            if ii + blk_size <= M and jj + blk_size <= N:
                values = A[ii:ii + blk_size, jj:jj + blk_size, :]
                if C is None:
                    B[ii:ii + blk_size, jj:jj + blk_size] = eval(funname)(values[:])
                else:
                    values1 = C[ii:ii + blk_size, jj:jj + blk_size, :]
                    B[ii:ii + blk_size, jj:jj + blk_size] = eval(funname)(values, values1)
    return B


def rho_cd09(muY):
    rho_muY = np.zeros_like(muY)
    rho_muY[muY <= 6] = 0.06
    rho_muY[np.logical_and(muY > 6, muY <= 100)] = 0.04
    rho_muY[np.logical_and(muY > 100, muY <= 140)] = 0.01
    rho_muY[muY > 140] = 0.03
    return rho_muY


def blocking(I):
    """Calculates a horizontal blocking metric for an image I (mxn)"""
    N = 7
    windowSize = 256
    som = np.zeros(I.shape, dtype=np.float32)
    aux = np.asarray(np.empty(I.shape), dtype=np.float32)
    for i in range(-N, N + 1):
        if i != 0:
            np.subtract(np.roll(I, -i + 1, axis=1), np.roll(I, -i, axis=1), out=aux)
            np.fabs(aux, out=aux)
            np.add(som, aux, out=som)
    som = som / 2 / N
    Dh = np.fabs(np.roll(I, 1, axis=1) - I)
    old_settings = np.seterr()
    np.seterr(all='ignore')
    np.divide(Dh, som, out=Dh)
    np.seterr(**old_settings)
    Dh[np.isnan(Dh)] = 0
    np.minimum(Dh, 10, out=Dh)
    Sh = np.mean(Dh, axis=0)
    Sh[0:N - 1] = 0
    Sh[len(Sh) - N:len(Sh)] = 0
    f, Pxx = signal.welch(Sh, fs=1, window=signal.get_window('hamming', windowSize))
    H = np.ones(2 * N)
    medianPxx = ndimage.filters.median_filter(Pxx, size=None, footprint=H)
    pos1 = np.argmin(np.fabs(f - 1. / 8))
    pos2 = np.argmin(np.fabs(f - 2. / 8))
    pos3 = np.argmin(np.fabs(f - 3. / 8))
    return (Pxx[pos1] - medianPxx[pos1]) + (Pxx[pos2] - medianPxx[pos2]) + (Pxx[pos3] - medianPxx[pos3])


def EstimateNoise_SBLE(cD):
    """Computes means in blocks of size 8"""
    B = 8
    alpha = 0.1
    corrFactor = 0.716
    window = np.ones((B + 1, B + 1), dtype=np.float32)
    window[:, 0] = 0.0
    window[0, :] = 0.0
    window = window / (B * B)
    aux = np.asarray(np.empty(cD.shape), dtype=np.float32)
    np.square(cD, out=cD)
    ndimage.filters.convolve(cD, window, output=aux)
    blockEn = aux[B / 2:-(B / 2 - 1):B, B / 2:-(B / 2 - 1):B]
    blockEn = np.sort(blockEn, axis=None)
    avgBlockEn = 1 / corrFactor * np.mean(blockEn[0:np.int(np.floor(alpha * (len(blockEn) - 1)))])
    return aux, np.sqrt(avgBlockEn)


def wt2_mz(I, s):
    """Computes non redundant dwt"""
    Detail_Filt = np.asarray([0.0, 2.0, -2.0, 0.0, 0.0], dtype=np.float32)
    Lowpass_Filt = np.asarray([0.125, 0.375, 0.375, 0.125, 0], dtype=np.float32)
    t = 2 ** s;
    Dh = np.zeros((1, (len(Detail_Filt) - 1) * t + 1), dtype=np.float32)
    Ah = np.zeros((1, (len(Lowpass_Filt) - 1) * t + 1), dtype=np.float32)
    A = np.asarray(np.empty(I.shape), dtype=np.float32)
    H = np.asarray(np.empty(I.shape), dtype=np.float32)
    V = np.asarray(np.empty(I.shape), dtype=np.float32)
    aux = np.asarray(np.empty(I.shape), dtype=np.float32)
    for i in range(0, len(Detail_Filt)):
        Dh[0, i * t] = Detail_Filt[i]
    for i in range(0, len(Lowpass_Filt)):
        Ah[0, i * t] = Lowpass_Filt[i]
    ndimage.filters.convolve(I, Ah, output=aux)
    ndimage.filters.convolve(aux, Ah.T, output=A)
    ndimage.filters.convolve(I, Dh, output=H)
    ndimage.filters.convolve(I, Dh.T, output=V)
    return A, H, V


def UnshiftMZwavelet(D, shift, direction):
    """Shifts wavelet coefficients"""
    dims = D.shape
    if direction == 1:
        De = np.asarray(np.empty([dims[0], dims[1] + shift]), dtype=np.float32)
        De[:, shift:dims[1] + shift] = D
        De[:, 0:shift] = D[:, dims[1] - shift:dims[1]]
        D_unshift = De[:, 0:dims[1]]
    else:
        De = np.asarray(np.empty([dims[0] + shift, dims[1]]), dtype=np.float32)
        De[shift:dims[0] + shift, :] = D
        De[0:shift, :] = D[dims[0] - shift:dims[0], :]
        D_unshift = De[0:dims[0], :]
    return D_unshift


def calculate_wavelet_images(I):
    """Computes wavelet coefficients for blurring metric"""
    A1, Dx1, Dy1 = wt2_mz(I, 0)
    A2, Dx2, Dy2 = wt2_mz(A1, 1)
    _, Dx3, Dy3 = wt2_mz(A2, 2)
    Dx1_u = UnshiftMZwavelet(Dx1, 1, 1)
    Dx2_u = UnshiftMZwavelet(Dx2, 2, 1)
    Dx3_u = UnshiftMZwavelet(Dx3, 4, 1)
    Dy1_u = UnshiftMZwavelet(Dy1, 1, 2)
    Dy2_u = UnshiftMZwavelet(Dy2, 2, 2)
    Dy3_u = UnshiftMZwavelet(Dy3, 4, 2)
    return Dx1_u, Dx2_u, Dx3_u, Dy1_u, Dy2_u, Dy3_u


def ConesMagSum(D, K1, K2, Dir):
    """Computes magnitudes in cones"""
    dims = D.shape
    if K1 > K2:
        hr = np.zeros([1, K1], dtype=np.float32)
        for i in range(0, K2):
            hr[0, i] = 1
        hl = np.ones([1, K1], dtype=np.float32)
    elif K2 > K1:
        hl = np.zeros([1, K2], dtype=np.float32)
        for i in range(0, K1):
            hl[0, i] = 1
        hl = hl[::-1]
        hr = np.ones([1, K2], dtype=np.float32)
    else:
        hr = np.ones([1, K2], dtype=np.float32)
        hl = np.ones([1, K1], dtype=np.float32)
    h = np.concatenate((hl, np.ones([1, 1], dtype=np.float32), hr), axis=1)
    aux = np.asarray(np.empty(dims), dtype=np.float32)
    C = np.asarray(np.empty(dims), dtype=np.float32)
    np.fabs(D, out=aux)
    if Dir == 1:
        h = np.rot90(h, k=2)
        ndimage.filters.convolve(aux, h, output=C)
        np.add(C, D, out=C)
        C[:, 0:K1] = 0
        C[:, dims[1] - K2 - 1:dims[1]] = 0
    else:
        h = np.rot90(h, k=2)
        ndimage.filters.convolve(aux, h.T, output=C)
        np.add(C, D, out=C)
        C[0:K1, :] = 0.0
        C[dims[0] - K2 - 1:dims[0], :] = 0.0
    return C


def Ratio_sector_row(Mx1, C1, C2, C3, C4, s):
    """Computes ratio sector row"""
    row, col = np.nonzero(Mx1 != 0.0)
    y = np.asarray(np.zeros([1, len(row)]), dtype=np.float64)
    temp = 1
    if s == 2:
        for i in range(0, len(row)):
            a = C1[row[i], col[i]]
            b = C2[row[i], col[i]]
            if a == 0 or b == 0:
                y[0, i] = temp
            else:
                y[0, i] = np.fabs(b / a)
                temp = y[0, i]
    elif s == 3:
        for i in range(0, len(row)):
            a = C1[row[i], col[i]]
            b = C2[row[i], col[i]]
            c = C3[row[i], col[i]]
            if a == 0 or b == 0 or c == 0:
                y[0, i] = temp
            else:
                y[0, i] = (np.fabs(c / b) + np.fabs(b / a)) / 2
                temp = y[0, i]
    else:
        for i in range(0, len(row)):
            a = C1[row[i], col[i]]
            b = C2[row[i], col[i]]
            c = C3[row[i], col[i]]
            d = C4[row[i], col[i]]
            if a == 0 or b == 0 or c == 0 or d == 0:
                y[0, i] = temp
            else:
                y[0, i] = (np.fabs(d / c) + np.fabs(c / b) + np.fabs(b / a)) / 3;
                temp = y[0, i]
    y = np.log(y) / np.log(2)
    return y


def lj_FindACRhists(X, Mx2, My2):
    "Find ACR and ACR histograms for the image ACR"
    OFFSET = 16
    x2 = np.linspace(-10, 15, num=251, endpoint=True)
    x2 = np.append(x2, np.inf)
    dims = X.shape
    Mx2 = Mx2[OFFSET - 1:dims[0], OFFSET - 1:dims[1]]
    My2 = My2[OFFSET - 1:dims[0], OFFSET - 1:dims[1]]
    A1, _, _ = wt2_mz(X, 0)
    A2, Dx2, Dy2 = wt2_mz(A1, 1)
    A3, Dx3, Dy3 = wt2_mz(A2, 2)
    _, Dx4, Dy4 = wt2_mz(A3, 3)
    lambda2 = 1.12
    lambda3 = 1.03
    lambda4 = 1.01
    np.multiply(Dx2, 1 / lambda2, out=Dx2)
    np.multiply(Dx3, 1 / lambda3, out=Dx3)
    np.multiply(Dx4, 1 / lambda4, out=Dx4)
    np.multiply(Dy2, 1 / lambda2, out=Dy2)
    np.multiply(Dy3, 1 / lambda3, out=Dy3)
    np.multiply(Dy4, 1 / lambda4, out=Dy4)
    c2 = ConesMagSum(Dx2, 3, 1, 1)
    c3 = ConesMagSum(Dx3, 7, 1, 1)
    c4 = ConesMagSum(Dx4, 15, 1, 1)
    c2 = c2[OFFSET - 1:dims[0], OFFSET - 1:dims[1]]
    c3 = c3[OFFSET - 1:dims[0], OFFSET - 1:dims[1]]
    c4 = c4[OFFSET - 1:dims[0], OFFSET - 1:dims[1]]
    y23 = Ratio_sector_row(Mx2, c2, c3, c4, c4, 2)
    n23, _ = np.histogram(y23, bins=x2)
    n23s = n23
    c2 = ConesMagSum(Dy2, 3, 1, 2)
    c3 = ConesMagSum(Dy3, 7, 1, 2)
    c4 = ConesMagSum(Dy4, 15, 1, 2)
    c2 = c2[OFFSET - 1:dims[0], OFFSET - 1:dims[1]]
    c3 = c3[OFFSET - 1:dims[0], OFFSET - 1:dims[1]]
    c4 = c4[OFFSET - 1:dims[0], OFFSET - 1:dims[1]]
    y23 = Ratio_sector_row(My2, c2, c3, c4, c4, 2)
    n23, _ = np.histogram(y23, bins=x2)
    n23s += n23
    sum23 = np.sum(n23s)
    x2 = np.delete(x2, len(x2) - 1)
    d = (np.amax(x2) - np.amin(x2)) / 100.0
    return x2, n23s / (sum23 * d)


def get_CogACR(x2, apdf):
    """Get the magnitude of CogACR"""
    np.multiply(x2, apdf, out=x2)
    return np.sum(x2) / np.sum(apdf)


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
            c = np.log(np.abs(Imax - Imin) / (Imin))
            if not np.isinf(c):
                wme += c
                n += 1
                C[ii:ii + blk_size[0], jj:jj + blk_size[1]] = c
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
            c = np.log((Imax - Imin) / (Imax + Imin))
            if not np.isinf(c):
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
    C[0] = cython_functions.crm_convolution(Yref, 3)
    cglobal = np.mean(C[0])
    A = Yref
    C_total = C[0]
    for ii in range(1, Numfilters + 1):
        # sigma = 2. ** (ii-1)
        # xymax = np.maximum(np.abs(nstds * sigma), np.abs(nstds * sigma))
        # xx, yy = np.meshgrid(np.arange(-xymax, xymax + 1), np.arange(-xymax, xymax + 1))
        # g = gaussian(xx, yy, sigma)
        # A = signal.convolve2d(A, np.rot90(g, 2), mode='same')
        A = A[0::2, 0::2]
        C[ii] = cython_functions.crm_convolution(A, 3)
        cglobal += np.mean(C[ii])
        C_total += misc.imresize(C[ii], Yref.shape)
    return C_total / (Numfilters + 1.), cglobal / (Numfilters + 1.)


def Isodata(I, precision=1e-6):
    T = np.mean(I)
    Tanterior = 0.
    mud = np.array([0, 0])
    while np.abs(T - Tanterior) > precision:
        Tanterior = T
        if np.sum(I < T) == 0:
            mud[0] = np.min(I)
        else:
            mud[0] = np.mean(I[I < T])
        if np.sum(I >= T) == 0:
            mud[1] = np.max(I)
        else:
            mud[1] = np.mean(I[I >= T])
        T = np.sum(mud) / 2
    return T


def dis_correlation(x, y):
    n = x.size
    a = np.abs(x[:, None] - x)
    b = np.abs(y[:, None] - y)
    A = a - np.mean(a, axis=0) - np.mean(a, axis=1)[:, None] + np.mean(a)
    B = b - np.mean(b, axis=0) - np.mean(b, axis=1)[:, None] + np.mean(b)
    dcov2_xy = np.sum(A * B) / float(n * n)
    dcov2_xx = np.sum(A * A) / float(n * n)
    dcov2_yy = np.sum(B * B) / float(n * n)
    return np.sqrt(dcov2_xy) / np.sqrt((np.sqrt(dcov2_xx) * np.sqrt(dcov2_yy)))


def compute_1dcorrelatiosn(x, y):
    p, _ = stats.pearsonr(x, y)
    s, _ = stats.spearmanr(x, y)
    t, _ = stats.kendalltau(x, y)
    pd = dis_correlation(x, y)
    return p, s, t, pd


def multiple_comparisons(P, type='F', ranks=False):
    n, k = P.shape
    if type == 'F':
        R = np.empty([n, k])
        for ii in range(n):
            R[ii, :] = stats.rankdata(P[ii, :], method='average')
            R[ii, :] = k + 1. - R[ii, :]
        Rj = np.mean(R, 0)
        den = np.sqrt((k * (k + 1.)) / (6. * n))
    elif type == 'Q':
        R = np.empty([n, k])
        Q = np.max(P, 1) - np.min(P, 1)
        Q = stats.rankdata(Q, method='average')
        Q = np.tile(Q, (k, 1)).T
        for ii in range(n):
            R[ii, :] = stats.rankdata(P[ii, :], method='average')
            R[ii, :] = k + 1. - R[ii, :]
        S = R * Q
        Rj = np.sum(S, 0)
        Rj = Rj / (n * (n + 1.) / 2.)
        den = np.sqrt((k * (k + 1.) * (2. * n + 1.) * (k - 1.)) / (18. * n * (n + 1.)))
    z = np.empty([k, k])
    for ii in range(k):
        for jj in range(k):
            z[ii, jj] = (Rj[ii] - Rj[jj]) / den
    p = 2. * stats.norm.cdf(z, 0., 1.)
    # Bonferroni - Dunn correction
    p_adj = np.minimum(1., (k - 1.) * p)
    if ranks:
        return p, p_adj, Rj
    else:
        return p, p_adj


def KullbackLeiblerDivergence(h1, h2):
    mah1 = np.max(h1)
    mah2 = np.max(h2)
    if mah1 > 1 or mah2 > 1:
        h1 = 1. * h1 / np.sum(h1)
        h2 = 1. * h2 / np.sum(h2)
    h = h1 + h2
    h = h[h != 0]
    h1 = h1[h1 != 0]
    h2 = h2[h2 != 0]
    return 2. * np.log(2.) + np.sum(h1 * np.log(h1)) + np.sum(h2 * np.log(h2)) - np.sum(h * np.log(h))


def fastbilateralfilter(img):
    if np.min(img.shape) < 1024:
        z = 2
    else:
        z = 4
    img[np.where(img < 0.0001)] = 0.0001
    logimg = np.log10(img)
    base_layer = PiecewiseBilateralFilter(logimg, z)
    base_layer = np.minimum(base_layer, np.max(logimg))
    detail_layer = logimg - base_layer
    detail_layer[np.where(detail_layer > 12.)] = 0.
    base_layer = np.power(10, base_layer)
    detail_layer = np.power(10, detail_layer)
    return base_layer, detail_layer


def PiecewiseBilateralFilter(imageIn, z):
    imSize = imageIn.shape
    xDim = imSize[1]
    yDim = imSize[0]
    sigma_s = 2. * xDim / z / 100.
    sigma_r = 0.35
    maxI = np.max(imageIn)
    minI = np.min(imageIn)
    nSeg = (maxI - minI) / sigma_r
    inSeg = np.int(round(nSeg))
    distMap = idl_dist(yDim, xDim)
    kernel = np.exp(-1. * np.power(distMap / sigma_s, 2))
    kernel = kernel / kernel[0, 0]
    fs = np.maximum(np.real(np.fft.fft(kernel)), 0)
    fs = fs / fs[0, 0]
    Ip = imageIn[0::z, 0::z]
    fsp = fs[0::z, 0::z]
    imageOut = np.zeros(imSize)
    for jj in range(0, inSeg):
        value_i = minI + jj * (maxI - minI) / inSeg
        jGp = np.exp((-1. / 2.) * np.power((Ip - value_i) / sigma_r, 2))
        jKp = np.maximum(np.real(np.fft.ifft(np.fft.fft(jGp) * fsp)), 0.0000000001)
        jHp = jGp * Ip
        sjHp = np.real(np.fft.ifft(np.fft.fft(jHp) * fsp))
        jJp = sjHp / jKp
        jJ = misc.imresize(jJp, float(z), 'nearest')
        jJ = jJ[0:yDim, 0:xDim]
        intW = np.maximum(np.ones(imSize) - np.abs(imageIn - value_i) * (inSeg) / (maxI - minI), 0)
        imageOut = imageOut + jJ * intW
    return imageOut


def idl_dist(m, n=None):
    x = np.arange(m)
    x = np.power(np.minimum(x, (m - x)), 2)
    if n is None:
        n = m
    a = np.zeros((m, n))
    for ii in range(0, n / 2):
        y = np.sqrt(x + ii ** 2)
        a[:, ii] = y
        if ii != 0:
            a[:, m - ii + 1] = y
    return a


def resize(orig, newSize, align=np.array([0, 0]), padding=0):
    sizem1n1 = orig.shape
    if isinstance(newSize, int):
        newSize = np.array([newSize, newSize])
    if isinstance(align, int):
        align = np.array([align, align])
    n1 = sizem1n1[0]
    if len(sizem1n1) < 2:
        m1 = 1
    else:
        m1 = sizem1n1[0]
        n1 = sizem1n1[1]
    orig = orig.reshape((m1, n1))
    m2 = newSize[0]
    n2 = newSize[1]
    m = np.minimum(m1, m2)
    n = np.minimum(n1, n2)
    result = np.ones((m2, n2)) * padding
    start1 = [np.floor((m1 - m) / 2. * (1. + align[0])), np.floor((n1 - n) / 2. * (1. + align[1]))]
    start2 = [np.floor((m2 - m) / 2. * (1. + align[0])), np.floor((n2 - n) / 2. * (1. + align[1]))]
    result[np.arange(int(start2[0]), int(start2[0] + m))[:,None], np.arange(int(start2[1]), int(start2[1] + n))] = \
        orig[np.arange(int(start1[0]), int(start1[0] + m))[:,None], np.arange(int(start1[1]), int(start1[1] + n))]
    return result

def separableFilters(sampPerDeg, dimension=1):
    minSAMPPERDEG = 224.
    if ((sampPerDeg < minSAMPPERDEG) and dimension != 2):
        uprate = np.ceil(minSAMPPERDEG / sampPerDeg)
        sampPerDeg = sampPerDeg * uprate
    else:
        uprate = 1.
    x1 = np.array([0.05, 1.00327, 0.225, 0.114416, 7.0, -0.117686])
    x2 = np.array([0.0685, 0.616725, 0.826, 0.383275])
    x3 = np.array([0.0920, 0.567885, 0.6451, 0.432115])
    x1[[1, 3, 5]] = x1[[1, 3, 5]] * sampPerDeg
    x2[[1, 3]] = x2[[1, 3]] * sampPerDeg
    x3[[1, 3]] = x3[[1, 3]] * sampPerDeg
    width = np.ceil(sampPerDeg / 2) * 2 - 1
    k1 = np.vstack((np.vstack((gauss(x1[0], width) * np.sqrt(np.abs(x1[1])) * np.sign(x1[1]),
                               gauss(x1[2], width) * np.sqrt(np.abs(x1[3])) * np.sign(x1[3]))),
                    gauss(x1[4], width) * np.sqrt(np.abs(x1[5])) * np.sign(x1[5])))
    k2 = np.vstack((gauss(x2[0], width) * np.sqrt(np.abs(x2[1])) * np.sign(x2[1]), \
                    gauss(x2[2], width) * np.sqrt(np.abs(x2[3])) * np.sign(x2[3])))
    k3 = np.vstack((gauss(x3[0], width) * np.sqrt(np.abs(x3[1])) * np.sign(x3[1]),
                    gauss(x3[2], width) * np.sqrt(np.abs(x3[3])) * np.sign(x3[3])))
    if ((dimension != 2) and uprate > 1):
        upcol = np.hstack((np.arange(1, uprate + 1), np.arange(uprate - 1, 0, -1))) / uprate
        s = upcol.size
        upcol = resize(upcol, np.array([1, s + width - 1]))
        up1 = signal.convolve2d(k1, upcol, mode='same')
        up2 = signal.convolve2d(k2, upcol, mode='same')
        up3 = signal.convolve2d(k3, upcol, mode='same')
        s = up1.shape[1]
        mid = np.ceil(s / 2.)
        downs = np.hstack((np.arange(1, mid, uprate), np.arange(mid + uprate, up1.shape[1], uprate)))
        k1 = up1[:, np.int_(downs)]
        k2 = up2[:, np.int_(downs)]
        k3 = up3[:, np.int_(downs)]
    return k1, k2, k3


def pad4conv(im, kernelsize, dim=None):
    if not isinstance(kernelsize, list):
        kernelsize = [kernelsize, kernelsize]
    if dim is None:
        dim = 3
    imsize = im.shape
    m = imsize[0]
    n = imsize[1]
    if (kernelsize[0] >= m):
        h = np.floor(m / 2.)
    else:
        h = np.floor(kernelsize[0] / 2)
    if (kernelsize[1] >= n):
        w = np.floor(n / 2.)
    else:
        w = np.floor(kernelsize[1] / 2)
    if h != 0 and dim != 2:
        im = np.vstack((im, np.flipud(im[range(int(m) - int(h), int(m)), :])))
        im = np.vstack((np.flipud(im[range(0, int(h)), :]), im))
    if w != 0 and dim != 1:
        im = np.hstack((im, np.fliplr(im[:, range(int(n) - int(w), int(n))])))
        im = np.hstack((np.fliplr(im[:, range(0, int(w))]), im))
    return im


def separableConv(im, xkernels, ykernels=None):
    if ykernels is None:
        ykernels = xkernels
    imsize = im.shape
    w1 = pad4conv(im, xkernels.shape[1], 2)
    result = np.zeros(imsize)
    for jj in range(xkernels.shape[0]):
        p = signal.convolve2d(w1, xkernels[jj, :].reshape((1, xkernels.shape[1])), mode='full')
        p = resize(p, imsize)
        w2 = pad4conv(p, ykernels.shape[1], 1)
        p = signal.convolve2d(w2, ykernels[jj, :].reshape((xkernels.shape[1], 1)), mode='full')
        p = resize(p, imsize)
        result = result + p
    return result


def gauss(halfWidth, width):
    alpha = 2. * np.sqrt(np.log(2.)) / (halfWidth - 1.)
    x = np.arange(1, width + 1) - round(width / 2.)
    g = np.exp(-alpha * alpha * x * x)
    return g / np.sum(g)


def f(Y):
    fY = np.real(np.power(Y,1./3.))
    ii = (Y < 0.008856)
    fY[ii] = Y[ii]*(841./108.) + (4./29.)
    return fY


def matlab_style_gauss2D(shape=(3,3),sigma=0.5):
    m,n = [(ss-1.)/2. for ss in shape]
    y,x = np.ogrid[-m:m+1,-n:n+1]
    h = np.exp( -(x*x + y*y) / (2.*sigma*sigma) )
    h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h


def entropy3d(H, edges):
    Hp1 = 0.
    Hp2 = 0.
    for ii in range(H.shape[0]):
        for jj in range(H.shape[1]):
            for kk in range(H.shape[2]):
                if H[ii, jj, kk] > 0:
                    Hp1 += H[ii, jj, kk] * np.log2(H[ii, jj, kk])
    return -1. * Hp1


def dist_edges(L_edges,a_edges,b_edges,pos):
    delta_Lab = np.sqrt(np.power(L_edges[pos[0]]-L_edges[pos[0]+1],2)+\
                        np.power(a_edges[pos[1]]-a_edges[pos[1]+1],2)+\
                        np.power(b_edges[pos[2]]-b_edges[pos[2]+1],2))
    return delta_Lab


def ciede2000(Lstd, astd, bstd, Lsample, asample, bsample, KLCH=np.array([1, 1, 1])):
    kl = KLCH[0]
    kc = KLCH[1]
    kh = KLCH[2]
    Cabstd = np.sqrt(np.power(astd, 2) + np.power(bstd, 2))
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
    return np.mean(DE00)


def deltaosa(Lstd, astd, bstd, Lsample, asample, bsample):
    C_ref = np.sqrt(np.power(astd, 2) + np.power(bstd, 2))
    h_ref = np.arctan2(astd, -bstd)
    C_pro = np.sqrt(np.power(asample, 2) + np.power(bsample, 2))
    h_pro = np.arctan2(asample, -bsample)
    S_L = 2.499 + 0.07 * (Lstd + Lsample) / 2.
    S_C = 1.235 + 0.58 * (C_ref + C_pro) / 2.
    S_H = 1.392 + 0.17 * (h_ref + h_pro) / 2.
    dL = (Lstd - Lsample) / S_L
    dC = (C_ref - C_pro) / S_C
    dh = (h_ref - h_pro) / S_H
    DE = 10. * np.sqrt(np.power(dL, 2) + np.power(dC, 2) + np.power(dh, 2))
    de = np.mean(DE)
    if np.isnan(de):
        de = np.nanmean(DE)
    return de


def dct_block_mse(A_ref, A_pro, blk_size=8):
    MaskCof = np.array([[0.390625, 0.826446, 1.000000, 0.390625, 0.173611, 0.062500, 0.038447, 0.026874],\
                        [0.694444, 0.694444, 0.510204, 0.277008, 0.147929, 0.029727, 0.027778, 0.033058],\
                        [0.510204, 0.591716, 0.390625, 0.173611, 0.062500, 0.030779, 0.021004, 0.031888],\
                        [0.510204, 0.346021, 0.206612, 0.118906, 0.038447, 0.013212, 0.015625, 0.026015],\
                        [0.308642, 0.206612, 0.073046, 0.031888, 0.021626, 0.008417, 0.009426, 0.016866],\
                        [0.173611, 0.081633, 0.033058, 0.024414, 0.015242, 0.009246, 0.007831, 0.011815],\
                        [0.041649, 0.024414, 0.016437, 0.013212, 0.009426, 0.006830, 0.006944, 0.009803],\
                        [0.019290, 0.011815, 0.011080, 0.010412, 0.007972, 0.010000, 0.009426, 0.010203]])
    CSFCof = np.array([[1.608443, 2.339554, 2.573509, 1.608443, 1.072295, 0.643377, 0.504610, 0.421887],\
                       [2.144591, 2.144591, 1.838221, 1.354478, 0.989811, 0.443708, 0.428918, 0.467911],\
                       [1.838221, 1.979622, 1.608443, 1.072295, 0.643377, 0.451493, 0.372972, 0.459555],\
                       [1.838221, 1.513829, 1.169777, 0.887417, 0.504610, 0.295806, 0.321689, 0.415082],\
                       [1.429727, 1.169777, 0.695543, 0.459555, 0.378457, 0.236102, 0.249855, 0.334222],\
                       [1.072295, 0.735288, 0.467911, 0.402111, 0.317717, 0.247453, 0.227744, 0.279729],\
                       [0.525206, 0.402111, 0.329937, 0.295806, 0.249855, 0.212687, 0.214459, 0.254803],\
                       [0.357432, 0.279729, 0.270896, 0.262603, 0.229778, 0.257351, 0.249855, 0.259950]])
    M, N = A_ref.shape
    S1blk = np.zeros((M, N))
    S2blk = np.zeros((M, N))
    for ii in xrange(0, M, blk_size):
        for jj in xrange(0, N, blk_size):
            if ii + blk_size <= M and jj + blk_size <= N:
                values_ref = A_ref[ii:ii + blk_size, jj:jj + blk_size]
                values_ref = fftpack.dct(values_ref)
                BMasket_ref = np.power(values_ref, 2) * MaskCof
                m_ref = np.sum(BMasket_ref)-BMasket_ref[0, 0]
                pop_ref = vari(values_ref)
                if pop_ref is not 0:
                    pop_ref = (vari(values_ref[0:3, 0:3])+vari(values_ref[0:3, 4:7])+vari(values_ref[4:7, 4:7])\
                               +vari(values_ref[4:7, 0:3]))/pop_ref
                m_ref = np.sqrt(m_ref * pop_ref) / 32
                values_pro = A_pro[ii:ii + blk_size, jj:jj + blk_size]
                values_pro = fftpack.dct(values_pro)
                BMasket_pro = np.power(values_pro, 2) * MaskCof
                m_pro = np.sum(BMasket_pro) - BMasket_pro[0, 0]
                pop_pro = vari(values_pro)
                if pop_pro is not 0:
                    pop_pro = (vari(values_pro[0:3, 0:3])+vari(values_pro[0:3, 4:7])+vari(values_pro[4:7, 4:7])+\
                               vari(values_pro[4:7, 0:3]))/pop_pro
                m_pro = np.sqrt(m_pro * pop_pro) / 32
                if m_pro > m_ref:
                    m_ref = m_pro
                Dif_dct = np.abs(values_ref - values_pro)
                S1 = np.power(Dif_dct * CSFCof, 2)
                for kk in xrange(0, blk_size):
                    for ll in xrange(0, blk_size):
                        if kk != 0 or ll != 0:
                            if Dif_dct[kk, ll] < m_ref / MaskCof[kk, ll]:
                                Dif_dct[kk, ll] = 0
                            else:
                                Dif_dct[kk, ll] -= m_ref / MaskCof[kk, ll]
                S2 = np.power(Dif_dct * CSFCof, 2)
                S1blk[ii:ii + blk_size, jj:jj + blk_size] = np.mean(S1)
                S2blk[ii:ii + blk_size, jj:jj + blk_size] = np.mean(S2)
    s1 = np.nanmean(S1blk)
    s2 = np.nanmean(S2blk)
    return s1, s2, S1blk, S2blk


def clip_psnr(s1):
    if s1 is 0:
        return 100000.
    else:
        return 10 * np.log10(255. * 255. / s1)

def clip_psnr_mat(s1):
    idx1 = np.where(s1 != 0)
    idx2 = np.where(s1 == 0)
    s1[idx1] = 10 * np.log10((255. * 255.) / s1[idx1])
    s1[idx2] = 100000.
    return s1


def vari(x):
    return np.var(x)*x.size


def SDSP(image, sigmaF=1.34, omega0=0.0210, sigmaD=145, sigmaC=0.001):
    oriRows, oriCols, _ = image.shape
    dsImage = misc.imresize(image, (256, 256),  'bilinear')
    lab = color.rgb2lab(dsImage)
    lab = np.double(lab)
    LChannel = lab[:, :, 0]
    AChannel = lab[:, :, 1]
    BChannel = lab[:, :, 2]

    LFFT = np.fft.fft2(LChannel)
    AFFT = np.fft.fft2(AChannel)
    BFFT = np.fft.fft2(BChannel)

    rows, cols, _ = dsImage.shape
    LG = logGabor(rows, cols, omega0, sigmaF)
    FinalLResult = np.real(np.fft.ifft2(LFFT * LG))
    FinalAResult = np.real(np.fft.ifft2(AFFT * LG))
    FinalBResult = np.real(np.fft.ifft2(BFFT * LG))

    SFMap = np.sqrt(np.power(FinalLResult, 2) + np.power(FinalAResult, 2) + np.power(FinalBResult, 2))

    coordinateMtx = np.zeros((rows, cols, 2))
    coordinateMtx[:,:,0] = np.matlib.repmat(np.arange(rows), cols, 1).T
    coordinateMtx[:,:,1] = np.matlib.repmat(np.arange(cols), rows, 1)

    centerY = rows / 2
    centerX = cols / 2
    centerMtx = np.zeros((rows, cols, 2))
    centerMtx[:,:, 0] = np.ones((rows, cols)) * centerY
    centerMtx[:,:, 1] = np.ones((rows, cols)) * centerX
    SDMap = np.exp(-np.sum(np.power(coordinateMtx - centerMtx,2), 2) / (sigmaD ** 2))


    maxA = np.max(AChannel)
    minA = np.min(AChannel)
    normalizedA = (AChannel - minA) / (maxA - minA)

    maxB = np.max(BChannel)
    minB = np.min(BChannel)
    normalizedB = (BChannel - minB) / (maxB - minB);

    labDistSquare = np.power(normalizedA, 2) + np.power(normalizedB, 2)
    SCMap = 1 - np.exp(-labDistSquare / (sigmaC ** 2));
    VSMap = SFMap * SDMap * SCMap

    VSMap = np.double(misc.imresize(VSMap, (oriRows, oriCols),  'bilinear'))

    return VSMap/np.max(VSMap)


def logGabor(rows, cols, omega0=0.0210, sigmaF=1.34):
    u1, u2 = np.meshgrid((np.arange(cols)-(np.fix(cols / 2)))/(cols - np.mod(cols, 2)),\
                         (np.arange(rows)-(np.fix(rows / 2)))/(rows - np.mod(rows, 2)))
    mask = np.ones((rows, cols))
    for rowIndex in range(rows):
        for colIndex in range(cols):
            if (u1[rowIndex, colIndex]**2 + u2[rowIndex, colIndex]**2) > 0.25:
                mask[rowIndex, colIndex] = 0
    u1 = u1 * mask
    u2 = u2 * mask
    u1 = np.fft.ifftshift(u1)
    u2 = np.fft.ifftshift(u2)
    radius = np.sqrt(np.power(u1, 2) + np.power(u2, 2))
    radius[0, 0] = 1
    idx = np.where(radius==0)
    radius[idx] = 1
    LG = np.exp((-np.power(np.log(radius / omega0),2))/(2*(sigmaF ** 2)))
    LG[idx] = 0
    LG[0, 0] = 0
    return LG


if __name__ == "__main__":
    print format_time(150000)
    # P = np.array([[0.57, 0.81, 0.85, 0.79], [0.69, 0.91, 0.90, 0.86],\
    #      [0.83, 0.89, 0.94, 0.92], [0.60, 0.86, 0.93, 0.91],\
    #      [0.67, 0.89, 0.89, 0.95], [0.56, 0.81, 0.73, 0.81]])
    # p, p_adj, Rj = multiple_comparisons(P, type='F', ranks=True)
    # width = np.round(1. / len(P[0, :]), 3)
    # plt.boxplot(np.abs(P), positions=np.arange(len(P[0, :])) + 0 * width, boxprops= \
    #     dict(color='r', linewidth=3, markersize=12), widths=width)
    # plt.tick_params(labelsize=18)
    # axes = plt.gca()
    # axes.set_ylim([0, 1])
    # axes.set_xticks(np.arange(len(P[0, :])) + 1.5 * width)
    # plt.rcParams.update({'font.size': 16})
    # plt.show()
    # print p
    # print p_adj
    # print Rj
    # MUltimedia_file_ref = './sample_images/test_ref_0.bmp'
    # image_ref = ndimage.imread(MUltimedia_file_ref)
    # Yar, meanT = contrast_peli(image_ref)
    # print meanT
    # plt.imshow(Yar)
    # plt.colorbar()
    # plt.show()


# def local_contrast_bcontent_measure(I, s = np.array([9,9]), formula='Weber'):
#     Yref = 0.299 * I[:, :, 0] + 0.587 * I[:, :, 1] + 0.114 * I[:, :, 2]
#     precision = 1e-6
#     C = np.zeros_like(Yref)
#     M, N = Yref.shape
#     for ii in xrange(np.int(np.floor(s[0]/2.)), M-np.int(np.floor(s[0]/2.))):
#         for jj in xrange(np.int(np.floor(s[1]/2.)), N-np.int(np.floor(s[1]/2.))):
# 			temp = I[ii-np.int(np.floor(s[0]/2.)):ii+np.int(np.floor(s[0]/2.)),\
# 				   jj-np.int(np.floor(s[1]/2.)):jj+np.int(np.floor(s[1]/2.))]
# 			mud = Isodata(temp,precision)
# 			mu1 = mud[0]
# 			mu2 = mud[1]
# 			Imin = np.max(np.array([0., mu1]))
# 			Imax = np.max(np.array([1., mu2]))
# 			if formula == 'Simple':
# 				c = Imin / Imax
# 			elif formula == 'Weber':
# 				c = 1. - Imin / Imax
# 			elif formula == 'Michelson':
# 				c = (Imax-Imin)/(Imax+Imin)
# 			C[ii, jj] = c
#     return C, np.mean(C)



