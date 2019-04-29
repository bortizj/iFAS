import numpy as np
import pywt
from scipy import ndimage
from scipy import misc
from scipy import signal
from skimage import measure
import myUtilities


def psnr(Ref_image, Pro_image):
    Yref = myUtilities.checkifRGB(Ref_image)
    Ypro = myUtilities.checkifRGB(Ref_image)
    mse = np.mean(np.power(Yref - Ypro, 2))
    if mse != 0:
        psnr = 10 * np.log10((255. ** 2) / mse)
    else:
        psnr = 1e6
    return np.abs(Yref - Ypro), psnr


def ssim(Ref_image, Pro_image):
    Yref = myUtilities.checkifRGB(Ref_image)
    Ypro = myUtilities.checkifRGB(Ref_image)
    val, SSIM = measure.compare_ssim(Yref, Ypro, full=True)
    return SSIM, val


def SSIM(img1, img2, K=np.array([0.01, 0.03]), wsize=11, mode='valid'):
    img1 = myUtilities.checkifRGB(img1)
    img2 = myUtilities.checkifRGB(img2)
    xx, yy = np.meshgrid(np.arange(-wsize / 2, wsize / 2), np.arange(-wsize / 2, wsize / 2))
    window = myUtilities.gaussian(xx, yy, 1.5)
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
    return ssim_map, np.mean(ssim_map)


def blocking_diff(Ref_image, Pro_image):
    Yref = myUtilities.checkifRGB(Ref_image)
    Ypro = myUtilities.checkifRGB(Ref_image)
    D, _ = psnr(Ref_image, Pro_image)
    return D, myUtilities.blocking(Yref) - myUtilities.blocking(Ypro)


def noise_diff(Ref_image, Pro_image):
    Yref = myUtilities.checkifRGB(Ref_image)
    Ypro = myUtilities.checkifRGB(Ref_image)
    _, (_, _, cDR) = pywt.dwt2(Yref, 'db1', 'sym')
    _, (_, _, cDP) = pywt.dwt2(Ypro, 'db1', 'sym')
    auxref, noise_ref = myUtilities.EstimateNoise_SBLE(cDR)
    auxpro, noise_pro = myUtilities.EstimateNoise_SBLE(cDP)
    Yref = misc.imresize(auxref, Yref.shape, interp='bilinear')
    Ypro = misc.imresize(auxpro, Ypro.shape, interp='bilinear')
    return Yref-Ypro, noise_ref-noise_pro


def blur_diff(Ref_image, Pro_image):
    Yref = myUtilities.checkifRGB(Ref_image)
    Ypro = myUtilities.checkifRGB(Ref_image)
    dims = Yref.shape
    mask_perc = 10 # 5% or 10%
    mask_threshold = np.int(np.floor(dims[0]*dims[1]*mask_perc/100))
    Dx23 = np.asarray(np.empty(Yref.shape),dtype=np.float32)
    Dy23 = np.asarray(np.empty(Yref.shape),dtype=np.float32)
    _, Dx2_u, Dx3_u, _, Dy2_u, Dy3_u = myUtilities.calculate_wavelet_images(Yref)
    np.multiply(Dx2_u,Dx3_u,out=Dx23)
    np.multiply(Dy2_u,Dy3_u,out=Dy23)
    np.fabs(Dx23,out=Dx23)
    Dx23_sort = np.sort(Dx23, axis=None)
    Dx23_sort = Dx23_sort[::-1]
    np.fabs(Dy23,out=Dy23)
    Dy23_sort = np.sort(Dy23, axis=None)
    Dy23_sort = Dy23_sort[::-1]
    Tx2 = Dx23_sort[mask_threshold-1]
    Ty2 = Dy23_sort[mask_threshold-1]
    Mx2 = Dx23 > Tx2
    My2 = Dy23 > Ty2
    x2, n23s = myUtilities.lj_FindACRhists(Yref, Mx2, My2)
    BlurRef = myUtilities.get_CogACR(x2, n23s)
    x2, n23s = myUtilities.lj_FindACRhists(Ypro, Mx2, My2)
    BlurDis = myUtilities.get_CogACR(x2, n23s)
    return Yref-Ypro, BlurRef-BlurDis


def epsnr(Ref_image, Pro_image):
    Yref = myUtilities.checkifRGB(Ref_image)
    Ypro = myUtilities.checkifRGB(Ref_image)
    aux = np.asarray(np.empty(Yref.shape),dtype=np.float32)
    mask = np.asarray(np.empty(Yref.shape),dtype=np.bool)
    ndimage.filters.generic_gradient_magnitude(Yref, derivative=ndimage.filters.sobel, output=aux)
    T = myUtilities.Isodata(aux)
    np.greater_equal(aux, T, out=mask)
    np.subtract(Yref,Ypro,out=aux)
    np.square(aux,out=aux)
    mse = np.mean(aux[mask])
    if mse != 0:
        psnr = (20*np.log10(255)-10*np.log10(mse))
    else:
        psnr = 1.e6
    return aux*mask, psnr
