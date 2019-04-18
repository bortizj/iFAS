import numpy as np
import pywt
from scipy import ndimage
from scipy import misc
from scipy import signal
from skimage import color
from skimage import measure
import my_utilities as MU
import color_spaces as CAM

def psnr(Ref_image, Pro_image):
    Yref = 0.299 * Ref_image[:, :, 0] + 0.587 * Ref_image[:, :, 1] + 0.114 * Ref_image[:, :, 2]
    Ypro = 0.299 * Pro_image[:, :, 0] + 0.587 * Pro_image[:, :, 1] + 0.114 * Pro_image[:, :, 2]

    return np.abs(Yref-Ypro), 10*np.log10((255.**2)/np.mean(np.power(Yref-Ypro,2)))

def ssim(Ref_image, Pro_image):
    Yref = 0.299 * Ref_image[:, :, 0] + 0.587 * Ref_image[:, :, 1] + 0.114 * Ref_image[:, :, 2]
    Ypro = 0.299 * Pro_image[:, :, 0] + 0.587 * Pro_image[:, :, 1] + 0.114 * Pro_image[:, :, 2]
    val, SSIM = measure.compare_ssim(Yref, Ypro,full=True)
    return SSIM, val

def blocking_diff(Ref_image, Pro_image):
    Yref = 0.299 * Ref_image[:, :, 0] + 0.587 * Ref_image[:, :, 1] + 0.114 * Ref_image[:, :, 2]
    Ypro = 0.299 * Pro_image[:, :, 0] + 0.587 * Pro_image[:, :, 1] + 0.114 * Pro_image[:, :, 2]
    D, _ = psnr(Ref_image, Pro_image)
    return D, MU.blocking(Yref) - MU.blocking(Ypro)


def noise_diff(Ref_image, Pro_image):
    Yref = 0.299 * Ref_image[:, :, 0] + 0.587 * Ref_image[:, :, 1] + 0.114 * Ref_image[:, :, 2]
    Ypro = 0.299 * Pro_image[:, :, 0] + 0.587 * Pro_image[:, :, 1] + 0.114 * Pro_image[:, :, 2]
    _, (_, _, cDR) = pywt.dwt2(Yref, 'db1', 'sym')
    _, (_, _, cDP) = pywt.dwt2(Ypro, 'db1', 'sym')
    auxref, noise_ref = MU.EstimateNoise_SBLE(cDR)
    auxpro, noise_pro = MU.EstimateNoise_SBLE(cDP)
    Yref = misc.imresize(auxref, Yref.shape, interp='bilinear')
    Ypro = misc.imresize(auxpro, Ypro.shape, interp='bilinear')
    return Yref-Ypro, noise_ref-noise_pro


def blur_diff(Ref_image, Pro_image):
    Yref = 0.299 * Ref_image[:, :, 0] + 0.587 * Ref_image[:, :, 1] + 0.114 * Ref_image[:, :, 2]
    Ypro = 0.299 * Pro_image[:, :, 0] + 0.587 * Pro_image[:, :, 1] + 0.114 * Pro_image[:, :, 2]
    dims = Yref.shape
    mask_perc = 10 # 5% or 10%
    mask_threshold = np.int(np.floor(dims[0]*dims[1]*mask_perc/100))
    Dx23 = np.asarray(np.empty(Yref.shape),dtype=np.float32)
    Dy23 = np.asarray(np.empty(Yref.shape),dtype=np.float32)
    _, Dx2_u, Dx3_u, _, Dy2_u, Dy3_u = MU.calculate_wavelet_images(Yref)
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
    x2, n23s = MU.lj_FindACRhists(Yref, Mx2, My2)
    BlurRef = MU.get_CogACR(x2, n23s)
    x2, n23s = MU.lj_FindACRhists(Ypro, Mx2, My2)
    BlurDis = MU.get_CogACR(x2, n23s)
    return Yref-Ypro, BlurRef-BlurDis


def epsnr(Ref_image, Pro_image):
    Yref = 0.299 * Ref_image[:, :, 0] + 0.587 * Ref_image[:, :, 1] + 0.114 * Ref_image[:, :, 2]
    Ypro = 0.299 * Pro_image[:, :, 0] + 0.587 * Pro_image[:, :, 1] + 0.114 * Pro_image[:, :, 2]
    aux = np.asarray(np.empty(Yref.shape),dtype=np.float32)
    mask = np.asarray(np.empty(Yref.shape),dtype=np.bool)
    ndimage.filters.generic_gradient_magnitude(Yref, derivative=ndimage.filters.sobel, output=aux)
    T = MU.Isodata(aux)
    np.greater_equal(aux, T, out=mask)
    np.subtract(Yref,Ypro,out=aux)
    np.square(aux,out=aux)
    mse = np.mean(aux[mask])
    return aux*mask, (20*np.log10(255)-10*np.log10(mse))


if __name__ == "__main__":
    MUltimedia_file_ref = './test_ref_0.bmp'
    image_ref = ndimage.imread(MUltimedia_file_ref)
    MUltimedia_file_pro = './test_pro_0.bmp'
    image_pro = ndimage.imread(MUltimedia_file_pro)
    DE00, de00 = ssim(image_ref, image_pro)
    print de00