"""
Copyleft 2021
This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation version 3.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.

author: Benhur Ortiz-Jaramillo
"""

from scipy import io
import numpy as np
import cv2
from pathlib import Path

# Even though the functions are implemented to support arbitrary sizes, they work better with squared size images
# Only process images with one channel (gray scale images)

FILE_PATH = Path(__file__).parent.absolute()

# -------------------- List of global variables only related to this texture analysis algorithms -------------------- #

# If image size is smaller than 64 x 64, autocorrelation function uses the whole autocorrelation instead of only the
# center maximum
MIN_IMG_SIZE = 64
# Dictionary of number of neighbors and scale for given radius of analysis
RADII = {1: [8, 1], 1.5: [12, 1. / 2.], 2: [16, 1. / 4.], 3: [24, 1. / 8.]}

# Set of filters to determine the neighbors in gaussian_markov_random_field
GMRF_FILTERS = [
    np.array([[0, 0, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 0, 0], [0, 0, 1, 0, 0], [0, 0, 0, 0, 0]]),
    np.array([[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 1, 0, 1, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]),
    np.array([[0, 0, 1, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 1, 0, 0]]),
    np.array([[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [1, 0, 0, 0, 1], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]),
    np.array([[0, 0, 0, 0, 0], [0, 0, 0, 1, 0], [0, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 0, 0, 0]]),
    np.array([[0, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 1, 0], [0, 0, 0, 0, 0]])
    ]

# 1 dimensional kernels proposed by laws [Level, Edge, Spot, Wave, and Ripple]
LAWS_FILTERS = [
    np.array([1, 4, 6, 4, 1]),
    np.array([-1, -2, 0, 2, 1]),
    np.array([-1, 0, 2, 0, -1]),
    np.array([-1, 2, 0, -2, 1]),
    np.array([1, -4, 6, -4, 1])
    ]

# Set of filters to determine the neighbors in gray_level_cooccurrence_matrix
GLCM_FILTERS = [
    np.array([[0, 0, 0], [0, 0, 1], [0, 0, 0]]),
    np.array([[0, 0, 0], [0, 0, 0], [0, 0, 1]]), 
    np.array([[0, 0, 0], [0, 0, 0], [0, 1, 0]]),
    np.array([[0, 0, 0], [0, 0, 0], [1, 0, 0]])
    ]

# Image size where Fourier mask where created (mask is later interpolated to the size of the image)
FOURIER_DOMAIN_SIZE = 128


def skewness_kurtosis(img, avg=None):
    """
    Computes skewness and kurtosis of a given image img
    """
    if avg is None:
        avg = np.mean(img)

    n = img.size
    num = (1. / n) * np.sum(np.power(img - avg, 3))
    den = np.power((1. / (n - 1)) * np.sum(np.power(img - avg, 2)), 3. / 2.)
    skew = num / den
    num = (1. / n) * np.sum(np.power(img - avg, 4))
    den = np.power((1. / n) * np.sum(np.power(img - avg, 2)), 2)
    kurt = (num / den) - 3

    return skew, kurt


def number_neighbour(r=1.5, look_at_table=False):
    """
    returns the number of neighbors n, the scale and the look at table for the given radius r
    """
    n = RADII[r][0]
    scale = RADII[r][1]

    if look_at_table:
        # Look at tables from previous investigation see Maenpaa work in the lbp function
        var = io.loadmat(str(FILE_PATH.joinpath('look_at_tables_lbp.mat')))
        table = var['look_table_' + str(n)]
        return n, scale, table
    else:
        return n, scale


# -------------------------------- Model based approaches -------------------------------- #


def autoregressive_model(img, r=1.5):
    """
    Estimates the parameters of a circular autoregressive model in a radius r around the central pixel
    based on
    Kashyap and Chellappa, "Estimation and choice of neighbors in spatial-interaction models of images.
    IEEE Transactions on Information Theory, 29:60 – 72, (1983)

    img: input image, r: the radius of analysis
    returns img_ar: reconstructed image using Autoregressive models, a: the parameters of the estimated model
    """
    if len(img.shape) > 2:
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype("float32")
    else:
        img_gray = img.astype('float32')

    n, __ = number_neighbour(r)
    avg = np.mean(img_gray)
    std = np.std(img_gray)

    # Standardizing image, it is required to compute AR models
    img_gray = (img_gray - avg) / std

    # Pixels located outside the grid are estimated using bilinear interpolation
    xx, yy = np.meshgrid(np.arange(0, img.shape[1]), np.arange(0, img.shape[0]))
    flag = False
    for ii in range(n):
        xi = xx - r * np.sin(2 * np.pi * ii / n)
        yi = yy + r * np.cos(2 * np.pi * ii / n)
        img_ii = cv2.remap(img_gray.astype('float32'), xi.astype('float32'), yi.astype('float32'), cv2.INTER_LINEAR)

        # Allocating pixel values in columns, each row is a pixel and each column is the bilinear interpolated pixel
        if flag:
            X = np.hstack((X, np.reshape(img_ii, (img_gray.size, 1))))
        else:
            X = np.reshape(img_ii, (img_gray.size, 1))
            flag = True

    # Estimating the parameters of the AR model
    y = np.reshape(img_gray, (img_gray.size, 1))
    a, _, _, _ = np.linalg.lstsq(X, y, rcond=None)

    # Reconstructing image using the estimated parameters
    img_ar = np.reshape(np.dot(X, a), (img_gray.shape[0], img_gray.shape[1]))
    sigma = (1. / ((img_gray.shape[0] - 2) * (img_gray.shape[1] - 2))) * np.sum(np.power(img_gray - img_ar, 2))
    a = np.vstack((a, sigma))

    img_ar = img_ar * std + avg
    a = a * std + avg

    return a.T[0], img_ar


def gaussian_markov_random_field(img):
    """
    Estimates the parameters of a Gaussian Markov random field model
    based on
    Grath, M. (2003). Markov random field image modelling. Master’s thesis, University of Cape Town.

    img: input image
    returns img_ar: reconstructed image using GMRF, a: the parameters of the estimated model
    """
    if len(img.shape) > 2:
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype("float32")
    else:
        img_gray = img.astype('float32')

    avg = np.mean(img_gray)
    std = np.std(img_gray)

    # Standardizing image, it is required to compute GMRF models
    img_gray = (img_gray - avg) / std

    Sxy = []
    # Computing neighbour relationships by means of image filtering
    for ii in range(len(GMRF_FILTERS)):
        Sxy.append(cv2.filter2D(img_gray, -1, GMRF_FILTERS[ii], borderType=cv2.BORDER_REPLICATE))

    # Estimating the parameters of the model
    S = np.zeros((6, 6))
    G = np.zeros((6, 1))
    for ii in range(6):
        for jj in range(6):
            S[ii, jj] = np.sum(Sxy[ii] * Sxy[jj])
        G[ii] = np.sum(img_gray * Sxy[ii])
    a, _, _, _ = np.linalg.lstsq(S, G, rcond=None)

    # Reconstructing the image using the estimated parameters
    img_gmrf = np.zeros_like(img_gray)
    for ii in range(len(GMRF_FILTERS)):
        img_gmrf += a[ii] * Sxy[ii]

    # Computing the standard deviation of the model
    sigma = (1. / ((img_gray.shape[0] - 2) * (img_gray.shape[1] - 2))) * np.sum(np.power(img_gray - img_gmrf, 2))
    a = np.vstack((a, sigma))
    a = a * std + avg

    img_gmrf = img_gmrf * std + avg

    return a.T[0], img_gmrf


# -------------------------------- Statistical based approaches -------------------------------- #


def gray_level_cooccurrence_matrix(img, nbins=16):
    """
    Estimates the features proposed by Haralick (grey level cooccurrence matrix). The features are computed
    from the GLCM in four directions 0, 45, 90, 135 degrees. Based on
    Tuceryan and Jain, "The Handbook of Pattern Recognition and Computer Vision", chapter Texture Analysis,
    pages 1 – 41. World Scientific Publishing Co (1998)

    img: input image, nbins: number of bins to compute the GLCM. 16, 32 and 64 are typical values. Default: 16
    returns total_glcm: GLCM, feats: the features extracted from total_glcm
    """
    if len(img.shape) > 2:
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype("float32")
    else:
        img_gray = img.astype('float32')
        if np.max(img_gray) <= 1:
            print("Error: Input image has to be gray scale image in the range 0 - 255")
            return None, None

    # Quantizing image to n_bins, improves GLCM estimation
    img_gray = np.uint8((nbins / 256.) * img_gray)

    # The neighbor pixels are obtained using image filtering
    # In c++ code would be possible just to go through the image
    neighbor_img = []
    glcm = []
    for kk in range(len(GLCM_FILTERS)):
        neighbor_img.append(cv2.filter2D(img_gray, -1, GLCM_FILTERS[kk], borderType=cv2.BORDER_REPLICATE))
        glcm.append(np.zeros((nbins, nbins)))

    for ii in range(1, nbins):
        pixels_equal_to_ii = (img_gray == ii)
        for jj in range(1, nbins):
            for kk in range(len(GLCM_FILTERS)):
                pixels_equal_to_jj = (neighbor_img[kk] == jj)
                glcm[kk][ii, jj] = np.sum(pixels_equal_to_ii * pixels_equal_to_jj)

    total_glcm = np.zeros((nbins, nbins))
    for kk in range(len(glcm)):
        total_glcm += glcm[kk]

    total_glcm /= len(glcm)
    total_glcm = total_glcm / np.sum(total_glcm)

    # Adding a very small constant to avoid numerical instability (division by 0)
    total_glcm[np.where(total_glcm == 0)] = 1.0e-12

    # Computing the statistics defined by Haralick
    xgrid, ygrid = np.meshgrid(np.arange(0, nbins), np.arange(0, nbins))

    energy = np.sqrt(np.sum(total_glcm * total_glcm))
    contrast = np.sum((xgrid - ygrid) * (xgrid - ygrid) * total_glcm)
    homogeneity = np.sum(total_glcm / (1 + np.abs(xgrid - ygrid)))
    variance = np.sum(np.power(xgrid - np.mean(total_glcm), 2) * total_glcm)
    entropy = - np.sum(total_glcm * np.log2(total_glcm))

    # Computing higher order statistics
    total_glcm_x = np.sum(total_glcm, 0)
    total_glcm_y = np.sum(total_glcm, 1)

    Ex = - np.sum(total_glcm_x * np.log2(total_glcm_x))
    Ey = - np.sum(total_glcm_y * np.log2(total_glcm_y))

    total_glcm_x = np.tile(total_glcm_x, (nbins, 1))
    total_glcm_y = np.tile(total_glcm_y, (nbins, 1))

    Hxy1 = - np.sum(total_glcm * np.log2(total_glcm_x * total_glcm_y))
    Hxy2 = -np.sum((total_glcm_x * total_glcm_y) * np.log2(total_glcm_x * total_glcm_y))

    info1 = (entropy - Hxy1) / np.maximum(Ex, Ey)
    info2 = np.sqrt(1 - np.exp(-2 * (Hxy2 - entropy)))

    feats = [energy, contrast, variance, homogeneity, entropy, info1, info2]

    return np.array(feats), total_glcm


def autocorrelation(img):
    """
    Estimates the autocorrelation function and approximate it to a second degree polynomial. The parameters of
    the polynomial are used as features. Based on
    Petrou and Sevilla, "Image Processing: Dealing With Textures", chapter Stationary grey texture images,
    pages 81 – 295. John Wiley and Sons, Ltd. (2006)

    img: input image
    returns rho: autocorrelation function, p: the parameters of the estimated polynomial
    """
    if len(img.shape) > 2:
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype("float32")
    else:
        img_gray = img.astype('float32')

    # Estimating autocorrelation function using Fourier transform. Faster than computing in the spatial domain
    rho = np.fft.ifft2(np.fft.fft2(img_gray) * np.conj(np.fft.fft2(img_gray)))
    rho = np.fft.ifftshift(rho)
    rho = np.real(rho / np.sum(img_gray * img_gray))

    # Locating center peak and its surrounding
    xc, yc = np.unravel_index(np.argmax(rho), rho.shape)

    # If image bigger than minimum size, takes the central part of the autocorrelation function
    if img_gray.shape[0] > MIN_IMG_SIZE and img_gray.shape[1] > MIN_IMG_SIZE:
        max_row = rho[xc-1, 0:yc-1]
        yc_lim = np.sum(max_row > 0.75 * np.max(rho))
        max_col = rho[0:xc-1, yc-1]
        xc_lim = np.sum(max_col > 0.75 * np.max(rho))
        rho_peak = rho[xc - xc_lim:xc + xc_lim - 1, yc - yc_lim:yc + yc_lim - 1]
        xx, yy = np.meshgrid(np.arange(-xc_lim, xc_lim - 1), np.arange(-yc_lim, yc_lim - 1))
    else:
        rho_peak = rho
        xx, yy = np.meshgrid(np.arange(-xc, xc - 1), np.arange(-yc, yc - 1))

    # Estimating parameters of the polynomial for the central peak
    xx2 = np.ravel(xx * xx)
    yy2 = np.ravel(yy * yy)
    xy = np.ravel(xx * yy)
    xx = np.ravel(xx)
    yy = np.ravel(yy)
    c = np.ones(xx.shape)
    r = np.ravel(rho_peak)
    A = np.column_stack((xx2, yy2, xy, xx, yy, c))
    p, _, _, _ = np.linalg.lstsq(A, r, rcond=None)

    # reconstructing autocorrelation function using the estimated second order polynomial parameters
    rho_poly = np.reshape(np.dot(A, p), (rho_peak.shape[0], rho_peak.shape[1]))
    sigma = (1. / ((rho_peak.shape[0] - 2) * (rho_peak.shape[1] - 2))) * np.sum(np.power(rho_peak - rho_poly, 2))
    p = np.hstack((p, sigma))

    return np.real(p), np.real(rho)


def lbp(img, r=1.5, th=10, return_patterns=False):
    """
    Estimates the local binary patterns in a radius r around the central pixel based on
    Maenpaa, "The Local Binary Pattern Approach To Texture Analysis: Extensions And Applications." PhD thesis,
    University of Oulu. (2003)

    img: input image, r: the radius of analysis, th: threshold around the neighbour
    returns lbp_hist: histogram of the lbp, lbp_img: lbp image
    """
    if len(img.shape) > 2:
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype("float32")
    else:
        img_gray = img.astype('float32')

    n, __, table = number_neighbour(r, look_at_table=True)

    lbp_img = np.zeros_like(img_gray)

    # Pixels located outside the grid are computed using bilinear interpolation
    xx, yy = np.meshgrid(np.arange(0, img.shape[1]), np.arange(0, img.shape[0]))
    for ii in range(n):
        xi = xx - r * np.sin(2 * np.pi * ii / n)
        yi = yy + r * np.cos(2 * np.pi * ii / n)

        # Comparison against neighbour
        img_ii = cv2.remap(img_gray.astype('float32'), xi.astype('float32'), yi.astype('float32'), cv2.INTER_LINEAR)
        img_ii = np.abs(img_gray - img_ii) > th

        # Accumulating pattern code
        lbp_img += img_ii.astype('float32') * (2 ** ii)

    # Look at tables as proposed by Maenpaa to combine similar patterns
    lbp_img = table[0, lbp_img.astype('int')]

    # Computing histogram of patterns
    lbp_hist = cv2.calcHist([lbp_img], [0], None, [int(n + 1)], [0, n + 1])
    lbp_hist = lbp_hist / lbp_img.size

    if return_patterns:
        n_patterns = len(np.unique(table[0, :]))
        return lbp_hist.T[0], lbp_img, n_patterns
    else:
        return lbp_hist.T[0], lbp_img


# -------------------------------- Filtering based approaches -------------------------------- #


def laws_filter_bank(img):
    """
    Filter bank decomposition using filters proposed by Laws. Mean and standard deviation of each band is
    used as features. Based on
    Laws, "Textured Image Segmentation". PhD thesis, University of Southern California Los Angeles Image
    Processing Inst. (1980)

    img: input image
    returns img_decomposition: input image decomposed using Laws filters, avg_std: Mean and standard dev. of each band
    """
    if len(img.shape) > 2:
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype("float32")
    else:
        img_gray = img.astype('float32')

    img_decomposition = {}
    H = np.ones((11, 11)) / (11 * 11)
    for ii in range(5):
        for jj in range(5):
            # Constructing 2d filter
            G = np.dot(
                np.reshape(LAWS_FILTERS[ii], (LAWS_FILTERS[ii].size, 1)),
                np.reshape(LAWS_FILTERS[jj], (1, LAWS_FILTERS[jj].size))
                )
            # Normalizing filter to keep the image in the same range using the Manhattan distance
            G = G / np.sum(np.abs(G))
            # Applying filter to image
            img_decomposition[ii, jj] = cv2.filter2D(img_gray, -1, G, borderType=cv2.BORDER_REPLICATE)
            img_decomposition[ii, jj] = img_decomposition[ii, jj] * img_decomposition[ii, jj]
            img_decomposition[ii, jj] = np.sqrt(cv2.filter2D(img_decomposition[ii, jj], -1, H, borderType=cv2.BORDER_REPLICATE))
            if ii != 0 and jj != 0:
                img_decomposition[ii, jj] = img_decomposition[ii, jj] / img_decomposition[0, 0]
            else:
                # To avoid dividing by 0
                img_decomposition[0, 0][np.where(img_decomposition[0, 0] == 0)] = 1

    img_decomposition_combined = []
    avg_std = []

    # Combining similar bands and computing the average and standard deviation in the bands
    for ii in range(5):
        for jj in range(ii + 1, 5):
            band = np.abs(img_decomposition[ii, jj] + img_decomposition[jj, ii]) / 2.
            img_decomposition_combined.append(band)
            if ii != 0 and jj != 0:
                avg_std.extend([np.mean(band), np.std(band)])

    return np.array(avg_std), img_decomposition_combined


def eigenfilter(img, r=1):
    """
    Estimates the eigen filters and use them in filter bank decomposition. Mean and standard deviation of each band is
    used as features also based on circular neighbours. Based on
    Ade, "Characterization of textures by eigenfilters. Signal Processing, 5:451 – 457, (1983)

    img: input image,  r: the radius of analysis
    returns img_decomposition: input image decomposed using eigenfilters, avg_std: Mean and standard dev. of each band
    """
    if len(img.shape) > 2:
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype("float32")
    else:
        img_gray = img.astype('float32')

    n, __ = number_neighbour(r)

    # Pixels located outside the grid are computed using bilinear interpolation
    xx, yy = np.meshgrid(np.arange(0, img.shape[1]), np.arange(0, img.shape[0]))
    X = np.reshape(img_gray, (img_gray.size, 1))
    for ii in range(n):
        xi = xx - r * np.sin(2 * np.pi * ii / n)
        yi = yy + r * np.cos(2 * np.pi * ii / n)
        img_ii = cv2.remap(img_gray.astype('float32'), xi.astype('float32'), yi.astype('float32'), cv2.INTER_LINEAR)

        # Allocating pixel values in columns, each row is a pixel and each column is the bilinear interpolated pixel
        X = np.hstack((X, np.reshape(img_ii, (img_ii.size, 1))))

    # Estimating the eigenvectors of the matrix each row is a pixel with its surrounding neighbours
    X = X - np.mean(X, 0)
    cov_matrix = np.cov(X.T)
    __, eigen_vectors = np.linalg.eig(cov_matrix)
    img_decomposition = []
    avg_std = []

    # Decomposing image in sub bands using the estimated eigen vectors
    for ii in range(n):
        # Normalizing filter to keep the same image using the Manhattan distance
        eigen_filter = eigen_vectors[:, ii] / np.sum(np.abs(eigen_vectors[:, ii]))
        img_band = np.abs(np.reshape(np.dot(X, eigen_filter), (img.shape[0], img.shape[1])))
        img_band += np.mean(X[:, ii])
        img_decomposition.append(img_band)
        avg_std.extend([np.mean(img_band), np.std(img_band)])

    return np.array(avg_std), img_decomposition


def ring_wedge_filters_fft(img):
    """
    Estimates the ring and wedge filters and use them in Fourier domain to estimate an image decomposition in sub bands.
    Energy of of each band is used as features. Based on
    Weszka, et.al., "A comparative study of texture measures for terrain classification.", IEEE Transactions On Systems,
    Man, And Cybernetics, 6(4):269 – 285. (1976)

    img: input image,
    returns img_decomposition: input image decomposed using the ring and wedge filters, energy: sum of Fourier
    coefficient magnitudes
    """
    if len(img.shape) > 2:
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype("float32")
    else:
        img_gray = img.astype('float32')

    # This algorithm requires a minimum image size 128 x 128
    if img_gray.shape[0] < 128 or img_gray.shape[1] < 128:
        img_gray = cv2.resize(img_gray, (128, 128), interpolation=cv2.INTER_LINEAR)

    img_fft = np.fft.fft2(np.fft.ifftshift(img_gray))
    img_fft_full = img_fft

    img_fft = np.power(np.absolute(np.fft.fftshift(img_fft)), 2)

    # Fourier filters are designed on 128 x 128 grid and rescale to the image size using nearest neighbour interpolation
    xc = int(FOURIER_DOMAIN_SIZE / 2.)
    yc = int(FOURIER_DOMAIN_SIZE / 2.)

    # normalizing
    img_fft /= np.sqrt(np.sum(np.max(img_fft) - img_fft))

    # Pixel locations in Fourier domain to build the filters
    radii = np.array([[2, 4, 8, 16, 32, 64], [4, 8, 16, 32, 64, 128]])
    theta1 = np.array([[112.5, 67.5, 22.5, 157.5], [67.5, 22.5, 337.5, 112.5]])
    theta2 = np.array([[247.5, 247.5, 202.5, 292.5], [292.5, 202.5, 157.5, 337.5]])
    phi = np.arange(0, 360)
    power_images = []
    energy = []

    # Estimating the ring filters
    for ii in range(radii.shape[1]):
        # Creating the filter (mask in Fourier domain)
        inner_circle = np.zeros((FOURIER_DOMAIN_SIZE, FOURIER_DOMAIN_SIZE)).astype('uint8')
        outer_circle = np.zeros((FOURIER_DOMAIN_SIZE, FOURIER_DOMAIN_SIZE)).astype('uint8')

        xi = xc + radii[0, ii] * np.sin(phi * np.pi / 180.)
        yi = yc + radii[0, ii] * np.cos(phi * np.pi / 180.)
        ((x_center, y_center), radius) = cv2.minEnclosingCircle(np.vstack((xi, yi)).T.astype('float32'))
        cv2.circle(inner_circle, (int(x_center), int(y_center)), int(radius), (1, 1, 1), -1)

        xi = xc + radii[1, ii] * np.sin(phi * np.pi / 180.)
        yi = yc + radii[1, ii] * np.cos(phi * np.pi / 180.)
        ((x_center, y_center), radius) = cv2.minEnclosingCircle(np.vstack((xi, yi)).T.astype('float32'))
        cv2.circle(outer_circle, (int(x_center), int(y_center)), int(radius), (1, 1, 1), -1)

        ring_filter = ((outer_circle - inner_circle) > 0).astype('float')
        # Making mask of the same size of the image
        ring_filter = cv2.resize(ring_filter, img_fft.shape[::-1], interpolation=cv2.INTER_NEAREST)
        energy.append(np.sum(ring_filter * img_fft))
        power_images.append(np.absolute(np.fft.ifftshift(np.fft.ifft2(ring_filter * img_fft_full))))

    rho = 128
    # Estimating the wedge filters
    for ii in range(theta1.shape[1]):
        # Creating the filter (mask in Fourier domain)
        left_wedge = np.zeros((FOURIER_DOMAIN_SIZE, FOURIER_DOMAIN_SIZE)).astype('uint8')
        right_wedge = np.zeros((FOURIER_DOMAIN_SIZE, FOURIER_DOMAIN_SIZE)).astype('uint8')

        xi = xc + rho * np.sin(theta1[0, ii] * np.pi / 180.)
        yi = yc + rho * np.cos(theta1[0, ii] * np.pi / 180.)
        xi = np.vstack((np.vstack((xi, xc + rho * np.sin(theta1[1, ii] * np.pi / 180.))), xc))
        yi = np.vstack((np.vstack((yi, yc + rho * np.cos(theta1[1, ii] * np.pi / 180.))), yc))
        cv2.fillConvexPoly(right_wedge, np.hstack((xi, yi)).astype('int32'), (1, 1, 1))

        xi = xc + rho * np.sin(theta2[0, ii] * np.pi / 180.)
        yi = yc + rho * np.cos(theta2[0, ii] * np.pi / 180.)
        xi = np.vstack((np.vstack((xi, xc + rho * np.sin(theta2[1, ii] * np.pi / 180.))), xc))
        yi = np.vstack((np.vstack((yi, yc + rho * np.cos(theta2[1, ii] * np.pi / 180.))), yc))
        cv2.fillConvexPoly(left_wedge, np.hstack((xi, yi)).astype('int32'), (1, 1, 1))

        wedge_filter = ((left_wedge + right_wedge) > 0).astype('float')
        # Making mask of the same size of the image
        wedge_filter = cv2.resize(wedge_filter, img_fft.shape[::-1], interpolation=cv2.INTER_NEAREST)
        energy.append(np.sum(wedge_filter * img_fft))
        power_images.append(np.absolute(np.fft.ifftshift(np.fft.ifft2(wedge_filter * img_fft_full))))

    return np.array(energy), power_images


def gabor_features(img, uh=0.4, ul=0.05, K=6, S=4):
    """
    Filter bank decomposition using filters proposed by Gabor. Mean and standard deviation of each band is
    used as features. Based on
    Manjunath and Ma, "Texture features for browsing and retrieval of image data". IEEE Transactions on Pattern Analysis
    and Machine Intelligence, 18:837 – 842. (1996)

    img: input image, uh: high cur off frequency, ul: lower cut off frequency, K: number orientations, S: number scales
    returns img_decomposition: input image decomposed using Gabor filters, avg_std: Mean and standard dev. of each band
    """
    if len(img.shape) > 2:
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype("float32")
    else:
        img_gray = img.astype('float32')

    # Parameters of the set of filters proposed by Manjunath and Ma
    W = uh
    nstds = 3
    a = (uh / ul) ** (1. / (S - 1.))
    sigmau = ((a - 1.) * uh) / ((a + 1.) * np.sqrt(2. * np.log(2.)))
    fac1 = np.tan(np.pi/(2. * K))
    fac2 = uh - 2. * np.log((2. * sigmau ** 2.) / uh)
    fac3 = (2. * np.log(2.) - (((2. * np.log(2.)) ** 2.) * (sigmau ** 2.) / (uh ** 2.))) ** (-0.5)
    sigmav = fac1 * fac2 * fac3
    sigmax = 1. / (2. * np.pi * sigmau)
    sigmay = 1. / (2. * np.pi * sigmav)

    # Estimating grid of the filters in the original scale and angle
    xmax = np.maximum(np.abs(nstds * sigmax), np.abs(nstds * sigmay))
    xmax = np.ceil(np.maximum(1, xmax))
    ymax = np.maximum(np.abs(nstds * sigmax), np.abs(nstds * sigmay))
    ymax = np.ceil(np.maximum(1, ymax))
    xmin = -xmax
    ymin = -ymax
    xx, yy = np.meshgrid(np.arange(xmin, xmax + 1), np.arange(ymin, ymax + 1))

    avg_std = []
    img_decomposition = {}

    # Decomposing image and creating filter bank
    fac1 = 1. / (2. * np.pi * sigmax * sigmay)
    for ii in range(K):
        theta = (ii * np.pi) / K
        for jj in range(S):
            # Creating filter for current angle and scale
            xprime = (a ** (-jj)) * (xx * np.cos(theta) + yy * np.sin(theta))
            yprime = (a ** (-jj)) * (-xx * np.sin(theta) + yy * np.cos(theta))
            fac2 = (np.power(xprime, 2) / (sigmax ** 2)) + (np.power(yprime, 2) / (sigmay ** 2))
            gabor_filter = (a ** (-jj)) * fac1 * np.exp(-0.5 * fac2 + 2 * np.pi * 1j * W * xprime)
            gabor_filter = gabor_filter - np.mean(gabor_filter)
            # Normalizing filter to keep the image in the same range
            gabor_filter = gabor_filter / np.sum(np.absolute(gabor_filter))

            # Gabor filter are complex filters which have real and imaginary components
            gabor_filter_i = np.imag(gabor_filter)
            gabor_filter_r = np.real(gabor_filter)

            # Imaginary and real components are estimated independently
            img_filtered_imag = cv2.filter2D(img_gray, -1, gabor_filter_i, borderType=cv2.BORDER_REPLICATE)
            img_filtered_real = cv2.filter2D(img_gray, -1, gabor_filter_r, borderType=cv2.BORDER_REPLICATE)

            # The filter response is estimated as the magnitude of the complex number
            img_decomposition[ii, jj] = np.sqrt(np.power(img_filtered_imag, 2) + np.power(img_filtered_real, 2))
            # Adding 1 to avoid numerical instability
            img_decomposition[ii, jj] = 10 * np.log10(img_decomposition[ii, jj] + 1)
            avg_std.extend([np.mean(img_decomposition[ii, jj]), np.std(img_decomposition[ii, jj])])
    img_decomposition = list(img_decomposition.values())
    return np.array(avg_std), img_decomposition


def laplacian_pyramid(img, scales=3, sigma=1.5, wsize=13):
    """
    Filter bank decomposition using Laplacian pyramid. Mean and standard deviation of each band is
    used as features. Based on
    Burt and Adelson, "The laplacian pyramid as a compact image code". IEEE Transactions on Communications,
    31:532 – 540, (1983)

    img: input image, scales: number of scales, sigma: standard deviation of Gaussian filter window size of analysis
    returns img_decomposition: input image decomposed using Gabor filters, avg_std: Mean and standard dev. of each band
    """
    if len(img.shape) > 2:
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype("float32")
    else:
        img_gray = img.astype('float32')

    # Creating gaussian kernel
    xx, yy = np.meshgrid(np.arange(-wsize / 2, wsize / 2 + 1), np.arange(-wsize / 2, wsize / 2 + 1))
    gaussian_filter = np.exp(-((xx * xx + yy * yy) / (sigma * sigma)))
    gaussian_filter = gaussian_filter / np.sum(gaussian_filter)

    avg_std = []
    img_decomposition = []

    # Computing the image for each band
    for ii in range(scales):
        band = img_gray
        img_gray = cv2.filter2D(img_gray, -1, gaussian_filter, borderType=cv2.BORDER_REPLICATE)

        # The current band is equal to the difference of the current low pass image and the original image
        band = np.abs(band - img_gray)
        img_decomposition.append(cv2.resize(band, img_gray.shape, cv2.INTER_LINEAR))

        avg_std.extend([np.mean(band), np.std(band)])

        # Downsampling for next scale
        img_gray = img_gray[::2, ::2]

    return np.array(avg_std), img_decomposition


def steerable_pyramid(img, scale=3, sigma=1.5, wsize=13):
    """
    Filter bank decomposition using steerable pyramid. Mean and standard deviation of each band is
    used as features. Based on
    Lindeberg and Eklundh, "Scale-space primal sketch: construction and experiments." Image and Vision Computing,
    10(1):3 – 18. (1992)

    img: input image, scales: number of scales, sigma: standard deviation of Gaussian filter window size of analysis
    returns img_decomposition: input image decomposed using Gabor filters, avg_std: Mean and standard dev. of each band
    """
    if len(img.shape) > 2:
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype("float32")
    else:
        img_gray = img.astype('float32')

    # Creating gaussian kernel
    xx, yy = np.meshgrid(np.arange(-wsize / 2, wsize / 2 + 1), np.arange(-wsize / 2, wsize / 2 + 1))
    gaussian_filter = np.exp(-((xx * xx + yy * yy) / (sigma * sigma)))
    gaussian_filter = gaussian_filter / np.sum(gaussian_filter)

    # Estimating derivatives of the Gaussian kernel
    gaussian_filter_dx, gaussian_filter_dy = np.gradient(gaussian_filter)
    gaussian_filter_dx = gaussian_filter_dx / np.sum(np.abs(gaussian_filter_dx))
    gaussian_filter_dy = gaussian_filter_dy / np.sum(np.abs(gaussian_filter_dy))

    # Defining angles of analysis
    theta = [0., 45., 90., 135.]
    avg_std = []
    img_decomposition = {}

    # Computing the image for each band
    for ii in range(scale):
        img_gray_dx = cv2.filter2D(img_gray, -1, gaussian_filter_dx, borderType=cv2.BORDER_REPLICATE)
        img_gray_dy = cv2.filter2D(img_gray, -1, gaussian_filter_dy, borderType=cv2.BORDER_REPLICATE)
        for jj in range(len(theta)):
            band = np.cos(np.pi * theta[jj] / 180.) * img_gray_dx + np.sin(np.pi * theta[jj] / 180.) * img_gray_dy
            avg_std.extend([np.mean(band), np.std(band)])
            img_decomposition[ii, jj] = cv2.resize(band, img_gray.shape, cv2.INTER_LINEAR)

        # Downsampling for next scale
        img_gray = cv2.filter2D(img_gray, -1, gaussian_filter, borderType=cv2.BORDER_REPLICATE)
        img_gray = img_gray[::2, ::2]
    img_decomposition = list(img_decomposition.values())
    return np.array(avg_std), img_decomposition


# -------------------------------- Structural based approaches -------------------------------- #


def granulometry_moments(img, scales=20):
    """
    Computes granulometry and anti-granulometry moments. Based on
    Aptoula and Lefevre, "Advances in Imaging and Electron Physics", chapter Morphological texture
    description of grayscale and color images, pages 1 – 74. Elsevier. (2011)

    img: input image, scales: number of scales
    returns img_decomposition: input image decomposed using morphological filters, features: descriptive statistics
    """
    if len(img.shape) > 2:
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype("float32")
    else:
        img_gray = img.astype('float32')

    # Descriptive statistics from the original image, it is used to normalize the statistics from other scales
    avg = np.mean(img_gray)
    var = np.var(img_gray)
    ske, kur = skewness_kurtosis(img_gray, avg)

    ave_open = []
    var_open = []
    ske_open = []
    kur_open = []
    ave_close = []
    var_close = []
    ske_close = []
    kur_close = []
    img_decomposition = {}
    for ii in range(scales):
        # The structuring element changes in every scale (grows)
        structuring_element = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ii+2, ii+2))

        # Morphological opening (granulometry)
        img_open = cv2.morphologyEx(img_gray, cv2.MORPH_OPEN, structuring_element)
        avg_ii = np.mean(img_open)
        ave_open.append(1. - avg_ii / avg)
        var_open.append(1. - np.var(img_open) / var)
        ske_ii, kur_ii = skewness_kurtosis(img_open, avg_ii)
        ske_open.append(1. - ske_ii / ske)
        kur_open.append(1. - kur_ii / kur)

        # Morphological closing (anti-granulometry)
        img_close = cv2.morphologyEx(img_gray, cv2.MORPH_CLOSE, structuring_element)
        avg_ii = np.mean(img_close)
        ave_close.append(1. - avg_ii / avg)
        var_close.append(1. - np.var(img_close) / var)
        ske_ii, kur_ii = skewness_kurtosis(img_close, avg_ii)
        ske_close.append(1. - ske_ii / ske)
        kur_close.append(1. - kur_ii / kur)

        img_decomposition[ii] = (img_open, img_close)
    feat_open = np.hstack((np.array(ave_open), np.array(var_open), np.array(ske_open), np.array(kur_open)))
    feat_close = np.hstack((np.array(ave_close), np.array(var_close), np.array(ske_close), np.array(kur_close)))

    return np.hstack((feat_open, feat_close)), img_decomposition
