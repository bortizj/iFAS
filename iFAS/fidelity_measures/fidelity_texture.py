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


import numpy as np
from iFAS.processing import compute_texture


def ar2d_dif(ref_img, tst_img):
    ar_ref, __ = compute_texture.autoregressive_model(ref_img, r=1.5)
    ar_tst, __ = compute_texture.autoregressive_model(tst_img, r=1.5)
    dif = np.sqrt(np.sum(np.power(ar_ref - ar_tst, 2)))

    return dif


def gmrf_dif(ref_img, tst_img):
    alpha_ref, __ = compute_texture.gaussian_markov_random_field(ref_img)
    alpha_tst, __ = compute_texture.gaussian_markov_random_field(tst_img)
    dif = np.sqrt(np.sum(np.power(alpha_ref - alpha_tst, 2)))

    return dif


def glcm_dif(ref_img, tst_img):
    ar_ref, __ = compute_texture.gray_level_cooccurrence_matrix(ref_img)
    ar_pro, __ = compute_texture.gray_level_cooccurrence_matrix(tst_img)
    dif = np.sqrt(np.sum(np.power(ar_ref - ar_pro, 2)))

    return dif


def acf2d_dif(ref_img, tst_img):
    ar_ref, __ = compute_texture.autocorrelation(ref_img)
    ar_tst, __ = compute_texture.autocorrelation(tst_img)
    dif = np.sqrt(np.sum(np.power(ar_ref - ar_tst, 2)))

    return dif


def lbp_dif(ref_img, tst_img):
    hist_ref, __ = compute_texture.lbp(ref_img, r=1.5)
    hist_tst, __ = compute_texture.lbp(tst_img, r=1.5)

    # Kullback Leibler Divergence
    if np.max(hist_ref) > 1 or np.max(hist_tst) > 1:
        hist_ref = 1. * hist_ref / np.sum(hist_ref)
        hist_tst = 1. * hist_tst / np.sum(hist_tst)
    h = hist_ref + hist_tst
    h = h[h != 0]
    hist_ref = hist_ref[hist_ref != 0]
    hist_tst = hist_tst[hist_tst != 0]
    dif = (
        2. * np.log(2.) + np.sum(hist_ref * np.log(hist_ref)) + np.sum(hist_tst * np.log(hist_tst)) - 
        np.sum(h * np.log(h))
        )

    return dif


def laws_dif(ref_img, tst_img):
    alpha_ref, __ = compute_texture.laws_filter_bank(ref_img)
    alpha_tst, __ = compute_texture.laws_filter_bank(tst_img)
    dif = np.sqrt(np.sum(np.power(alpha_ref - alpha_tst, 2)))

    return dif


def eig_dif(ref_img, tst_img):
    ar_ref, __ = compute_texture.eigenfilter(ref_img)
    ar_tst, __ = compute_texture.eigenfilter(tst_img)
    dif = np.sqrt(np.sum(np.power(ar_ref - ar_tst, 2)))

    return dif


def fft_dif(ref_img, tst_img):
    psfft_ref, __ = compute_texture.ring_wedge_filters_fft(ref_img)
    psfft_tst, __ = compute_texture.ring_wedge_filters_fft(tst_img)
    dif = np.sqrt(np.sum(np.power(psfft_ref - psfft_tst, 2)))

    return dif


def gabor_dif(ref_img, tst_img):
    gabor_ref, __ = compute_texture.gabor_features(ref_img)
    gabor_tst, __ = compute_texture.gabor_features(tst_img)
    dif = np.sqrt(np.sum(np.power(gabor_ref - gabor_tst, 2)))

    return dif


def laplacian_dif(ref_img, tst_img):
    lap_ref, __ = compute_texture.laplacian_pyramid(ref_img)
    lap_tst, __ = compute_texture.laplacian_pyramid(tst_img)
    dif = np.sqrt(np.sum(np.power(lap_ref - lap_tst, 2)))

    return dif


def steerable_dif(ref_img, tst_img):
    ste_ref, __ = compute_texture.steerable_pyramid(ref_img)
    ste_tst, __ = compute_texture.steerable_pyramid(tst_img)
    dif = np.sqrt(np.sum(np.power(ste_ref - ste_tst, 2)))

    return dif


def granulometry_dif(ref_img, tst_img):
    gra_ref, __ = compute_texture.granulometry_moments(ref_img)
    gra_tst, __ = compute_texture.granulometry_moments(tst_img)
    dif = np.sqrt(np.sum(np.power(gra_ref - gra_tst, 2)))

    return dif
