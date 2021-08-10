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

from iFAS.processing import compute_contrast


def sme_dif(ref_img, tst_img):
    sme_ref, __ = compute_contrast.sme(ref_img)
    sme_tst, __ = compute_contrast.sme(tst_img)
    dif = sme_ref - sme_tst

    return dif


def wme_dif(ref_img, tst_img):
    wme_ref, __ = compute_contrast.wme(ref_img)
    wme_tst, __ = compute_contrast.wme(tst_img)
    dif = wme_ref - wme_tst

    return dif


def mme_dif(ref_img, tst_img):
    mme_ref, __ = compute_contrast.mme(ref_img)
    mme_tst, __ = compute_contrast.mme(tst_img)
    dif = mme_ref - mme_tst

    return dif


def rme_dif(ref_img, tst_img):
    rme_ref, __ = compute_contrast.rme(ref_img)
    rme_tst, __ = compute_contrast.rme(tst_img)
    dif = rme_ref - rme_tst

    return dif


def peli_dif(ref_img, tst_img):
    peli_ref, __ = compute_contrast.contrast_peli(ref_img)
    peli_tst, __ = compute_contrast.contrast_peli(tst_img)
    dif = peli_ref - peli_tst

    return dif
