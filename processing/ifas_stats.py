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

author: Benhur Ortiz Jaramillo
"""

import numpy as np
from scipy import stats

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


def compute_1dcorrelations(data):
    p = np.zeros((data.shape[1], data.shape[1]))
    s = np.zeros((data.shape[1], data.shape[1]))
    t = np.zeros((data.shape[1], data.shape[1]))
    pd = np.zeros((data.shape[1], data.shape[1]))
    data = data.astype('float')
    for ii in range(data.shape[1]):
        for jj in range(data.shape[1]):
            if not (np.any(np.isnan(data[::, ii])) or np.any(np.isnan(data[::, jj]))):
                p[ii, jj], __ = stats.pearsonr(data[::, ii], data[::, jj])
                s[ii, jj], __ = stats.spearmanr(data[::, ii], data[::, jj])
                t[ii, jj], __ = stats.kendalltau(data[::, ii], data[::, jj])
                pd[ii, jj] = dis_correlation(data[::, ii], data[::, jj])

    return p, s, t, pd
