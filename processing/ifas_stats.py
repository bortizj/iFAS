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
from scipy import stats
from gui.ifas_misc import convert_ifnan

# Distance correlation between 2 given vectors
def dis_correlation(x, y):
    n = x.size
    a = np.abs(x[:, None] - x)
    b = np.abs(y[:, None] - y)
    A = a - np.mean(a, axis=0) - np.mean(a, axis=1)[:, None] + np.mean(a)
    B = b - np.mean(b, axis=0) - np.mean(b, axis=1)[:, None] + np.mean(b)
    dcov2_xy = np.sum(A * B) / float(n * n)
    dcov2_xx = np.sum(A * A) / float(n * n)
    dcov2_yy = np.sum(B * B) / float(n * n)
    den = np.sqrt((np.sqrt(dcov2_xx) * np.sqrt(dcov2_yy)))
    if den != 0:
        return np.sqrt(dcov2_xy) / np.sqrt((np.sqrt(dcov2_xx) * np.sqrt(dcov2_yy)))
    else:
        return 0


# Correlation between 2 given vectors or between columns of a matrix
def compute_1dcorrelations(data, data_y=None):
    # If second input is given then the correlation is between those 2
    if data_y is not None:
        # if second data is given then only single correlation is computed
        # any nan is changed for 0
        x = convert_ifnan(data.ravel())
        y = convert_ifnan(data_y.ravel())
        # correlation with constants return zero
        if np.unique(x).size == 1 or np.unique(y).size == 1:
            p, s, t, pd = 0, 0, 0, 0
        else:
            p, __ = stats.pearsonr(x, y)
            s, __ = stats.spearmanr(x, y)
            t, __ = stats.kendalltau(x, y)
            pd = dis_correlation(x, y)

        return p, s, t, pd

    # computing correlation matrices
    p = np.zeros((data.shape[1], data.shape[1]))
    s = np.zeros((data.shape[1], data.shape[1]))
    t = np.zeros((data.shape[1], data.shape[1]))
    pd = np.zeros((data.shape[1], data.shape[1]))
    data = data.astype("float")
    for ii in range(data.shape[1]):
        for jj in range(data.shape[1]):
            x = convert_ifnan(data[::, ii])
            y = convert_ifnan(data[::, jj])
            if np.unique(x).size > 1 and np.unique(y).size > 1:
                p[ii, jj], __ = stats.pearsonr(x, y)
                s[ii, jj], __ = stats.spearmanr(x, y)
                t[ii, jj], __ = stats.kendalltau(x, y)
                pd[ii, jj] = dis_correlation(x, y)

    return p, s, t, pd
