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
from scipy import optimize
from scipy import special

from processing.ifas_stats import compute_1dcorrelations
from gui import ifas_plotting

# Defining the available models (error functions) -> the most relevant for image fidelity assessment

def linear(a, x, y):
    return (a[0] + a[1] * x) - y


def quadratic(a, x, y):
    return (a[0] + a[1] * x + a[2] * np.power(x,2)) - y


def cubic(a, x, y):
    return (a[0] + a[1] * x + a[2] * np.power(x, 2) + a[3] * np.power(x, 3)) - y


def exponential(a, x, y):
    return (a[0] + np.exp(a[1] * x + a[2])) - y


def logistic(a, x, y):
    return (a[0] / (1 + np.exp(-(a[1] * x + a[2])))) - y


def complementary_error(a, x, y):
    return (1 - 0.5 * (1 + special.erf((x - a[0]) / (np.sqrt(2) * a[1])))) - y


class RegressionModel(object):
    """
    Class object to initialize the regression model
    """
    def __init__(self, model_type='linear', ini_par=None):
        self.model_type = model_type
        self.ini_par = ini_par

    def optimize_model(self, x, y):
        res_soft_l1 = optimize.least_squares(
            globals()[self.model_type], self.ini_par, loss='soft_l1', f_scale=0.1, args=(x, y)
            )
        # returns the solution x -> parameters of the function to optimize
        return res_soft_l1.x

    def optimize_over_data(self, in_mat, in_y):
        n_var = in_mat.shape[1]
        parameters = []
        for ii in range(n_var):
            par = self.optimize_model(in_mat[::, ii], in_y)
            parameters.append(par)

        return parameters

    def evaluate_over_data(self, in_par, in_mat):
        n_var = in_mat.shape[1]
        y_p = []
        x_p = []
        for ii in range(n_var):
            x = np.linspace(np.min(in_mat[::, ii]), np.max(in_mat[::, ii]), in_mat[::, ii].size)
            y_temp = self.evaluate_model(in_par[ii], x)
            y_p.append(y_temp)
            x_p.append(x)

        return np.array(x_p).T, np.array(y_p).T

    def evaluate_model(self, a, x):
        if self.model_type == 'linear':
            y = a[0] + a[1] * x
        elif self.model_type == 'quadratic':
            y = a[0] + a[1] * x + a[2] * np.power(x,2)
        elif self.model_type == 'cubic':
            y = a[0] + a[1] * x + a[2] * np.power(x, 2) + a[3] * np.power(x, 3)
        elif self.model_type == 'exponential':
            y = a[0] + np.exp(a[1] * x + a[2])
        elif self.model_type == 'logistic':
            y = a[0] / (1 + np.exp(-(a[1] * x + a[2])))
        elif self.model_type == 'complementary_error':
            y = 1 - 0.5 * (1 + special.erf((x - a[0]) / (np.sqrt(2) * a[1])))
        else:
            y = None

        return y
