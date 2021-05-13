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

# This file contains helper functions which do not belong to any class

import matplotlib.pyplot as plt
import numpy as np
from gui.ifas_misc import convert_ifnan

# Encoding names with keyboard keys
set_of_keys = ("right", "left", "up", "down")


# Heat map for the given correlation matrices 
def heat_map(pearson, spearman, ccd, tau):
    # Given 4 matrices a heat map is ploted for each one
    fig, ((map_pearson, map_spearman), (map_ccd, map_tau)) = plt.subplots(2, 2)
    fig.canvas.set_window_title("Correlation heat map plot")
    map_pearson.tick_params(labelsize=16)
    map_spearman.tick_params(labelsize=16)
    map_ccd.tick_params(labelsize=16)
    map_tau.tick_params(labelsize=16)

    map_pearson.imshow(np.abs(pearson), vmin=0., vmax=1., interpolation="none", aspect="equal", origin="lower", cmap="inferno")
    map_spearman.imshow(np.abs(spearman), vmin=0., vmax=1., interpolation="none", aspect="equal", origin="lower", cmap="inferno")
    map_ccd.imshow(np.abs(ccd), vmin=0., vmax=1., interpolation="none", aspect="equal", origin="lower", cmap="inferno")
    axxes = map_tau.imshow(np.abs(tau), vmin=0., vmax=1., interpolation="none", aspect="equal", origin="lower", cmap="inferno")

    map_pearson.set_xticks(range(pearson.shape[1]))
    map_pearson.set_yticks(range(pearson.shape[0]))
    map_pearson.set_xticklabels(range(pearson.shape[1]), fontdict={"fontweight": 16})
    map_pearson.set_yticklabels(range(pearson.shape[0]), fontdict={"fontweight": 16})
    map_spearman.set_xticks(range(pearson.shape[1]))
    map_spearman.set_yticks(range(pearson.shape[0]))
    map_spearman.set_xticklabels(range(spearman.shape[1]), fontdict={"fontweight": 16})
    map_spearman.set_yticklabels(range(spearman.shape[0]), fontdict={"fontweight": 16})
    map_ccd.set_xticks(range(pearson.shape[1]))
    map_ccd.set_yticks(range(pearson.shape[0]))
    map_ccd.set_xticklabels(range(ccd.shape[1]), fontdict={"fontweight": 16})
    map_ccd.set_yticklabels(range(ccd.shape[0]), fontdict={"fontweight": 16})
    map_tau.set_xticks(range(pearson.shape[1]))
    map_tau.set_yticks(range(pearson.shape[0]))
    map_tau.set_xticklabels(range(tau.shape[1]), fontdict={"fontweight": 16})
    map_tau.set_yticklabels(range(tau.shape[0]), fontdict={"fontweight": 16})

    cax = fig.add_axes([0.925, 0.1, 0.03, 0.8])
    cbar = fig.colorbar(axxes, cax=cax)
    cbar.ax.tick_params(labelsize=16)

    map_pearson.set_title("Pearson", fontsize=16)
    map_spearman.set_title("Spearman", fontsize=16)
    map_ccd.set_title("Distance correlation", fontsize=16)
    map_tau.set_title("tau", fontsize=16)
    plt.get_current_fig_manager().window.showMaximized()

    plt.subplots_adjust(top=0.95, bottom=0.1, left=0.05, right=0.95, hspace=0.2, wspace=0.05)
    plt.show(block=False)


# Bar plot of the correlations for the given data matrix
def bar_plot(correlations, axes_labels=None, target_var_idx=""):
    n_var = correlations.shape[0]
    if axes_labels is None:
        axes_labels = range(n_var)
    correlations = np.abs(correlations)

    fig, acorrs = plt.subplots(1, 1)
    fig.canvas.set_window_title("Correlation bar plot with " + target_var_idx)
    acorrs.tick_params(labelsize=16)
    p = correlations[::, 0].tolist()
    s = correlations[::, 1].tolist()
    t = correlations[::, 2].tolist()
    pd = correlations[::, 3].tolist()

    width = np.round(1. / (len(axes_labels) + 1), 6)
    acorrs.bar(np.arange(len(p)) + 0 * width, p, width, color="r", label="Pearson R")
    acorrs.bar(np.arange(len(s)) + 1 * width, s, width, color="g", label="Spearman R")
    acorrs.bar(np.arange(len(t)) + 2 * width, t, width, color="b",label="Kendall")
    acorrs.bar(np.arange(len(pd)) + 3 * width, pd, width, color="y", label="Distance R")

    box = acorrs.get_position()
    acorrs.set_position([box.x0 - 0.065, box.y0 + 0.05, box.width - 0.05, box.height])
    acorrs.legend(framealpha=0.1, loc="center left", bbox_to_anchor=(1, 0.5), fontsize=16)
    acorrs.set_xticks(np.arange(len(p)) + 1. * width + width / 2.)
    acorrs.set_xticklabels(axes_labels)
    acorrs.grid(False)
    acorrs.set_ylim([0, 1])
    acorrs.set_xlim([-0.5, len(p) + 0.])

    plt.get_current_fig_manager().window.showMaximized()
    plt.subplots_adjust(top=0.95, bottom=0.1, left=0.05, right=0.85, hspace=0.2, wspace=0.05)
    plt.show(block=False)
    
# Box plot of the correlations for the given data matrix per source
def box_plot(correlations, axes_labels=None, target_var_idx=""):
    p = correlations[::, ::, 0]
    s = correlations[::, ::, 1]
    t = correlations[::, ::, 2]
    pd = correlations[::, ::, 3]

    width = np.round(1. / (len(axes_labels) + 1), 6)
    fig, acorrs = plt.subplots(1, 1)
    fig.canvas.set_window_title("Correlation box plot with " + target_var_idx)
    acorrs.tick_params(labelsize=16)
    # Work around to only generate one legend per type of correlation
    acorrs.bar([0], [0], width, color="r", label="Pearson R")
    acorrs.bar([0], [0], width, color="g", label="Spearman R")
    acorrs.bar([0], [0], width, color="b", label="Kendall")
    acorrs.bar([0], [0], width, color="y", label="Distance R")
    acorrs.legend(framealpha=0.1, loc="center left", bbox_to_anchor=(1, 0.5), fontsize=16)

    acorrs.boxplot(
        np.abs(p), positions=np.arange(len(axes_labels)) + 0 * width + width / 8, 
        boxprops=dict(color="r", linewidth=5, markersize=12), widths=width
        )
    acorrs.boxplot(
        np.abs(s), positions=np.arange(len(axes_labels)) + 1 * width + width / 8, 
        boxprops=dict(color="g", linewidth=5, markersize=12), widths=width
        )
    acorrs.boxplot(
        np.abs(t), positions=np.arange(len(axes_labels)) + 2 * width + width / 8, 
        boxprops=dict(color="b", linewidth=5, markersize=12), widths=width
        )
    acorrs.boxplot(
        np.abs(pd), positions=np.arange(len(axes_labels)) + 3 * width + width / 8, 
        boxprops=dict(color="y", linewidth=5, markersize=12), widths=width
        )
    
    box = acorrs.get_position()
    acorrs.set_position([box.x0 - 0.065, box.y0 + 0.05, box.width - 0.05, box.height])
    acorrs.set_xticks(np.arange(len(axes_labels)) + 1. * width)
    acorrs.set_ylim([-0.005, 1.005])
    acorrs.set_xlim([-0.5, p.shape[1] + 0.])
    acorrs.set_xticklabels(axes_labels)

    plt.get_current_fig_manager().window.showMaximized()
    plt.subplots_adjust(top=0.95, bottom=0.1, left=0.05, right=0.85, hspace=0.2, wspace=0.05)
    plt.show(block=False)


# Scatter plot for the given data matrix
class ScatterPlotWithHistograms(object):
    """
    Use "left", "right", "up", "down" keys to browse through the next and previous feature
    """
    def __init__(self, data, axes_labels=None):
        self.data = data
        self.n_var = data.shape[1]
        if axes_labels is None:
            self.axes_labels = range(self.n_var)
        else:
            self.axes_labels = axes_labels

        self.feat_idx_x = 0
        if self.n_var <= 1:
            self.feat_idx_y = 0
        else:
            self.feat_idx_y = 1

        # Setting up plot places
        self.fig = plt.figure("iFAS - Scatter plot with histograms widget", figsize=(9.5, 9.5))

        # definitions for the axes
        left, width = 0.1, 0.65
        bottom, height = 0.1, 0.65
        spacing = 0.005

        rect_scatter = [left, bottom, width, height]
        rect_histx = [left, bottom + height + spacing, width, 0.2]
        rect_histy = [left + width + spacing, bottom, 0.2, height]

        self.ax_scatter = self.fig.add_axes(rect_scatter)
        self.ax_scatter.tick_params(direction="in", top=True, right=True)
        self.ax_histx = self.fig.add_axes(rect_histx)
        self.ax_histx.tick_params(direction="in", labelbottom=False)
        self.ax_histy = self.fig.add_axes(rect_histy)
        self.ax_histy.tick_params(direction="in", labelleft=False)

        self.fig.canvas.mpl_connect("key_press_event", self.on_key)

        self.scatter_plot()
        self.ax_scatter.tick_params("both", labelsize=18)
        self.ax_histx.tick_params("x", labelsize=18)
        self.ax_histy.tick_params("y", labelsize=18)
        self.ax_histx.grid(True)
        self.ax_histy.grid(True)
        self.ax_scatter.grid(True)
        plt.show(block=False)

    def scatter_plot(self):
        x = convert_ifnan(self.data[:, self.feat_idx_x])
        y = convert_ifnan(self.data[:, self.feat_idx_y])

        self.ax_scatter.cla()
        self.ax_histx.cla()
        self.ax_histy.cla()

        # the scatter plot:
        self.ax_scatter.plot(x, y, "rx", ms=3)
        self.ax_scatter.set_xlabel(self.axes_labels[self.feat_idx_x], fontsize=18)
        self.ax_scatter.set_ylabel(self.axes_labels[self.feat_idx_y], fontsize=18)

        # Determine limits:
        nbins = 50
        x_min = np.min(x)
        x_max = np.max(x)
        y_min = np.min(y)
        y_max = np.max(y)
        if x_min != x_max:
            self.ax_scatter.set_xlim((x_min, x_max))
        if y_min != y_max:
            self.ax_scatter.set_ylim((y_min, y_max))

        # Determining the histograms
        bins_x = np.linspace(x_min, x_max, nbins)
        bins_y = np.linspace(y_min, y_max, nbins)
        height_x = np.mean(bins_x[1:] - bins_x[:-1]) / 2.
        height_y = np.mean(bins_y[1:] - bins_y[:-1]) / 2.

        hist_x, __ = np.histogram(x, bins_x)
        hist_x = hist_x / np.sum(hist_x)

        self.ax_histx.bar(list((bins_x[1:] + bins_x[:-1]) / 2.), list(hist_x), fc=(0, 1, 0, 0.5), width=height_x)

        hist_y, __ = np.histogram(y, bins_y)
        hist_y = hist_y / np.sum(hist_y)

        self.ax_histy.barh(list((bins_y[1:] + bins_y[:-1]) / 2.), list(hist_y), fc=(0, 1, 0, 0.5), height=height_y)

        self.ax_histx.set_xlim(self.ax_scatter.get_xlim())
        self.ax_histy.set_ylim(self.ax_scatter.get_ylim())

        self.ax_histx.grid(True)
        self.ax_histy.grid(True)
        self.ax_scatter.grid(True)
        plt.draw_all()

    def on_key(self, event):
        inc_x = 0
        inc_y = 0
        if event.key not in set_of_keys:
            return
        if event.key == "right":
            inc_x = 1
        elif event.key == "left":
            inc_x = -1
        elif event.key == "down":
            inc_y = -1
        elif event.key == "up":
            inc_y = 1

        self.feat_idx_x += inc_x
        self.feat_idx_y += inc_y

        self.feat_idx_x = np.clip(self.feat_idx_x, 0, self.n_var - 1)
        self.feat_idx_y = np.clip(self.feat_idx_y, 0, self.n_var - 1)

        self.scatter_plot()


# Scatter plot for the given data matrix against a target and the regression values
class ScatterPlotTargetWithHistograms(object):
    """
    Use "left", "right", "up", "down" keys to browse through the next and previous feature
    """
    def __init__(self, data, pdata, pdatax, target, axes_labels=None):
        self.data = data
        self.pdata = pdata
        self.pdatax = pdatax
        self.y_var = data[::, target]
        self.n_var = data.shape[1]
        if axes_labels is None:
            self.axes_labels = range(self.n_var)
        else:
            self.axes_labels = axes_labels

        self.feat_idx_x = 0
        self.feat_idx_y = target

        # Setting up plot places
        self.fig = plt.figure("iFAS - Scatter plot with histograms widget", figsize=(9.5, 9.5))

        # definitions for the axes
        left, width = 0.1, 0.65
        bottom, height = 0.1, 0.65
        spacing = 0.005

        rect_scatter = [left, bottom, width, height]
        rect_histx = [left, bottom + height + spacing, width, 0.2]
        rect_histy = [left + width + spacing, bottom, 0.2, height]

        self.ax_scatter = self.fig.add_axes(rect_scatter)
        self.ax_scatter.tick_params(direction="in", top=True, right=True)
        self.ax_histx = self.fig.add_axes(rect_histx)
        self.ax_histx.tick_params(direction="in", labelbottom=False)
        self.ax_histy = self.fig.add_axes(rect_histy)
        self.ax_histy.tick_params(direction="in", labelleft=False)

        self.fig.canvas.mpl_connect("key_press_event", self.on_key)

        self.scatter_plot()
        self.ax_scatter.tick_params("both", labelsize=18)
        self.ax_histx.tick_params("x", labelsize=18)
        self.ax_histy.tick_params("y", labelsize=18)
        self.ax_histx.grid(True)
        self.ax_histy.grid(True)
        self.ax_scatter.grid(True)
        plt.show(block=False)

    def scatter_plot(self):
        x = convert_ifnan(self.data[:, self.feat_idx_x])
        y = convert_ifnan(self.data[:, self.feat_idx_y])
        xp = convert_ifnan(self.pdatax[:, self.feat_idx_x])
        yp = convert_ifnan(self.pdata[:, self.feat_idx_x])

        if np.sum(np.isnan(x)) != 0:
            x = np.zeros_like(x)
        if np.sum(np.isnan(y)) != 0:
            y = np.zeros_like(y)
        if np.sum(np.isnan(y)) != 0:
            y = np.zeros_like(y)

        self.ax_scatter.cla()
        self.ax_histx.cla()
        self.ax_histy.cla()

        # the scatter plot:
        self.ax_scatter.plot(xp, yp, "b-", ms=3, lw=3)
        self.ax_scatter.plot(x, y, "rx", ms=5, lw=3)
        self.ax_scatter.set_xlabel(self.axes_labels[self.feat_idx_x], fontsize=18)
        self.ax_scatter.set_ylabel(self.axes_labels[self.feat_idx_y], fontsize=18)

        # Determine limits:
        nbins = 50
        x_min = np.min(x)
        x_max = np.max(x)
        y_min = np.min(y)
        y_max = np.max(y)

        if x_min != x_max:
            self.ax_scatter.set_xlim((x_min, x_max))
        if y_min != y_max:
            self.ax_scatter.set_ylim((y_min, y_max))

        # Determining the histograms
        bins_x = np.linspace(x_min, x_max, nbins)
        bins_y = np.linspace(y_min, y_max, nbins)
        height_x = np.mean(bins_x[1:] - bins_x[:-1]) / 2.
        height_y = np.mean(bins_y[1:] - bins_y[:-1]) / 2.

        hist_x, __ = np.histogram(x, bins_x)
        hist_x = hist_x / np.sum(hist_x)

        self.ax_histx.bar(list((bins_x[1:] + bins_x[:-1]) / 2.), list(hist_x), fc=(0, 1, 0, 0.5), width=height_x)

        hist_y, __ = np.histogram(y, bins_y)
        hist_y = hist_y / np.sum(hist_y)

        self.ax_histy.barh(list((bins_y[1:] + bins_y[:-1]) / 2.), list(hist_y), fc=(0, 1, 0, 0.5), height=height_y)

        self.ax_histx.set_xlim(self.ax_scatter.get_xlim())
        self.ax_histy.set_ylim(self.ax_scatter.get_ylim())

        self.ax_histx.grid(True)
        self.ax_histy.grid(True)
        self.ax_scatter.grid(True)
        plt.draw_all()

    def on_key(self, event):
        inc_x = 0
        inc_y = 0
        if event.key not in set_of_keys:
            return
        if event.key == "right":
            inc_x = 1
        elif event.key == "left":
            inc_x = -1
        elif event.key == "down":
            inc_y = 0
        elif event.key == "up":
            inc_y = 0

        self.feat_idx_x += inc_x
        self.feat_idx_y += inc_y

        self.feat_idx_x = np.clip(self.feat_idx_x, 0, self.n_var - 1)
        self.feat_idx_y = np.clip(self.feat_idx_y, 0, self.n_var - 1)

        self.scatter_plot()
