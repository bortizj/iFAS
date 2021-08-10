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

import tkinter as tk
import tkinter.ttk as ttk
import tkinter.messagebox
import tempfile
import pathlib
import shutil
import os
import numpy as np
import pandas as pd
import cv2

from iFAS.processing.ifas_stats import compute_1dcorrelations
from iFAS.processing.regression_tools import RegressionModel
from iFAS.gui import ifas_plotting

class ImgDatabase(object):
    """
    Class object to initialize the database of images
    the input is a the string with the folder path of the set of images
    """
    def __init__(self, data_folder):
        self.db_folder = pathlib.Path(data_folder)
        self.get_img_paths()

    # Set of fidelity measures which will be computed on the set of images
    def set_test_measures(self, measures):
        self.measures = measures

    # Reference to the logging instance where 
    def set_logger(self, logger):
        self.logger = logger

    # Resolves the image paths and put them in a dictionary key -> ref image values -> test images
    def get_img_paths(self):
        flag = False
        # Removing first row -> It is the root folder
        list_folders = list(self.db_folder.glob("**"))[1:]
        # one liner for getting only the file name of the reference images
        self.list_ref = [str(ii_folder).split(os.sep)[-1] for ii_folder in list_folders]
        self.dict_tst = {}
        for ii in range(len(self.list_ref)):
            list_imgs_crt = list(list_folders[ii].glob("*.png"))

            # If any of the selected sub-folders is empty, it is a good indication for wrong selected folder
            if len(list_imgs_crt) == 0:
                flag = True

            # one liner for getting only the file name of the test images for the current folder
            self.dict_tst[self.list_ref[ii]] = [str(ii_img).split(os.sep)[-1] for ii_img in list_imgs_crt]
            # removing the source image from the list of test images
            if self.list_ref[ii] + ".png" in self.dict_tst[self.list_ref[ii]]:
                self.dict_tst[self.list_ref[ii]].remove(self.list_ref[ii] + ".png")
            else:
                # If reference image not in list good indication folder is incorrect
                flag = True
        if flag:
            self.dict_tst = None
            self.list_ref = None

    # Loading the pandas data frame if available 
    def get_csv(self):
        if self.db_folder.joinpath(self.db_folder.name + "_ifas_ouput.csv").is_file():
            try:
                self.data = pd.read_csv(str(self.db_folder.joinpath(self.db_folder.name + "_ifas_ouput.csv")))
            except:
                self.data = None
        else:
            self.data = None

    # Executing the process using the database
    def compute_measures(self):
        psh = ProcessHandler(self, self.measures)
        try:
            psh.process_data()
        except Exception as error:
                self.logger.print(level="ERROR", message="Processing database " + repr(error))
                try:
                    tk.messagebox.showerror("Error", "Something went wrong processing! \n" + repr(error), master=self.win)
                except:
                    return

    # Computing the correlations for the set of data
    def compute_correlations(self):
        # First 2 columns are the file names
        p, s, t, pd = compute_1dcorrelations(self.data.values[::, 2:])
        self.pearson = p
        self.spearman = s
        self.tau = t
        self.dist_corr = pd
        self.save_correlations()
        ifas_plotting.heat_map(p, s, pd, t)

    # Computing the correlations for the set of data against the target column
    def get_correlations_with(self, idx):
        correlations = np.vstack((
            self.pearson[::, idx], self.spearman[::, idx], self.tau[::, idx] , self.dist_corr[::, idx]
            )).T
        correlations = np.delete(correlations, idx, 0)
        return correlations

    # Computing the correlations for the set of data per source
    def compute_correlations_per_source(self, idx):
        # First 2 columns are the file names
        n_var = self.data.values[::, 2:].shape[1]
        n_ref = len(self.list_ref)
        correlations = np.zeros((n_ref, n_var, 4))
        for ii in range(n_ref):
            for jj in range(n_var):
                idx_s = np.where(self.data.values[::, 0] == self.list_ref[ii])
                p, s, t, pd = compute_1dcorrelations(
                    self.data.values[idx_s, jj + 2].T, self.data.values[idx_s, idx + 2].T
                    )
                correlations[ii, jj, 0], correlations[ii, jj, 1], correlations[ii, jj, 2], correlations[ii, jj, 3] = (
                    p, s, t, pd
                    )
        correlations = np.delete(correlations, idx, 1)
        return correlations

    # Storing correlations as csv
    def save_correlations(self):
        # Storing the computed correlations as csv files
        list_measures = self.get_list_measures_dataframe()
        list_measures.insert(0, " ")

        dst_dir = self.db_folder.joinpath(self.db_folder.name + "_ifas_pearson.csv")
        np.savetxt(str(dst_dir), np.array(list_measures).reshape(1, -1), delimiter=",", fmt="%s")
        with open(str(dst_dir), "ab") as f:
            np.savetxt(
                f, np.hstack((np.array(list_measures[1:]).reshape(-1, 1), self.pearson)), delimiter=",", fmt="%s"
                )

        dst_dir = self.db_folder.joinpath(self.db_folder.name + "_ifas_spearman.csv")
        np.savetxt(str(dst_dir), np.array(list_measures).reshape(1, -1), delimiter=",", fmt="%s")
        with open(str(dst_dir), "ab") as f:
            np.savetxt(
                f, np.hstack((np.array(list_measures[1:]).reshape(-1, 1), self.spearman)), delimiter=",", fmt="%s"
                )

        dst_dir = self.db_folder.joinpath(self.db_folder.name + "_ifas_tau.csv")
        np.savetxt(str(dst_dir), np.array(list_measures).reshape(1, -1), delimiter=",", fmt="%s")
        with open(str(dst_dir), "ab") as f:
            np.savetxt(
                f, np.hstack((np.array(list_measures[1:]).reshape(-1, 1), self.tau)), delimiter=",", fmt="%s"
                )

        dst_dir = self.db_folder.joinpath(self.db_folder.name + "_ifas_dist_corr.csv")
        np.savetxt(str(dst_dir), np.array(list_measures).reshape(1, -1), delimiter=",", fmt="%s")
        with open(str(dst_dir), "ab") as f:
            np.savetxt(
                f, np.hstack((np.array(list_measures[1:]).reshape(-1, 1), self.dist_corr)), delimiter=",", fmt="%s"
                )

    # Getting the highest correlations from the correlation matrix
    def get_highest_corr(self):
        dist_corr = np.triu(self.dist_corr)
        # correlations with itself are not considered for the ranking
        for ii in range(dist_corr.shape[0]):
            dist_corr[ii, ii] = 0

        dist_corr_arr = np.ravel(dist_corr)
        dist_corr_arr = np.abs(dist_corr_arr)
        idx = np.argsort(dist_corr_arr)[::-1]
        idx = np.unravel_index(idx[0:5], self.dist_corr.shape)
        vals = dist_corr[idx]

        return vals, idx

    # Getting the list of measures in the pandas data frame
    def get_list_measures_dataframe(self):
        # First 2 columns are the file names
        return list(self.data.columns)[2:]

    # Getting the matrix of values from the data frame
    def get_data(self):
        # First 2 columns are the file names
        return self.data.values[::, 2:]

    # This function optimize the model for the database
    def optimize_model(self, model, target, ini_par):
        self.reg_model = RegressionModel(model_type=model, ini_par=ini_par)
        data = self.get_data().astype("float64")
        self.model_par = self.reg_model.optimize_over_data(data, data[::, target])
        self.target = target

    # This function estimate values using the model for the database
    def estimate_using_model(self):
        data = self.get_data().astype("float64")
        y_est = self.reg_model.evaluate_over_data(self.model_par, data)
        return y_est


class ProcessHandler(object):
    """
    Class object to initialize the processor
    This object is used to spawn process on the database and the selected measures
    """
    def __init__(self, db, measures):
        self.main_window = tk.Tk()
        self.main_window.geometry("500x100+700+500") # WidthxHeight+xoffset+yoffset
        self.db = db
        self.measures = measures
        self.main_window.title("Progress")
        self.label = tk.Label(self.main_window, text="Your data is being processed!", font=36)
        self.label_ref = tk.Label(self.main_window, text="Reference 0 %", font=20)
        self.progress_bar_ref = ttk.Progressbar(
            master=self.main_window, orient=tk.HORIZONTAL, mode="determinate", maximum=100, value=0, length=370
            )
        self.label_tst = tk.Label(self.main_window, text="Test 0 %", font=20)
        self.progress_bar_tst = ttk.Progressbar(
            master=self.main_window, orient=tk.HORIZONTAL, mode="determinate", maximum=100, value=0, length=370
            )
        self.label_ref.grid(row=0, column=0)
        self.progress_bar_ref.grid(row=0, column=1)
        self.label_tst.grid(row=1, column=0)
        self.progress_bar_tst.grid(row=1, column=1)
        self.label.grid(row=2, column=0, columnspan=2, pady=15)

    def update_progress_bar_value(self, value1=1, value2=1):
        # Keep updating the master object to redraw the progress bar
        self.progress_bar_ref["value"] = value1
        self.progress_bar_tst["value"] = value2
        self.label_ref.configure(text="Reference " + str(int(value1)) + " %")
        self.label_tst.configure(text="Test " + str(int(value2)) + " %")
        self.main_window.update()

    # Function in charge of processing the data and updating the progress bars
    def process_data(self):
        with open(tempfile.gettempdir() + r"\temp_ifas_csv", "w") as f:
            # one liner to get the names of the functions
            func_names = [ii_meas[0] for ii_meas in self.measures]
            # Adding the column names to the csv file
            print("ref", "tst", *func_names, sep=",", file=f)
        total_ref = len(self.db.list_ref)
        cnt1 = 0
        with open(tempfile.gettempdir() + r"\temp_ifas_csv", "a") as f:
            for ii in self.db.list_ref:
                total_tst = len(self.db.dict_tst[ii])
                cnt2 = 0
                ref_img = cv2.imread(str(self.db.db_folder.joinpath(ii, ii + ".png")))
                for jj in self.db.dict_tst[ii]:
                    tst_img = cv2.imread(str(self.db.db_folder.joinpath(ii, jj)))
                    vals = []
                    val1 = 100. * (cnt1) / total_ref
                    val2 = 100. * (cnt2) / total_tst
                    self.update_progress_bar_value(value1=val1, value2=val2)
                    for kk in range(len(self.measures)):
                        # If any image is not read then return a nan instead
                        if ref_img is None or tst_img is None:
                            val = np.nan
                            self.db.logger.print(level="ERROR", message="ref or tst image is None ")
                        else:
                            try:
                                val = self.measures[kk][1](ref_img, tst_img)
                            except Exception as error:
                                self.db.logger.print(level="ERROR", message="Processing database " + repr(error))
                                val = np.nan

                        vals.append(val)

                    # Adding the computed data to the csv file
                    print(ii, jj, *vals, sep=",", file=f)

                    msg = ("Computing " + ii + ", " + jj + ", " + ", ".join([str(val) for val in vals]))
                    self.db.logger.print(level="DEBUG", message=msg)
                    cnt2 += 1

                cnt1 += 1
                val1 = 100. * (cnt1) / total_ref
                val2 = 100. * (cnt2) / total_tst
                self.update_progress_bar_value(value1=val1, value2=val2)

        src_dir = tempfile.gettempdir() + r"\temp_ifas_csv"
        dst_dir = str(self.db.db_folder.joinpath(self.db.db_folder.name + "_ifas_ouput.csv"))
        shutil.copy(src_dir, dst_dir)
        self.label.configure(text="You can close this window!")
        self.main_window.update()
        self.db.logger.print(level="INFO", message="Processing finished!")
        # Comment line below if you want the window close automatically
        self.main_window.mainloop()
