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

from processing.ifas_stats import compute_1dcorrelations
from gui import ifas_misc

class ImgDatabase(object):
    """
    Class object to initialize the database of images
    """
    def __init__(self, data_folder):
        self.db_folder = pathlib.Path(data_folder)
        self.get_img_paths()

    def set_test_measures(self, measures):
        self.measures = measures

    def set_logger(self, logger):
        self.logger = logger

    def get_img_paths(self):
        flag = False
        # Removing first row -> It is the root folder
        list_folders = list(self.db_folder.glob('**'))[1:]
        # one liner for getting only the file name of the reference images
        self.list_ref = [str(ii_folder).split(os.sep)[-1] for ii_folder in list_folders]
        self.dict_tst = {}
        for ii in range(len(self.list_ref)):
            list_imgs_crt = list(list_folders[ii].glob('*.png'))

            # If any of the selected sub-folders is empty, it is a good indication for wrong selected folder
            if len(list_imgs_crt) == 0:
                flag = True

            # one liner for getting only the file name of the test images for the current folder
            self.dict_tst[self.list_ref[ii]] = [str(ii_img).split(os.sep)[-1] for ii_img in list_imgs_crt]
            # removing the source image from the list of test images
            if self.list_ref[ii] + '.png' in self.dict_tst[self.list_ref[ii]]:
                self.dict_tst[self.list_ref[ii]].remove(self.list_ref[ii] + '.png')
            else:
                flag = True
        if flag:
            self.dict_tst = None
            self.list_ref = None

    # Place holder for loading the pandas data frame
    def get_csv(self):
        if self.db_folder.joinpath(self.db_folder.name + '_ifas_ouput.csv').is_file():
            self.data = pd.read_csv(str(self.db_folder.joinpath(self.db_folder.name + '_ifas_ouput.csv')))
        else:
            self.data = None

    # Place holder for executing the process using the database
    def compute_measures(self):
        psh = ProcessHandler(self, self.measures)
        psh.process_data()

    # Computing the correlations for the set of data
    def compute_correlations(self):
        # First 2 columns are the file names
        p, s, t, pd = compute_1dcorrelations(self.data.values[::, 2:])
        self.pearson = p
        self.spearman = s
        self.tau = t
        self.dist_corr = pd
        self.save_correlations()
        ifas_misc.heat_map(p, s, pd, t)

    # Storing correlations as csv
    def save_correlations(self):
        # Storing the computed correlations as csv files
        list_measures = self.get_list_measures_dataframe()
        list_measures.insert(0, ' ')

        dst_dir = self.db_folder.joinpath(self.db_folder.name + '_ifas_pearson.csv')
        np.savetxt(str(dst_dir), np.array(list_measures).reshape(1, -1), delimiter=',', fmt='%s')
        with open(str(dst_dir), "ab") as f:
            np.savetxt(
                f, np.hstack((np.array(list_measures[1:]).reshape(-1, 1), self.pearson)), delimiter=',', fmt='%s'
                )

        dst_dir = self.db_folder.joinpath(self.db_folder.name + '_ifas_spearman.csv')
        np.savetxt(str(dst_dir), np.array(list_measures).reshape(1, -1), delimiter=',', fmt='%s')
        with open(str(dst_dir), "ab") as f:
            np.savetxt(
                f, np.hstack((np.array(list_measures[1:]).reshape(-1, 1), self.spearman)), delimiter=',', fmt='%s'
                )

        dst_dir = self.db_folder.joinpath(self.db_folder.name + '_ifas_tau.csv')
        np.savetxt(str(dst_dir), np.array(list_measures).reshape(1, -1), delimiter=',', fmt='%s')
        with open(str(dst_dir), "ab") as f:
            np.savetxt(
                f, np.hstack((np.array(list_measures[1:]).reshape(-1, 1), self.tau)), delimiter=',', fmt='%s'
                )

        dst_dir = self.db_folder.joinpath(self.db_folder.name + '_ifas_dist_corr.csv')
        np.savetxt(str(dst_dir), np.array(list_measures).reshape(1, -1), delimiter=',', fmt='%s')
        with open(str(dst_dir), "ab") as f:
            np.savetxt(
                f, np.hstack((np.array(list_measures[1:]).reshape(-1, 1), self.dist_corr)), delimiter=',', fmt='%s'
                )

    # Getting the highest correlations from the correlation matrix
    def get_highest_corr(self):
        dist_corr = np.triu(self.dist_corr)
        for ii in range(dist_corr.shape[0]):
            dist_corr[ii, ii] = 0

        dist_corr_arr = np.ravel(dist_corr)
        dist_corr_arr = np.abs(dist_corr_arr)
        idx = np.argsort(dist_corr_arr)[::-1]
        idx = np.unravel_index(idx[0:5], self.dist_corr.shape)
        vals = dist_corr[idx]

        return vals, idx

    def get_list_measures_dataframe(self):
        return list(self.data.columns)[2:]


class ProcessHandler(object):
    """
    Class object to initialize the processor
    """
    def __init__(self, db, measures):
        self.main_window = tk.Tk()
        self.main_window.geometry("500x100+700+500") # WidthxHeight+xoffset+yoffset
        self.db = db
        self.measures = measures
        self.main_window.title('Progress')
        self.label = tk.Label(self.main_window, text='Your data is being processed!', font=36)
        self.label_ref = tk.Label(self.main_window, text='Reference 0 %', font=20)
        self.progress_bar_ref = ttk.Progressbar(
            master=self.main_window, orient=tk.HORIZONTAL, mode='determinate', maximum=100, value=0, length=370
            )
        self.label_tst = tk.Label(self.main_window, text='Test 0 %', font=20)
        self.progress_bar_tst = ttk.Progressbar(
            master=self.main_window, orient=tk.HORIZONTAL, mode='determinate', maximum=100, value=0, length=370
            )
        self.label_ref.grid(row=0, column=0)
        self.progress_bar_ref.grid(row=0, column=1)
        self.label_tst.grid(row=1, column=0)
        self.progress_bar_tst.grid(row=1, column=1)
        self.label.grid(row=2, column=0, columnspan=2, pady=15)

    def update_progress_bar_value(self, value1=1, value2=1):
        # Keep updating the master object to redraw the progress bar
        self.progress_bar_ref['value'] = value1
        self.progress_bar_tst['value'] = value2
        self.label_ref.configure(text='Reference ' + str(int(value1)) + ' %')
        self.label_tst.configure(text='Test ' + str(int(value2)) + ' %')
        self.main_window.update()

    def process_data(self, cont=False):
        # TODO add here the behaviour when continuening is allow in case of crashes 
        if cont is False:
            with open(tempfile.gettempdir() + r'\temp_ifas_csv', 'w') as f:
                # one liner to get the names of the functions
                func_names = [ii_meas[0] for ii_meas in self.measures]
                print('ref', 'tst', *func_names, sep=',', file=f)
        total_ref = len(self.db.list_ref)
        cnt1 = 0
        for ii in self.db.list_ref:
            total_tst = len(self.db.dict_tst[ii])
            cnt2 = 0
            ref_img = cv2.imread(str(self.db.db_folder.joinpath(ii, ii + '.png')))
            for jj in self.db.dict_tst[ii]:
                tst_img = cv2.imread(str(self.db.db_folder.joinpath(ii, jj)))
                vals = []
                for kk in range(len(self.measures)):
                    val = self.measures[kk][1](ref_img, tst_img)
                    vals.append(val)

                with open(tempfile.gettempdir() + r'\temp_ifas_csv', 'a') as f:    
                    print(ii, jj, *vals, sep=',', file=f)

                # TODO probably here logging
                msg = ('Computing ' + ii + ', ' + jj + ', ' + ', '.join([str(val) for val in vals]))
                self.db.logger.print(level='INFO', message=msg)
                val1 = 100. * (cnt1 + 1) / total_ref
                val2 = 100. * (cnt2 + 1) / total_tst
                self.update_progress_bar_value(value1=val1, value2=val2)
                cnt2 += 1
            cnt1 += 1

        src_dir = tempfile.gettempdir() + r'\temp_ifas_csv'
        dst_dir = str(self.db.db_folder.joinpath(self.db.db_folder.name + '_ifas_ouput.csv'))
        shutil.copy(src_dir, dst_dir)
        tk.messagebox.showinfo("Information", "DATA PROCESSED")
        self.label.configure(text='You can close this window!')
        self.main_window.update()
        self.db.logger.print(level='INFO', message='Processing finished!')
        self.main_window.mainloop()
