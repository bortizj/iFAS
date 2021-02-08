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


import pathlib
import os
import numpy as np
import cv2

class ImgDatabase(object):
    """
    Class object to initialize the database of images
    """
    def __init__(self, data_folder):
        self.db_folder = pathlib.Path(data_folder)
        self.get_img_paths()

    def get_img_paths(self):
        # Removing first row -> It is the root folder
        list_folders = list(self.db_folder.glob('**'))[1:]
        # one liner for getting only the file name of the reference images
        self.list_ref = [str(ii_folder).split(os.sep)[-1] for ii_folder in list_folders]
        self.dict_tst = {}
        for ii in range(len(self.list_ref)):
            list_imgs_crt = list(list_folders[ii].glob('*.png'))
            # one liner for getting only the file name of the test images for the current folder
            self.dict_tst[self.list_ref[ii]] = [str(ii_img).split(os.sep)[-1] for ii_img in list_imgs_crt]
            # removing the source image from the list of test images
            self.dict_tst[self.list_ref[ii]].remove(self.list_ref[ii] + '.png')

    # Place holder for loading the pandas data frame
    def get_csv(self):
        pass

    # Place holder for executing the process using the database
    def compute_measures(self, measures):
        fidelity_process = ProcessHandler(self, measures)
        print("WAIT")

    def print(self):
        for ii in self.list_ref:
            print(ii)
            for jj in self.dict_tst[ii]:
                print('    ', jj)

# Place holder for the class in charge of the processing
class ProcessHandler(object):
    """
    Class object to initialize the processor
    """
    def __init__(self, db, measures):
        pass


if __name__ == "__main__":
    db = ImgDatabase('D:\gitProjects\iFAS\example_data\db')
    db.compute_measures([])
