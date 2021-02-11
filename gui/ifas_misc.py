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

import tkinter as tk
from datetime import datetime
import tkinter.ttk as ttk
import numpy as np
import pathlib
import tempfile
import cv2

# Creates iFas logo image
def logo_image(size_in):
    # Create a black image
    img = np.zeros((int(size_in[1] - 50), int(size_in[0] / 2 - 25), 3), np.uint8)

    # Write some Text
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_color = (0,255,255)
    font_thickness = 2

    (label_width, label_height), __ = cv2.getTextSize(
        'iFAS: Image fidelity assessment software', font, font_scale, font_thickness
        )
    pos = (int(img.shape[1] / 2 - label_width / 2), int(img.shape[0] / 2 - label_height / 2))

    cv2.putText(
        img, 'iFAS: Image fidelity assessment software', pos, font, font_scale, font_color, 
        font_thickness
        )
    return img


class Logging(object):
    """
    Class object to initialize the logging functionality
    """
    def __init__(self, file_name=None):
        if file_name is None:
            file_name = tempfile.gettempdir() + r'\ifas_log.log'
        self.file_path = pathlib.Path(file_name)
        if self.file_path.is_file():
            with open(str(self.file_path), 'a') as f:    
                print(self.get_time(), 'Logging started', sep=', ', file=f)
        else:
            with open(str(self.file_path), 'w') as f:    
                print(self.get_time(), 'Logging started', sep=', ', file=f)

    def get_time(self):
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")

    def print(self, level, message):
        with open(str(self.file_path), 'a') as f:    
                print(self.get_time(), level, message, sep=', ', file=f)


class ProgressBar(object):
    """
    Class object to initialize the main window of a progressbar
    """
    def __init__(self, title='Default'):
        self.main_window = tk.Tk()
        self.main_window.title(title)
        self.progress_bar = ttk.Progressbar(
            master=self.main_window, orient=tk.HORIZONTAL, mode='determinate', maximum=100, value=0
            )
        self.progress_bar.pack(fill=tk.BOTH, expand=1)

        self.main_window.mainloop()

    def update_value(self, value=1):
        # Keep updating the master object to redraw the progress bar
        self.progress_bar['value'] += value
        self.main_window.update()


class Radiobutton(object):
    """
    Class object to initialize the main window of a popup with radiobuttons
    """
    def __init__(self, list_dist=[''], title='Default'):
        self.main_window = tk.Tk()
        self.main_window.title(title)
        self.var = tk.StringVar()
        self.var.set(list_dist[0])

        for item in list_dist:
            self.button = tk.Radiobutton(
                self.main_window, text=item, variable=self.var, value=item, command=self.selection
                )
            self.button.pack(anchor=tk.N)

        self.main_window.mainloop()

    def selection(self):
        return self.var.get()
