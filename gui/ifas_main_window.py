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

from PIL import ImageTk, Image
import tkinter as tk
import tkinter.ttk as ttk
import tkinter.messagebox
import tkinter.filedialog
from tkinter import font
import numpy as np
import importlib
import inspect
import pathlib
import shutil
import pandas
import time
import glob
import cv2
import csv
import sys
import os

from image_processing import add_distortions
from image_processing.image_database import ImgDatabase
import ifas_misc

PATH_FILE = pathlib.Path(__file__).parent.absolute()
PATH_MEASURES = PATH_FILE.parents[0].joinpath('fidelity_measures')
LIST_VALID_EXTENSIONS = ['.png', '.jpg', '.bmp']
temp_list = inspect.getmembers(add_distortions, inspect.isfunction)
LIST_DIST = []
for ii in range(len(temp_list)):
    LIST_DIST.append(temp_list[ii][0])

class AppIFAS(object):
    """
    Class object to initialize the main window of iFASS application
    """
    def __init__(self):
        self.win = tk.Tk()
        self.win.title('iFAS: image fidelity assessment software')
        self.win.configure(background='black')
        self.win.geometry("1800x900+70+50")
        self.size = (1700, 500)

        self.frame_imgs = tk.LabelFrame(
            master=self.win, width=self.size[0], height=self.size[1], bg="black", fg="white", font=18, 
            text='Image view'
            )
        self.frame_imgs.place(x=50, y=25)
        # self.frame_imgs.pack(fill=tk.BOTH, side=tk.TOP)
        self.canvas_left = tk.Canvas(master=self.frame_imgs, width=self.size[0] / 2 - 25, height=self.size[1] - 50)
        self.canvas_right = tk.Canvas(master=self.frame_imgs, width=self.size[0] / 2 - 25, height=self.size[1] - 50)
        self.canvas_left.place(x=15, y=10)
        self.canvas_right.place(x=self.size[0] / 2 + 5, y=10)

        self.frame_ctrl = tk.Label(
            master=self.win, width=self.size[0], height=self.size[1] - 200, bg="black", fg="white"
            )
        self.frame_ctrl.place(x=50, y=self.size[1] + 50)
        self.set_control_frames()
        self.set_buttons_data()
        self.set_buttons_measures()

        self.dummy_img = ifas_misc.logo_image(self.size)
        self.disp_imgs()

        # Creating the progress bar
        self.frame_progess = tk.Frame(master=self.win, bg="black")
        self.frame_progess.place(anchor=tk.N, relx=0.5, rely=0.955, relheight=0.035, relwidth=0.5)
        self.progress_bar = ttk.Progressbar(
            master=self.frame_progess, orient=tk.HORIZONTAL, mode='determinate', maximum=100, value=0
            )
        self.progress_bar.place(relx=0.0, rely=0.0, relheight=1, relwidth=1)

        self.win.resizable(False, False)
        self.win.mainloop()

    # Creating the space for the buttons. They will be organized in sections 1 x 5 bellow the images
    def set_control_frames(self):
        self.frame_data = tk.LabelFrame(
            master=self.frame_ctrl, width=int(self.size[0] / 5) - 10, height=self.size[1] - 200, bg="black", fg="white", 
            font=18, text='Data'
            )
        self.frame_measures = tk.LabelFrame(
            master=self.frame_ctrl, width=int(self.size[0] / 5) - 10, height=self.size[1] - 200, bg="black", fg="white", 
            font=18, text='Measures'
            )
        self.frame_plots = tk.LabelFrame(
            master=self.frame_ctrl, width=int(self.size[0] / 5) - 10, height=self.size[1] - 200, bg="black", fg="white", 
            font=18, text='Plotting'
            )
        self.frame_stats = tk.LabelFrame(
            master=self.frame_ctrl, width=int(self.size[0] / 5) - 10, height=self.size[1] - 200, bg="black", fg="white", 
            font=18, text='Statistics'
            )
        self.frame_models = tk.LabelFrame(
            master=self.frame_ctrl, width=int(self.size[0] / 5) - 10, height=self.size[1] - 200, bg="black", fg="white", 
            font=18, text='Modeling'
            )
        self.frame_data.place(x=0, y=0)
        self.frame_measures.place(x=int(self.size[0] / 5), y=0)
        self.frame_plots.place(x=int(2 * self.size[0] / 5), y=0)
        self.frame_stats.place(x=int(3 * self.size[0] / 5), y=0)
        self.frame_models.place(x=int(4 * self.size[0] / 5), y=0)

    # Setting the controls
    def set_buttons_data(self):
        # Setting controls for data frame
        self.button_create = tk.Button(
            master=self.frame_data, bg="black", fg="white", activebackground='gray', font=18, text='Create', 
            command=self.create_data)
        self.button_process = tk.Button(
            master=self.frame_data, bg="black", fg="white", activebackground='gray', font=18, text='Process', 
            command=self.process_data)
        self.button_load = tk.Button(
            master=self.frame_data, bg="black", fg="white", activebackground='gray', font=18, text='Load', 
            command=self.load_data)
        self.button_create.place(anchor=tk.N, relx=0.5, rely=0.01, relheight=0.3, relwidth=0.8)
        self.button_process.place(anchor=tk.N, relx=0.5, rely=0.35, relheight=0.3, relwidth=0.8)
        self.button_load.place(anchor=tk.N, relx=0.5, rely=0.69, relheight=0.3, relwidth=0.8)

    # Setting the controls
    def set_buttons_measures(self):    
        # Guetting the list of python files and functions in each file
        list_files = list(PATH_MEASURES.glob('**/fidelity_*.py'))
        modules = {}
        for ii in range(len(list_files)):
            current_module = importlib.import_module(str(list_files[ii].stem))
            mod_members = inspect.getmembers(current_module, inspect.isfunction)
            modules[list_files[ii].stem] = []
            for jj in range(len(mod_members)):
                # Getting the function string
                modules[list_files[ii].stem].append(mod_members[jj][0])
            modules[list_files[ii].stem].sort()

        # Setting controls for measures
        self.listbox = tk.Listbox(master=self.frame_measures, selectmode=tk.MULTIPLE, selectbackground='gray')
        self.listbox.place(relx=0.0, rely=0.0, relheight=1, relwidth=0.95)
        self.listbox.config(font=font.Font(size=18))

        # Here go into the measures and add them to the listbox
        count = 0
        for ii in modules:
            self.listbox.insert(tk.END, ii)
            self.listbox.itemconfig(count, bg='cyan')  # Different color for the main python file name
            count += 1
            # Adding each function within the python file
            for jj in range(len(modules[ii])):
                self.listbox.insert(tk.END, modules[ii][jj])
                count += 1

        self.scrollbar = tk.Scrollbar(master=self.frame_measures, orient=tk.VERTICAL)
        self.scrollbar.place(relx=0.95, rely=0.00, relheight=1, relwidth=0.05)

        # attach listbox to scrollbar
        self.listbox.config(yscrollcommand=self.scrollbar.set)
        self.scrollbar.config(command=self.listbox.yview)
        self.modules = modules
        
    # Setting the controls
    def set_buttons_plots(self):
        # Setting controls for plotting
        pass

    # Setting the controls
    def set_buttons_stats(self):
        # Setting controls for stats
        pass

    # Setting the controls
    def set_buttons_models(self):
        # Setting controls for modeling
        pass

    # Creates a new database and makes it ready to be processed
    def create_data(self):
        # Getting the selected directory
        folder_selected = tk.filedialog.askdirectory(initialdir="/", title="Select source directory", master=self.win)
        folder_path = pathlib.Path(folder_selected)

        # one liner for getting only the list of valid extensions png, jpg, bmp
        list_source_imgs = sorted(filter(lambda path: path.suffix in LIST_VALID_EXTENSIONS, folder_path.glob('*')))
        if len(list_source_imgs) == 0:
            tk.messagebox.showerror("Error", "Directory is empty!", master=self.win)
            return

        name_database = tk.filedialog.asksaveasfilename(initialdir="/", title="Name your database", master=self.win)
        if os.path.exists(name_database) or name_database == '':
            tk.messagebox.showerror("Error", "Incorrect directory!", master=self.win)
            return

        # Reading the settings for the type of distortion and the distortion
        with open(str(folder_path.joinpath('db_settings'))) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            settings = []
            for row in csv_reader:
                settings.append(row)
            distortion = settings[0][0]
            dis_levels = list(map(float, settings[1]))
            if distortion not in LIST_DIST:
                tk.messagebox.showerror("Error", "Incorrect distortion selection!", master=self.win)
                return

        # Creating the individual directories per source
        os.mkdir(name_database)
        increment = 100. / len(list_source_imgs)
        for ii in range(len(list_source_imgs)):
            current_path = pathlib.Path(name_database).joinpath(list_source_imgs[ii].stem)
            os.mkdir(str(current_path))
            # Copying the source images to their individual folders as .png
            current_source = str(current_path.joinpath(current_path.stem + '.png'))
            img = cv2.imread(str(list_source_imgs[ii]))
            cv2.imwrite(current_source, img)
            # Generating the distorted images
            for jj in range(len(dis_levels)):
                out_fol = str(
                    current_path.joinpath(
                        current_path.stem + '_' + distortion + str(dis_levels[jj]) + '.png'
                        ))
                add_distortions.brightness(current_source, lvl=dis_levels[jj], out_folder=out_fol)
            
            self.progress_bar['value'] += increment
            self.frame_progess.update()

        tk.messagebox.showinfo("Information", "DATA CREATED at \n" + name_database, master=self.win)
        self.progress_bar['value'] = 0
        self.frame_progess.update()
    
    # Process the most recently loaded/created database using the selected measures
    def process_data(self):
        tk.messagebox.showinfo("Information", "DATA PROCESSED", master=self.win)

    # Load existing data into memory -> main_folder_name_ifas file
    def load_data(self):
        folder_selected = tk.filedialog.askdirectory(initialdir="/", title="Select database directory", master=self.win)
        db = ImgDatabase(folder_selected)
        db.print()
        tk.messagebox.showinfo("Information", "DATA LOADED", master=self.win)

    # Display 2 given images, if not given iFAS logo is displayed
    def disp_imgs(self, img_left=None, img_right=None):
        # Reading the images
        if img_left is None or img_right is None:
            img_left = self.dummy_img
            img_right = self.dummy_img
        else:
            img_left = cv2.imread(img_left)
            img_right = cv2.imread(img_right)

        # Setting images in the canvas
        img_left = Image.fromarray(img_left)
        self.imgtk_left = ImageTk.PhotoImage(img_left) 
        img_right = Image.fromarray(img_right)
        self.imgtk_right = ImageTk.PhotoImage(img_right)
        self.canvas_left.create_image(0, 0, anchor=tk.NW, image=self.imgtk_left)
        self.canvas_right.create_image(0, 0, anchor=tk.NW, image=self.imgtk_right)
