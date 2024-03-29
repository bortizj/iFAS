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

from PIL import ImageTk, Image
import tkinter as tk
import tkinter.ttk as ttk
import tkinter.messagebox
from tkinter import scrolledtext
import tkinter.filedialog
from tkinter import font
import numpy as np
import importlib
import traceback
import threading
import inspect
import pathlib
import copy
import cv2
import csv
import os

from iFAS.processing import add_distortions
from iFAS.processing.image_database import ImgDatabase
from iFAS.gui import ifas_misc, ifas_plotting

# Global variables of paths and available methods
PATH_FILE = pathlib.Path(__file__).parent.absolute()
PATH_MEASURES = PATH_FILE.parents[0].joinpath("fidelity_measures")
LIST_VALID_EXTENSIONS = [".png", ".jpg", ".bmp"]
CHOICES_MODEL = ["linear", "quadratic", "cubic", "exponential", "logistic", "complementary_error"]
LENGHT_PARAMETERS = {"linear": 2, "quadratic": 3, "cubic": 4, "exponential": 3, "logistic": 3, "complementary_error": 2}
DEFAULT_PARAMETERS = {
    "linear": "-1,1", "quadratic": "-1,0.5,1", "cubic": "-1,0.5,-0.5,1", "exponential": "-1,0.5,1", 
    "logistic": "-1,0.5,1", "complementary_error": "0.5,-0.5"
    }
# Oneliner to get all the functions in the add_distortion script
LIST_DIST = [fun_name[0] for fun_name in inspect.getmembers(add_distortions, inspect.isfunction)]

class AppIFAS(object):
    """
    Class object to initialize the main window of iFAS application
    There are no input parameters
    """
    def __init__(self):
        # Initializing iFAS main window
        self.win = tk.Tk()
        self.win.title("iFAS: image fidelity assessment software")
        self.win.configure(background="black")
        self.win.geometry("1800x900+70+50")
        self.size = (1700, 500)

        # Variable for the iFAS database
        self.db = None

        # Initializing logging functionality
        self.logger = ifas_misc.Logging()
        self.logger.print(level="DEBUG", message="Ifas logging started!")

        # Frame for displaying the images side by side
        self.frame_imgs = tk.LabelFrame(
            master=self.win, width=self.size[0], height=self.size[1], bg="black", fg="white", font=18, 
            text="Image view"
            )
        self.frame_imgs.place(x=50, y=25)
        self.canvas_left = tk.Canvas(master=self.frame_imgs, width=self.size[0] / 2 - 25, height=self.size[1] - 50)
        self.canvas_right = tk.Canvas(master=self.frame_imgs, width=self.size[0] / 2 - 25, height=self.size[1] - 50)
        self.canvas_left.place(x=15, y=10)
        self.canvas_right.place(x=self.size[0] / 2 + 5, y=10)

        # Setting the function to call when click on the image canvas (ref and tst)
        self.canvas_left.bind("<Button-1>", lambda event: self.click_on_image(canvas="ref"))
        self.canvas_right.bind("<Button-1>", lambda event: self.click_on_image(canvas="tst"))

        # Reading the icons for the image controls (left and right)
        left_arrow = cv2.imread(str(PATH_FILE.joinpath("left.png")), cv2.IMREAD_UNCHANGED)
        left_arrow = cv2.cvtColor(left_arrow, cv2.COLOR_BGRA2RGBA)
        left_arrow = Image.fromarray(left_arrow)
        self.left_arrow = ImageTk.PhotoImage(left_arrow)
        right_arrow = cv2.imread(str(PATH_FILE.joinpath("right.png")), cv2.IMREAD_UNCHANGED)
        right_arrow = cv2.cvtColor(right_arrow, cv2.COLOR_BGRA2RGBA)
        right_arrow = Image.fromarray(right_arrow)
        self.right_arrow = ImageTk.PhotoImage(right_arrow)

        # Setting the buttons for the image controls (left and right)
        self.button_left_ref = tk.Button(
            master=self.frame_imgs, bg="black", fg="white", activebackground="gray", image=self.left_arrow, 
            command=lambda: self.image_changed(button="ref_left"))
        self.button_right_ref = tk.Button(
            master=self.frame_imgs, bg="black", fg="white", activebackground="gray", image=self.right_arrow, 
            command=lambda: self.image_changed(button="ref_right"))
        self.button_left_tst = tk.Button(
            master=self.frame_imgs, bg="black", fg="white", activebackground="gray", image=self.left_arrow, 
            command=lambda: self.image_changed(button="tst_left"))
        self.button_right_tst = tk.Button(
            master=self.frame_imgs, bg="black", fg="white", activebackground="gray", image=self.right_arrow, 
            command=lambda: self.image_changed(button="tst_right"))

        # Placing the buttons for the image controls (left and right)
        self.button_left_ref.place(relx=0.25 - 0.025, rely=-0.02, relheight=0.05, relwidth=0.03)
        self.button_right_ref.place(relx=0.25, rely=-0.02, relheight=0.05, relwidth=0.03)
        self.button_left_tst.place(relx=0.75 - 0.025, rely=-0.02, relheight=0.05, relwidth=0.03)
        self.button_right_tst.place(relx=0.75, rely=-0.02, relheight=0.05, relwidth=0.03)
        self.logger.print(level="DEBUG", message="Place holder image side by side created!")

        # Frame for the user controls
        self.frame_ctrl = tk.Label(
            master=self.win, width=self.size[0], height=self.size[1] - 200, bg="black", fg="white"
            )
        self.frame_ctrl.place(x=50, y=self.size[1] + 50)
        self.set_control_frames()

        # Frames for the buttons in each control frame
        self.set_buttons_data()
        self.set_buttons_measures()
        self.set_buttons_stats()
        self.set_buttons_plots()
        self.set_buttons_models()

        # Creating dummy image to display
        self.dummy_img = ifas_misc.logo_image(self.size)
        self.disp_imgs()

        # Creating the progress bar
        self.frame_progess = tk.Frame(master=self.win, bg="black")
        self.frame_progess.place(anchor=tk.N, relx=0.5, rely=0.955, relheight=0.035, relwidth=0.5)
        self.progress_bar = ttk.Progressbar(
            master=self.frame_progess, orient=tk.HORIZONTAL, mode="determinate", maximum=100, value=0
            )
        self.progress_bar.place(relx=0.0, rely=0.0, relheight=1, relwidth=1)

        # To avoid dealing with window resizing issues
        self.win.resizable(False, False)
        self.win.mainloop()

    # Creating the space for the buttons. They will be organized in a grid 1 x 5 bellow the images
    def set_control_frames(self):
        self.logger.print(level="DEBUG", message="Set set_control_frames started!")
        self.frame_data = tk.LabelFrame(
            master=self.frame_ctrl, width=int(self.size[0] / 5) - 10, height=self.size[1] - 200, bg="black", fg="white", 
            font=18, text="Data"
            )
        self.frame_measures = tk.LabelFrame(
            master=self.frame_ctrl, width=int(self.size[0] / 5) - 10, height=self.size[1] - 200, bg="black", fg="white", 
            font=18, text="Measures"
            )
        self.frame_stats = tk.LabelFrame(
            master=self.frame_ctrl, width=int(self.size[0] / 5) - 10, height=self.size[1] - 200, bg="black", fg="white", 
            font=18, text="Statistics"
            )
        self.frame_plots = tk.LabelFrame(
            master=self.frame_ctrl, width=int(self.size[0] / 5) - 10, height=self.size[1] - 200, bg="black", fg="white", 
            font=18, text="Plotting"
            )
        self.frame_models = tk.LabelFrame(
            master=self.frame_ctrl, width=int(self.size[0] / 5) - 10, height=self.size[1] - 200, bg="black", fg="white", 
            font=18, text="Modeling"
            )

        # Placing the frames for the control buttons
        self.frame_data.place(x=0, y=0)
        self.frame_measures.place(x=int(self.size[0] / 5), y=0)
        self.frame_stats.place(x=int(2 * self.size[0] / 5), y=0)
        self.frame_plots.place(x=int(3 * self.size[0] / 5), y=0)
        self.frame_models.place(x=int(4 * self.size[0] / 5), y=0)
        self.logger.print(level="DEBUG", message="Set set_control_frames finished!")

    # Setting controls for data frame
    def set_buttons_data(self):
        self.logger.print(level="DEBUG", message="Set set_buttons_data started!")
        self.button_create = tk.Button(
            master=self.frame_data, bg="black", fg="white", activebackground="gray", font=18, text="Create", 
            command=self.create_data)
        self.button_process = tk.Button(
            master=self.frame_data, bg="black", fg="white", activebackground="gray", font=18, text="Process", 
            command=self.process_data)
        self.button_load = tk.Button(
            master=self.frame_data, bg="black", fg="white", activebackground="gray", font=18, text="Load", 
            command=lambda: self.load_data(get_data=True))

        # Placing the buttons for data frame
        self.button_create.place(anchor=tk.N, relx=0.5, rely=0.01, relheight=0.3, relwidth=0.8)
        self.button_process.place(anchor=tk.N, relx=0.5, rely=0.35, relheight=0.3, relwidth=0.8)
        self.button_load.place(anchor=tk.N, relx=0.5, rely=0.69, relheight=0.3, relwidth=0.8)
        self.logger.print(level="DEBUG", message="Set set_buttons_data finished!")

    # Guetting the list of python files and functions in each file fidelity_*.py
    def set_buttons_measures(self):    
        self.logger.print(level="DEBUG", message="Set set_buttons_measures started!")
        list_files = list(PATH_MEASURES.glob("**/fidelity_*.py"))
        modules = {} # -> Contains only the string name of the function
        modules_list = {} # -> Contains the callable function
        for ii in range(len(list_files)):
            # Each key is a python script and the values correspond to the functions in the script
            current_module = importlib.import_module(str(list_files[ii].stem))
            mod_members = inspect.getmembers(current_module, inspect.isfunction)
            modules[list_files[ii].stem] = []
            modules_list[list_files[ii].stem] = []
            for jj in range(len(mod_members)):
                modules[list_files[ii].stem].append(mod_members[jj][0])
                modules_list[list_files[ii].stem].append(mod_members[jj])

            # Sorting the modules
            modules_list[list_files[ii].stem].sort()
            modules_list[list_files[ii].stem] = list(zip(
                np.argsort(modules[list_files[ii].stem]).tolist(), modules_list[list_files[ii].stem]
                ))
            # one liner to remove the index from the zip return
            modules_list[list_files[ii].stem] = [mod[1] for mod in modules_list[list_files[ii].stem]]
            modules[list_files[ii].stem].sort()

        # Setting controls for measures
        self.listbox = tk.Listbox(master=self.frame_measures, selectmode=tk.MULTIPLE, selectbackground="gray")
        self.listbox.place(relx=0.0, rely=0.0, relheight=1, relwidth=0.95)
        self.listbox.config(font=font.Font(size=18))

        # Adding the measures to the listbox
        count = 0
        for ii in modules:
            self.listbox.insert(tk.END, ii)
            # Different color for the main python file name -> Allows selecting the whole set
            self.listbox.itemconfig(count, bg="cyan")
            count += 1
            # Adding each function within the python file
            for jj in range(len(modules[ii])):
                self.listbox.insert(tk.END, modules[ii][jj])
                count += 1

        # Makes scrollable the list of fidelity measures
        self.scrollbar = tk.Scrollbar(master=self.frame_measures, orient=tk.VERTICAL)
        self.scrollbar.place(relx=0.95, rely=0.00, relheight=1, relwidth=0.05)

        # attach listbox to scrollbar
        self.listbox.config(yscrollcommand=self.scrollbar.set)
        self.scrollbar.config(command=self.listbox.yview)
        self.modules = modules
        self.modules_list = modules_list
        self.logger.print(level="DEBUG", message="Set set_buttons_measures finished!")

    # Setting controls for plotting
    def set_buttons_plots(self):
        self.logger.print(level="DEBUG", message="Set set_buttons_plots started!")
        self.button_scatter = tk.Button(
            master=self.frame_plots, bg="black", fg="white", activebackground="gray", font=18, text="Scatter", 
            command=self.scatter_plot)
        self.button_bar = tk.Button(
            master=self.frame_plots, bg="black", fg="white", activebackground="gray", font=18, text="Bar", 
            command=self.bar_plot)
        self.button_box = tk.Button(
            master=self.frame_plots, bg="black", fg="white", activebackground="gray", font=18, text="Box", 
            command=self.box_plot)
        self.button_reg = tk.Button(
            master=self.frame_plots, bg="black", fg="white", activebackground="gray", font=18, text="Regression", 
            command=self.reg_plot)

        # Placing the controls for plotting
        self.button_scatter.place(anchor=tk.N, relx=0.5, rely=0.02, relheight=0.215, relwidth=0.8)
        self.button_bar.place(anchor=tk.N, relx=0.5, rely=0.27, relheight=0.215, relwidth=0.8)
        self.button_box.place(anchor=tk.N, relx=0.5, rely=0.52, relheight=0.215, relwidth=0.8)
        self.button_reg.place(anchor=tk.N, relx=0.5, rely=0.77, relheight=0.215, relwidth=0.8)
        self.logger.print(level="DEBUG", message="Set set_buttons_plots finished!")

    # Setting controls for stats
    def set_buttons_stats(self):
        self.logger.print(level="DEBUG", message="Set set_buttons_stats started!")
        self.button_correlations = tk.Button(
            master=self.frame_stats, bg="black", fg="white", activebackground="gray", font=18, text="Correlations", 
            command=self.compute_correlations)
        self.button_correlations.place(anchor=tk.N, relx=0.5, rely=0.01, relheight=0.25, relwidth=0.8)
        self.edit_corr_area = scrolledtext.ScrolledText(master=self.frame_stats, wrap=tk.WORD)
        self.edit_corr_area.place(relx=0.005, rely=0.3, relheight=0.75, relwidth=0.99)
        self.logger.print(level="DEBUG", message="Set set_buttons_stats finished!")

    # Setting controls for modeling
    def set_buttons_models(self):
        self.logger.print(level="DEBUG", message="Set set_buttons_models started!")
        self.label_entry = tk.Label(master=self.frame_models, bg="black", fg="white", font=18, text="Parameters ini:")
        self.label_entry.place(relx=0.1, rely=0.52, relheight=0.215, relwidth=0.4)
        self.par_entry = tk.Entry(self.frame_models, font=18)
        self.par_entry.insert(0, DEFAULT_PARAMETERS["linear"])
        self.par_entry.place(relx=0.55, rely=0.52, relheight=0.215, relwidth=0.4)

        self.label_functions = tk.Label(master=self.frame_models, bg="black", fg="white", font=18, text="Model:")
        self.label_functions.place(relx=0.1, rely=0.02, relheight=0.215, relwidth=0.4)
        self.combobox_functions = ttk.Combobox(self.frame_models, values=CHOICES_MODEL, font=18)
        self.combobox_functions.bind('<<ComboboxSelected>>', self.combobox_functions_changed)
        self.combobox_functions.current(0)
        self.combobox_functions.place(relx=0.55, rely=0.02, relheight=0.215, relwidth=0.4)

        # The selected features is empty at ini
        choices_feat = [""]
        self.label_features = tk.Label(master=self.frame_models, bg="black", fg="white", font=18, text="Target:")
        self.label_features.place(relx=0.1, rely=0.27, relheight=0.215, relwidth=0.4)
        self.combobox_features = ttk.Combobox(self.frame_models, values=choices_feat, font=18)
        self.combobox_features.current(0)
        self.combobox_features.place(relx=0.55, rely=0.27, relheight=0.215, relwidth=0.4)

        self.button_start_model = tk.Button(
            master=self.frame_models, bg="black", fg="white", activebackground="gray", font=18, text="Optimize", 
            command=self.optimize
            )
        self.button_start_model.place(anchor=tk.N, relx=0.5, rely=0.77, relheight=0.215, relwidth=0.8)
        self.logger.print(level="DEBUG", message="Set set_buttons_models finished!")

    # Creates a new database
    def create_data(self):
        self.logger.print(level="INFO", message="Database creation started!")
        # Getting the selected directory
        folder_selected = tk.filedialog.askdirectory(initialdir="/", title="Select source directory", master=self.win)
        folder_path = pathlib.Path(folder_selected)
        self.logger.print(level="INFO", message="Selected folder " + str(folder_path))

        # one liner for getting only the list of valid extensions png, jpg, bmp
        list_source_imgs = sorted(filter(lambda path: path.suffix in LIST_VALID_EXTENSIONS, folder_path.glob("*")))
        if len(list_source_imgs) == 0:
            self.logger.print(level="ERROR", message="Directory is empty " + str(folder_path))
            tk.messagebox.showerror("Error", "Directory is empty!", master=self.win)
            return

        # it is mandatory to have a db_setings file in the selected directory
        # First line name of the type of distortion to be applied to the images
        # Second line comma separated values of distortion levels
        if not folder_path.joinpath("db_settings").is_file():
            self.logger.print(level="ERROR", message="No db_settings file in " + str(folder_path))
            tk.messagebox.showerror("Error", "No db_settings file!", master=self.win)
            return

        name_database = tk.filedialog.asksaveasfilename(initialdir="/", title="Store database", master=self.win)
        if os.path.exists(name_database) or name_database == "":
            self.logger.print(level="ERROR", message="Path exists or empty name " + str(folder_path))
            tk.messagebox.showerror("Error", "Path exists or empty string name!", master=self.win)
            return

        # Reading the settings for the type of distortion and the distortion levels
        with open(str(folder_path.joinpath("db_settings"))) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=",")
            settings = []
            for row in csv_reader:
                settings.append(row)
            distortion = settings[0][0]

            # Avoiding crashes due to errors in setting file -> Particularly the distortion levels must be floats
            try:
                dis_levels = list(map(float, settings[1]))
            except Exception as error:
                str_trace = traceback.format_exc()
                self.logger.print(level="ERROR", message="Reading distortion levels: " + repr(error))
                self.logger.print(level="ERROR", message=str_trace)
                tk.messagebox.showerror(
                    "Error", "Something wrong reading distortion levels! \n" + repr(error), master=self.win
                    )
                return

            # Avoiding crashes due to errors in setting file -> Particularly the distortion must be in LIST_DIST
            if distortion not in LIST_DIST:
                self.logger.print(level="ERROR", message="Incorrect distortion selection in db_settings " + distortion)
                tk.messagebox.showerror(
                    "Error", "Incorrect distortion selection in db_settings: "  + distortion, master=self.win
                    )
                return

        # Creating the individual directories per source
        os.mkdir(name_database)
        increment = 100. / len(list_source_imgs)
        for ii in range(len(list_source_imgs)):
            current_path = pathlib.Path(name_database).joinpath(list_source_imgs[ii].stem)
            os.mkdir(str(current_path))
            # Copying the source images to their individual folders as .png
            current_source = str(current_path.joinpath(current_path.stem + ".png"))
            img = cv2.imread(str(list_source_imgs[ii]))
            if img is None:
                self.logger.print(level="ERROR", message="Source image corrupted " + current_source)
            else:
                cv2.imwrite(current_source, img)
            try:
                # Generating the distorted images
                for jj in range(len(dis_levels)):
                    out_fol = str(
                        current_path.joinpath(
                            current_path.stem + "_" + distortion + str(dis_levels[jj]) + ".png"
                            ))
                    getattr(add_distortions, distortion)(current_source, lvl=dis_levels[jj], out_folder=out_fol)
                    self.logger.print(level="INFO", message=out_fol)
            except Exception as error:
                str_trace = traceback.format_exc()
                self.logger.print(level="ERROR", message="Generating database " + current_source + " " + repr(error))
                self.logger.print(level="ERROR", message=str_trace)
                tk.messagebox.showerror("Error", "Something went wrong! \n" + repr(error), master=self.win)

            self.progress_bar["value"] += increment
            self.frame_progess.update()

        self.logger.print(level="INFO", message="DATA CREATED at \n" + name_database)
        tk.messagebox.showinfo("Information", "DATA CREATED at \n" + name_database, master=self.win)
        self.progress_bar["value"] = 0
        self.frame_progess.update()

    # Prompt to load a database using the selected measures and process the images
    def process_data(self):
        self.logger.print(level="INFO", message="Set process data started!")
        self.db = None
        modules_selected = self.get_seleted_measures()
        if len(modules_selected) == 0:
            self.logger.print(level="ERROR", message="No measures selected ")
            tk.messagebox.showerror("Error", "No measures selected!", master=self.win)
            return

        # Prompt window if database does not exists to select database
        try:
            self.load_data()
            self.disp_imgs()
        except Exception as error:
            str_trace = traceback.format_exc()
            self.logger.print(level="ERROR", message="Generating database " + repr(error))
            self.logger.print(level="ERROR", message=str_trace)
            tk.messagebox.showerror("Error", "Something went wrong! \n" + repr(error), master=self.win)
            self.db = None
            return

        # Database can be empty if something was wrong while loading so getting out of the function
        if self.db is None:
            return

        self.db.set_test_measures(modules_selected)

        self.logger.print(level="INFO", message="Processing started!")
        self.db.set_logger(self.logger)
        # Copying the database to another variable so the process can be attached to it and the object database can be 
        # clear in order to allow another process
        db = copy.copy(self.db)
        newthread = threading.Thread(target=db.compute_measures)
        newthread.start()
        self.db = None

    # Get the list of measures seleted
    def get_seleted_measures(self):
        self.logger.print(level="DEBUG", message="get_seleted_measures started!")
        seleccion = self.listbox.curselection()
        list_selected = []
        modules_selected = []
        for ii in seleccion:
            current = self.listbox.get(ii)
            if current in self.modules:
                list_selected.extend(self.modules[current])
                modules_selected.extend(self.modules_list[current])
            elif current not in list_selected:
                list_selected.append(current)
                for jj in self.modules:
                    if current in self.modules[jj]:
                        modules_selected.append(self.modules_list[jj][self.modules[jj].index(current)])

        self.logger.print(level="DEBUG", message="get_seleted_measures finished!")

        return modules_selected

    # Load existing data into memory -> main_folder_name_ifas file
    def load_data(self, get_data=False):
        self.logger.print(level="INFO", message="Load data started!")
        folder_selected = tk.filedialog.askdirectory(initialdir="/", title="Select database directory", master=self.win)
        self.db = ImgDatabase(folder_selected)

        if self.db.list_ref is None:
            self.db = None
            self.logger.print(level="ERROR", message="Wrong database selection: empty reference list or folder")
            tk.messagebox.showerror("Error", "Wrong database selection!", master=self.win)
            return

        # Reads the csv file if requested, eg, for load functionality
        if get_data:
            self.db.get_csv()
            if self.db.data is None:
                self.logger.print(level="WARNING", message="No csv file in database ")
                tk.messagebox.showerror("Warning", "No csv file in database please reprocess!", master=self.win)
                self.db = None
                return
            else:
                # Updating the list of available features in the regression frame
                self.combobox_features["values"] = self.db.get_list_measures_dataframe()
                self.combobox_features.current(0)
                self.logger.print(level="DEBUG", message=self.db.data.head(5))
                self.logger.print(level="INFO", message="Data loaded finished!")
                tk.messagebox.showinfo("Information", "DATA LOADED", master=self.win)

        # Displays the first source and test images
        self.ref_img_idx = 0
        self.tst_img_idx = 0
        crt_ref = self.db.db_folder.joinpath(
            self.db.list_ref[self.ref_img_idx], self.db.list_ref[self.ref_img_idx] + ".png"
            )
        crt_tst = self.db.db_folder.joinpath(
            self.db.list_ref[self.ref_img_idx], self.db.dict_tst[self.db.list_ref[self.ref_img_idx]][self.tst_img_idx]
            )
        self.disp_imgs(img_left=str(crt_ref), img_right=str(crt_tst))

    # Computes correlation on the exiting measures
    def compute_correlations(self):
        self.logger.print(level="INFO", message="Computing correlations started ")
        if not self.verify_db():
            return

        self.db.compute_correlations()
        self.show_corr_summary()
        self.logger.print(level="INFO", message="Correlations computed finished ")

    # Shows the correlation summary on the exiting measures
    def show_corr_summary(self):
        self.logger.print(level="DEBUG", message="show_corr_summary started")
        vals, idx = self.db.get_highest_corr()
        self.edit_corr_area.insert(tk.INSERT, "----Distance correlation summary----" + "\n")
        self.edit_corr_area.insert(tk.INSERT, "Top 5 correlations:"+ "\n")
        # Printing the top 5 correlation values
        for ii in range(len(vals)):
            self.edit_corr_area.insert(
                tk.INSERT, "N" + str(ii + 1) + " -> (" + str(idx[0][ii]) + "," + str(idx[1][ii]) + "): " + str(vals[ii]) 
                + "\n"
                )
        # Printing the measure names
        self.edit_corr_area.insert(tk.INSERT, "Legend of measures:"+ "\n")
        measures_name = self.db.get_list_measures_dataframe()
        for ii in range(len(measures_name)):
            self.edit_corr_area.insert(tk.INSERT, str(ii) + " - " + measures_name[ii] + "\n")
        self.logger.print(level="DEBUG", message="show_corr_summary finished")

    # Scatter plot of the available data matrix
    def scatter_plot(self):
        self.logger.print(level="INFO", message="Scatter plot started ")
        if not self.verify_db():
            return

        self.scatter = ifas_plotting.ScatterPlotWithHistograms(
            self.db.get_data(), self.db.get_list_measures_dataframe()
            )
        self.logger.print(level="INFO", message="Scatter plot finished ")

    # Bar plot of the available correlations between measures
    def bar_plot(self):
        self.logger.print(level="INFO", message="Bar plot started ")
        if not self.verify_db():
            return

        if not hasattr(self.db, "dist_corr"):
            self.logger.print(level="ERROR", message="No correlation available ")
            tk.messagebox.showerror("Error", "No correlation available!", master=self.win)
            return

        list_meas = self.db.get_list_measures_dataframe()
        meas_idx = list_meas.index(self.combobox_features.get())
        correlations = self.db.get_correlations_with(idx=meas_idx)
        del list_meas[meas_idx]
        ifas_plotting.bar_plot(
            correlations, axes_labels=list_meas, target_var_idx=self.db.get_list_measures_dataframe()[meas_idx]
            )

        self.logger.print(level="INFO", message="Bar plot finished ")

    # Box plot of the available correlations between measures per source
    def box_plot(self):
        self.logger.print(level="INFO", message="Box plot started ")
        if not self.verify_db():
            return

        list_meas = self.db.get_list_measures_dataframe()
        meas_idx = list_meas.index(self.combobox_features.get())
        correlations = self.db.compute_correlations_per_source(idx=meas_idx)
        del list_meas[meas_idx]
        ifas_plotting.box_plot(
            correlations, axes_labels=list_meas, target_var_idx=self.db.get_list_measures_dataframe()[meas_idx]
            )

        self.logger.print(level="INFO", message="Box plot finished ")

    # Regression plot of available measures with the human scores
    def reg_plot(self):
        self.logger.print(level="INFO", message="Regression plot started ")
        if not self.verify_db():
            return

        if not hasattr(self.db, "model_par"):
            self.logger.print(level="ERROR", message="No models optimized in database ")
            tk.messagebox.showerror("Error", "No models optimized in database!", master=self.win)
            return

        xp, yp = self.db.estimate_using_model()
        list_meas = self.db.get_list_measures_dataframe()
        self.scatter = ifas_plotting.ScatterPlotTargetWithHistograms(
            self.db.get_data(), yp, xp, list_meas.index(self.combobox_features.get()), 
            axes_labels=self.db.get_list_measures_dataframe()
            )
        self.logger.print(level="INFO", message="Regression plot finished ")

    # Image changed 
    def image_changed(self, button=None):
        self.logger.print(level="INFO", message="Image change started ")
        if not self.verify_db():
            return

        if button == "ref_left":
            self.ref_img_idx -= 1
        elif button == "ref_right":
            self.ref_img_idx += 1
        elif button == "tst_left":
            self.tst_img_idx -= 1
        elif button == "tst_right":
            self.tst_img_idx += 1

        self.ref_img_idx = int(np.clip(self.ref_img_idx, 0, len(self.db.list_ref) - 1))
        self.tst_img_idx = int(
            np.clip(self.tst_img_idx, 0, len(self.db.dict_tst[self.db.list_ref[self.ref_img_idx]]) - 1)
            )

        crt_ref = self.db.db_folder.joinpath(
            self.db.list_ref[self.ref_img_idx], self.db.list_ref[self.ref_img_idx] + ".png"
            )
        crt_tst = self.db.db_folder.joinpath(
            self.db.list_ref[self.ref_img_idx], self.db.dict_tst[self.db.list_ref[self.ref_img_idx]][self.tst_img_idx]
            )
        self.disp_imgs(img_left=str(crt_ref), img_right=str(crt_tst))
        self.logger.print(level="INFO", message="Image change finished ")

    # Verify if the database is valid for iFAS
    def verify_db(self):
        if self.db is None:
            self.logger.print(level="ERROR", message="No database selected ")
            tk.messagebox.showerror("Error", "No database selected!", master=self.win)
            return False

        if not hasattr(self.db, "data"):
            self.logger.print(level="ERROR", message="No database selected ")
            tk.messagebox.showerror("Error", "No database selected!", master=self.win)
            return False

        return True

    # Display 2 given images, if not given iFAS logo is displayed
    def disp_imgs(self, img_left=None, img_right=None):
        self.logger.print(level="INFO", message="Displaying image started ")
        # Reading the images
        if img_left is None or img_right is None:
            img_left = self.dummy_img
            img_right = self.dummy_img
        else:
            dim = (int(self.size[0] / 2 - 25), int(self.size[1] - 50))
            img_left = cv2.imread(img_left)
            img_left = cv2.resize(img_left, dim)
            img_left = cv2.cvtColor(img_left, cv2.COLOR_BGR2RGB)
            img_right = cv2.imread(img_right)
            img_right = cv2.resize(img_right, dim)
            img_right = cv2.cvtColor(img_right, cv2.COLOR_BGR2RGB)

        # Setting images in the canvas
        img_left = Image.fromarray(img_left)
        self.imgtk_left = ImageTk.PhotoImage(img_left) 
        img_right = Image.fromarray(img_right)
        self.imgtk_right = ImageTk.PhotoImage(img_right)
        self.canvas_left.create_image(0, 0, anchor=tk.NW, image=self.imgtk_left)
        self.canvas_right.create_image(0, 0, anchor=tk.NW, image=self.imgtk_right)
        self.logger.print(level="INFO", message="Displaying image finished ")

    # Click on image to be shown
    def click_on_image(self, canvas=None):
        self.logger.print(level="INFO", message="Image clicked started ")
        if not self.verify_db():
            return

        if canvas == "ref":
            img_file = self.db.db_folder.joinpath(
                self.db.list_ref[self.ref_img_idx], self.db.list_ref[self.ref_img_idx] + ".png"
            )
        elif canvas == "tst":
            img_file = self.db.db_folder.joinpath(
                self.db.list_ref[self.ref_img_idx], 
                self.db.dict_tst[self.db.list_ref[self.ref_img_idx]][self.tst_img_idx]
            )

        img = cv2.imread(str(img_file))
        cv2.imshow(img_file.stem, img)

        self.logger.print(level="INFO", message="Image clicked finished ")

    # optimize the selected model
    def optimize(self):
        self.logger.print(level="INFO", message="Optimization started ")
        if not self.verify_db():
            return

        model = self.combobox_functions.get()
        target = self.db.get_list_measures_dataframe().index(self.combobox_features.get())
        input_txt = self.par_entry.get().split(",")
        if len(input_txt) != LENGHT_PARAMETERS[model]:
            tk.messagebox.showerror(
                "Error", model + " N parameters entered: " + str(len(input_txt)) + " Expected " + 
                str(LENGHT_PARAMETERS[model]), master=self.win
                )
            self.logger.print(
                level="ERROR", message=model + " N parameters entered: " + str(len(input_txt)) + " Expected " + 
                str(LENGHT_PARAMETERS[model])
                )
            return

        try:
            ini_par = list(map(float, input_txt))
            self.db.optimize_model(model, target, ini_par)
        except Exception as error:
            str_trace = traceback.format_exc()
            self.logger.print(level="ERROR", message="Something went wrong! \n" + repr(error))
            self.logger.print(level="ERROR", message=str_trace)
            tk.messagebox.showerror("Error", "Something went wrong! \n" + repr(error), master=self.win)
            return        

        self.logger.print(level="INFO", message="Optimization finished ")
        self.reg_plot()

    # Function to update the default settings when the model changes
    def combobox_functions_changed(self, event=None):
        model = self.combobox_functions.get()
        self.par_entry.delete(0, "end")
        self.par_entry.insert(0, DEFAULT_PARAMETERS[model])
