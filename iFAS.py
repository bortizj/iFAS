#!/usr/bin/env python2.7
# Importing necessary packages
import gi
gi.require_version('Gtk', '3.0')
from gi.repository import Gtk, Pango
import os, sys, pickle, subprocess, imp, time

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
os.environ['NO_AT_BRIDGE'] = str(1)

import numpy as np
import numpy.matlib
np.set_printoptions(precision=3)
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from matplotlib.backends.backend_gtk3agg import FigureCanvasGTK3Agg as FigureCanvas
from scipy import ndimage
from scipy import misc
import importlib
import inspect

# importing our costume scripts
import optimizationTools
import contentFeatures
import myUtilities
import imageSample
import myWidgets




# Create GUI Class and its components
class Main(object):
    def __init__(self):
        # Variable initialization
        self.workingPath = os.path.dirname(os.path.abspath(__file__))
        # Creating the new session logging file
        self.logManager = myWidgets.iFasLog(self.workingPath)

        # It is a list of strings with the name of the packages
        self.listAvailablePackages = None
        self.get_list_packages("./listPackages")
        self.selectedPackage = 'miselaneusPack'

        self.listAvailableMethods = []
        # Measures available in the pyhton script
        package = importlib.import_module(self.selectedPackage)
        for ii in range(len(inspect.getmembers(package, inspect.isfunction))):
            self.listAvailableMethods.append(inspect.getmembers(package, inspect.isfunction)[ii][0])

        # Measure selected by user to display in the difference map
        self.currentMeasure = self.listAvailableMethods[0]
        # Measures selected by user
        self.listSelectedMethods = [self.currentMeasure]

        # Axis selected by user to display in the plot
        self.xAxis = self.currentMeasure
        self.yAxis = self.currentMeasure

        # Flags to stop and measure time
        self.stopped = False
        self.startTime = 0

        # Default settings for sample image set
        self.selectedRefFile = self.workingPath + '/sample_images/ref_0.bmp'
        self.selectedProFile = self.workingPath + '/sample_images/pro_0.bmp'
        self.listReferenceFiles = [self.selectedRefFile]
        self.listProcessedFiles = dict([(self.selectedRefFile, [self.selectedProFile])])

        # Database set of reference images with their corresponding processed images
        self.setData = imageSample.DataSet(self.listReferenceFiles, self.listProcessedFiles,\
                                           self.selectedPackage, self.listSelectedMethods, object=self.logManager)
        if not self.setData.flagCorrectData:
            self.logManager.onLogging(logType='error', message='Error Loading your data.')
            myWidgets.popupWindowWithLabel()
            exit(0)

        # Current reference with processed images
        self.setImages = self.setData.data[self.selectedRefFile]
        # Setting up UI

        # Getting Window started with glade (See .glade File)
        self.builder = Gtk.Builder()
        self.window = self.builder.add_from_file('iFAS.glade')
        self.window = self.builder.get_object('MainWindow')
        self.window.set_title("IPI-imec: iFAS - Image Fidelity Assessment Software")

        # TODO Screen sizes could be improved by playing with the parameters
        # Getting monitor sizes in pixels
        screen = self.window.get_screen()
        monitors = []
        for m in range(screen.get_n_monitors()):
            monitors.append(screen.get_monitor_geometry(m))
        currentMonitor = screen.get_monitor_at_window(screen.get_active_window())
        monitorSettings = monitors[currentMonitor]
        self.videoHeightDefault = int((monitorSettings.height - 160.) / 2.)
        self.videoWidthDefault = int((monitorSettings.width - 30.) / 2.)

        # Creating the menu bar
        myWidgets.createMenuBar(self)

        # Creating the status bar
        self.statusBar = None
        self.create_status_bar()

        # Setting Image Reference space on UI
        drawable_loc = self.builder.get_object("hbox4")
        self.drawableReference = Gtk.Image()
        self.set_images_scrolled_window(drawable_loc, self.drawableReference)
        # Setting Image Processed space on UI
        self.drawableProcessed = Gtk.Image()
        self.set_images_scrolled_window(drawable_loc, self.drawableProcessed)
        # Setting Image Difference space on UI
        drawable_loc = self.builder.get_object("box5")
        self.drawableDifference = Gtk.Image()
        self.set_images_scrolled_window(drawable_loc, self.drawableDifference)

        # Setting Plot canvas
        self.plotPlace = self.builder.get_object("box4")
        self.figure = Figure()
        self.axis = self.figure.add_subplot(111)
        self.axis.tick_params(labelsize=7)
        self.axis.set_autoscale_on(True)
        self.axis.set_title('Reference set scatter plot')
        self.plotAxis, = self.axis.plot([], [], linestyle='None', marker='x', color='r', markersize=8, \
                                        fillstyle='full', markeredgewidth=3.0)
        self.canvas = FigureCanvas(self.figure)
        # TODO Screen sizes could be improved by playing with the parameters
        self.canvas.set_size_request(int(9. * self.videoWidthDefault / 16.), self.videoHeightDefault)
        self.plotPlace.pack_start(self.canvas, True, True, 0)

        # Setting Text Viewer
        self.scrolledWindowPlace = self.builder.get_object("box6")
        self.scrolledWindow = Gtk.ScrolledWindow()
        self.textView = Gtk.TextView()
        self.textBuffer = self.textView.get_buffer()
        self.textTag = self.textBuffer.create_tag("bold", size_points=12, weight=Pango.Weight.BOLD)
        self.create_text_view()

        # Setting labels settings
        self.labelProcessedImage = self.builder.get_object("processed_image")
        self.labelProcessedImage.modify_font(Pango.FontDescription('Sans 16'))
        self.labelReferenceImage = self.builder.get_object("label3")
        self.labelReferenceImage.modify_font(Pango.FontDescription('Sans 16'))
        self.labelDifferenceImage = self.builder.get_object("label_dif_image")
        self.labelDifferenceImage.modify_font(Pango.FontDescription('Sans 16'))

        # Setting Stop button
        self.buttonStop = self.builder.get_object("button2")
        self.buttonStop.connect("clicked", self.on_stop_all)

        # Setting Start button
        # self.buttonStart = self.builder.get_object("button1")
        # self.buttonStart.connect("clicked", self.executeSelection)

        self.update_images()
        # Destroying when quiting
        self.window.connect("delete_event", lambda w, e: Gtk.main_quit())
        self.window.connect("destroy", lambda w: Gtk.main_quit())
        # Displaying UI window
        self.window.show_all()
        self.on_about_click()

    # Getting list of packages aka python scripts, fidelity group
    def get_list_packages(self, fileLocation):
        self.logManager.onLogging(logType='info', message='Verifying list of iFAS packages')
        listPackages = []
        with open(fileLocation) as f:
            for line in f:
                line = line.replace('\n', '')
                if self.verify_package(line):
                    listPackages.append(line)
                else:
                    message = 'Verify your listPackages file. One ore more lines' +\
                                          'could be corrupted or empty: ' + line
                    self.logManager.onLogging(logType='error', message=message)
                    try:
                        self.print_message(message)
                    except AttributeError:
                        print message
        # It is a list of strings with the name of the packages
        self.listAvailablePackages = listPackages
        self.logManager.onLogging(logType='info', message='; '.join(listPackages))
        self.logManager.onLogging(logType='info', message='Link to packages are available')

    # Verifying that packages aka python scripts, fidelity groups are located in folder
    def verify_package(self, packageName):
        self.logManager.onLogging(logType='debug', message='Verifying Package: ' + packageName)
        try:
            if not packageName == '':
                result = importlib.import_module(packageName)
                if result:
                    self.logManager.onLogging(logType='debug', message='Package verified')
                    return True
        except ImportError:
            self.logManager.onLogging(logType='debug', message='Package corrupted')
            return False

    # function to print messages in the interface textview
    def print_message(self, message):
        self.logManager.onLogging(logType='debug', message='Printing message')
        currentTime = myWidgets.getTime()
        self.textBuffer.insert_with_tags(self.textBuffer.get_end_iter(), '\n\n' + currentTime + '\n', self.textTag)
        self.textBuffer.insert(self.textBuffer.get_end_iter(), message + '\n')
        self.logManager.onLogging(logType='debug', message='Message printed')

    # function to create the status bar
    def create_status_bar(self):
        self.logManager.onLogging(logType='debug', message='Creating status bar')
        self.statusBar = self.builder.get_object("levelbar1")
        self.statusBar.set_property("width-request", self.videoWidthDefault)
        self.statusBar.set_min_value(0)
        self.statusBar.set_max_value(100)
        self.statusBar.set_mode(Gtk.LevelBarMode.CONTINUOUS)
        self.logManager.onLogging(logType='debug', message='Status bar created')

    # function to set image on scrolled window
    def set_images_scrolled_window(self, location, image):
        self.logManager.onLogging(logType='debug', message='Setting image in scrolling window')
        scrolledwindow = Gtk.ScrolledWindow()
        scrolledwindow.set_hexpand(True)
        scrolledwindow.set_vexpand(True)
        scrolledwindow.set_policy(Gtk.PolicyType.AUTOMATIC, Gtk.PolicyType.AUTOMATIC)
        scrolledwindow.set_property("width-request", self.videoWidthDefault)
        scrolledwindow.set_property("height-request", self.videoHeightDefault)
        scrolledwindow.add_with_viewport(image)
        location.pack_start(scrolledwindow, True, True, 0)
        self.logManager.onLogging(logType='debug', message='Image in scrolling window set')

    # function to set the parameters of the text viewer
    def create_text_view(self):
        self.logManager.onLogging(logType='debug', message='Creating text view')
        self.scrolledWindow.set_hexpand(True)
        self.scrolledWindow.set_vexpand(True)
        self.scrolledWindow.set_policy(Gtk.PolicyType.AUTOMATIC, Gtk.PolicyType.ALWAYS)
        # TODO Screen sizes could be improved by playing with the parameters
        self.scrolledWindow.set_property("width-request", int(7. * self.videoWidthDefault / 16.))
        self.scrolledWindow.set_property("height-request", self.videoHeightDefault)
        self.scrolledWindowPlace.pack_start(self.scrolledWindow, True, True, 0)
        self.textView.connect("size-allocate", self.auto_scroll)
        self.scrolledWindow.add(self.textView)
        self.logManager.onLogging(logType='debug', message='Text view created')

    # function to autoscroll the textViewer
    def auto_scroll(self, *args):
        adj = self.textView.get_vadjustment()
        adj.set_value(adj.get_upper() - adj.get_page_size())

    # function to stop every process
    def on_stop_all(self, button=None):
        self.logManager.onLogging(logType='debug', message='Stop button pressed')
        self.print_message("Stop Pressed")
        self.stopped = True
        self.logManager.onLogging(logType='debug', message='Stopped')

    def update_images(self):
        self.logManager.onLogging(logType='debug', message='Updating images')
        self.save_temp_files()
        self.drawableReference.set_from_file('/tmp/temp_ref.png')
        self.drawableProcessed.set_from_file('/tmp/temp_pro.png')
        self.drawableDifference.set_from_file('/tmp/temp_cd.png')
        # Removing temp files
        os.remove("/tmp/temp_ref.png")
        os.remove("/tmp/temp_pro.png")
        os.remove("/tmp/temp_cd.png")
        self.window.show_all()
        self.logManager.onLogging(logType='debug', message='Images updated')

    # Saving temporary image files to be displayed on the canvas
    def save_temp_files(self):
        self.logManager.onLogging(logType='debug', message='Saving temporal files')
        self.setImages = self.setData.data[self.selectedRefFile]
        image = self.setImages.imageReference
        misc.imsave('/tmp/temp_ref.png', image)
        image = self.setImages.imageProcessed
        misc.imsave('/tmp/temp_pro.png', image)
        image = self.setImages.imageDifference
        cmap = plt.get_cmap('jet')
        if np.max(image) != np.min(image):
            image = (np.double(image) - np.min(image)) / (np.max(image) - np.min(image))
        elif np.max(image) != 0:
            image = np.double(image) / np.max(image)
        else:
            image = np.zeros_like(image)
        image = np.delete(cmap(np.uint8(255 * image)), 3, 2)
        misc.imsave('/tmp/temp_cd.png', image)
        self.logManager.onLogging(logType='debug', message='Temporal files saved')

    # Setting to default UI parameters
    def return_default(self):
        self.logManager.onLogging(logType='debug', message='Returning to defaults')
        self.listAvailablePackages = None
        self.get_list_packages("./listPackages")
        self.selectedPackage = 'miselaneusPack'

        # Measures available in the pyhton script
        self.listAvailableMethods = []
        # Measures available in the pyhton script
        package = importlib.import_module(self.selectedPackage)
        for ii in range(len(inspect.getmembers(package, inspect.isfunction))):
            self.listAvailableMethods.append(inspect.getmembers(package, inspect.isfunction)[ii][0])

        # Measure selected by user to display in the difference map
        self.currentMeasure = self.listAvailableMethods[0]
        # Measures selected by user
        self.listSelectedMethods = [self.currentMeasure]

        # Axis selected by user to display in the plot
        self.xAxis = self.currentMeasure
        self.yAxis = self.currentMeasure

        # Flags to stop and measure time
        self.stopped = False
        self.startTime = 0

        # Default settings for sample image set
        self.selectedRefFile = self.workingPath + '/sample_images/ref_0.bmp'
        self.selectedProFile = self.workingPath + '/sample_images/pro_0.bmp'
        self.listReferenceFiles = [self.selectedRefFile]
        self.listProcessedFiles = dict([(self.selectedRefFile, [self.selectedProFile])])
        self.setData = imageSample.DataSet(self.listReferenceFiles, self.listProcessedFiles, \
                                           self.selectedPackage, self.listSelectedMethods, object=self.logManager)
        self.setImages = self.setData.data[self.selectedRefFile]
        message = 'Returning to default parameters'
        self.print_message(message)
        self.update_data()
        self.update_images()
        self.logManager.onLogging(logType='debug', message='Default parameters set')

    def update_data(self):
        self.logManager.onLogging(logType='debug', message='Updating data to display')
        self.setData.computeData()
        self.logManager.onLogging(logType='debug', message='Data updated')

    def data_to_string(self):
        return self.setData.data2String()

    # Changing the list of methods
    def on_select_measures(self, button=None, doCompute=True):
        self.logManager.onLogging(logType='debug', message='Changing selected measures')
        # Measures available in the pyhton script
        popwin = myWidgets.popupWindowWithList(self.listAvailableMethods, sel_method=Gtk.SelectionMode.MULTIPLE,\
                                               message="List of Measures")
        self.listSelectedMethods = popwin.list_items
        if not self.listSelectedMethods:
            self.return_default()

        self.currentMeasure = self.listSelectedMethods[0]
        if doCompute:
            self.update_data()
            str2Print = self.data_to_string()
            self.print_message('Package: ' + self.selectedPackage + '\n' + 'Difference map: ' + self.currentMeasure +\
                              '\n' + str2Print)
            self.update_images()
            self.logManager.onLogging(logType='info', message='Package: ' + self.selectedPackage + '\n'\
                                                              + 'Difference map: ' + self.currentMeasure + '\n'\
                                                              + str2Print)
        self.logManager.onLogging(logType='debug', message='Selected measures changed')

    # Changing the package
    def on_select_package(self, button=None, doCompute=True):
        self.logManager.onLogging(logType='debug', message='Changing selected package')
        popwin = myWidgets.popupWindowWithList(self.listAvailablePackages, sel_method=Gtk.SelectionMode.SINGLE,\
                                               message="List of Packages")
        self.selectedPackage = popwin.list_items
        if not self.selectedPackage:
            self.return_default()
        else:
            package = importlib.import_module(self.selectedPackage)
            self.listAvailableMethods = []
            for ii in range(len(inspect.getmembers(package, inspect.isfunction))):
                self.listAvailableMethods.append(inspect.getmembers(package, inspect.isfunction)[ii][0])
            # Measure selected by user to display in the difference map
            self.currentMeasure = self.listAvailableMethods[0]
            # Measures selected by user
            self.listSelectedMethods = [self.currentMeasure]
            if doCompute:
                self.update_data()
                self.update_images()
        self.logManager.onLogging(logType='debug', message='Selected package changed')

    # Changing the difference map
    def on_change_diff_map(self, button):
        self.logManager.onLogging(logType='debug', message='Changing difference map')
        popwin = myWidgets.popupWindowWithList(self.listSelectedMethods, sel_method=Gtk.SelectionMode.SINGLE,\
                                               message="List of Measures")
        self.currentMeasure = popwin.list_items
        if not self.currentMeasure:
            self.return_default()
        else:
            self.setData.changeDiffMap(self.currentMeasure)
        str2Print = self.data_to_string()
        self.print_message('Package: ' + self.selectedPackage + '\n' + 'Difference map: ' + self.currentMeasure +\
                          '\n' + str2Print)
        self.logManager.onLogging(logType='info', message='Package: ' + self.selectedPackage + '\n'\
                                                          + 'Difference map: ' + self.currentMeasure + '\n' + str2Print)
        self.update_images()
        self.logManager.onLogging(logType='debug', message='Difference map changed')

    # Changing the displaying image
    def on_image_set_changed(self, button):
        self.logManager.onLogging(logType='debug', message='Changing selected image reference')
        previousFile = self.selectedRefFile
        popwin = myWidgets.popupWindowWithList(self.listReferenceFiles, sel_method=Gtk.SelectionMode.SINGLE,\
                                               split_=True, message="Reference images")
        self.selectedRefFile = popwin.list_items
        if not self.selectedRefFile:
            self.selectedRefFile = previousFile
            self.print_message("Error in Reference file selection. Returning to: " + previousFile.split('/')[-1])
            self.logManager.onLogging(logType='error', message="Error in Reference file selection. Returning to: "\
                                                                + previousFile.split('/')[-1])
        else:
            flag = False
            for ii in self.listReferenceFiles:
                if self.selectedRefFile in ii:
                    newFile = ii
                    flag = True
            if flag:
                self.selectedRefFile = newFile
                self.print_message("Reference file selection set to: " + newFile.split('/')[-1])
                self.logManager.onLogging(logType='info', message="Reference file selection set to: "\
                                                                  + newFile.split('/')[-1])
            else:
                self.selectedRefFile = previousFile
                self.print_message("Error finding Reference. Returning to: " + previousFile.split('/')[-1])
                self.logManager.onLogging(logType='error', message="Error finding Reference. Returning to: "\
                                                                   + previousFile.split('/')[-1])
        previousFile = self.selectedProFile
        self.setImages = self.setData.data[self.selectedRefFile]
        listFiles = self.setImages.returnListProcessed()
        popwin = myWidgets.popupWindowWithList(listFiles, sel_method=Gtk.SelectionMode.SINGLE, split_=True,\
                                               message="Processed images")
        self.selectedProFile = popwin.list_items
        if not self.selectedProFile:
            self.selectedProFile = previousFile
            self.print_message("Error in processed file selection. Returning to: " + previousFile.split('/')[-1])
            self.logManager.onLogging(logType='error', message="Error in processed file selection. Returning to: "\
                                                               + previousFile.split('/')[-1])
        else:
            flag = False
            for ii in listFiles:
                if self.selectedProFile in ii:
                    newFile = ii
                    flag = True
            if flag:
                self.selectedProFile = newFile
                self.print_message("Processed file selection set to: " + newFile.split('/')[-1])
                self.logManager.onLogging(logType='info', message="Processed file selection set to: "\
                                                                  + newFile.split('/')[-1])
            else:
                self.selectedProFile = previousFile
                self.print_message("Error finding selection. Returning to sample: " + previousFile.split('/')[-1])
                self.logManager.onLogging(logType='error', message="Error finding selection. Returning to sample: "\
                                                                   + previousFile.split('/')[-1])
        self.setData.changeProcessedImage(self.selectedRefFile, self.selectedProFile)
        self.update_images()
        self.logManager.onLogging(logType='debug',  message='Images updated')
        # TODO modify to clear plot place

    # Computing single reference single processed image fidelity using the selected measures
    def single_ref_single_pro(self, button):
        self.logManager.onLogging(logType='debug', message='Starting new: Single - Single')
        self.return_default()
        self.on_select_package(doCompute=False)
        self.on_select_measures(doCompute=False)
        self.selectedRefFile = myWidgets.load_file(self.window, 'img', 'Select your Reference Image')
        self.selectedProFile = myWidgets.load_file(self.window, 'img', 'Select your Processed Image')
        if self.selectedRefFile and self.selectedProFile:
            self.listReferenceFiles = [self.selectedRefFile]
            self.currentMeasure = self.listSelectedMethods[0]
            self.listProcessedFiles = dict([(self.selectedRefFile, [self.selectedProFile])])
            self.setData = imageSample.DataSet(self.listReferenceFiles, self.listProcessedFiles, \
                                               self.selectedPackage, self.listSelectedMethods, object=self.logManager)
            self.update_images()
            str2Print = self.data_to_string()
            self.print_message('Package: ' + self.selectedPackage + '\n' + 'Difference map: ' + self.currentMeasure +\
                              '\n' + str2Print)
            self.logManager.onLogging(logType='info', message='Package: ' + self.selectedPackage + '\n'\
                                                              + 'Difference map: ' + self.currentMeasure + '\n'\
                                                              + str2Print)
            self.logManager.onLogging(logType='info', message="Process Finished!")
            self.print_message("Process Finished!")
        else:
            self.return_default()
        self.logManager.onLogging(logType='debug', message='Single - Single finished')

    # Computing single reference multiple processed image fidelity using the selected measures
    def single_ref_multiple_pro(self, button):
        self.logManager.onLogging(logType='debug', message='Starting new: Single - Multiple')
        self.return_default()
        self.on_select_package(doCompute=False)
        self.on_select_measures(doCompute=False)
        self.selectedRefFile = myWidgets.load_file(self.window, 'img', 'Select your Reference Image')
        processedFiles = myWidgets.load_file(self.window, 'img', 'Select your Processed Images', multiple=True)
        if self.selectedRefFile and self.selectedProFile:
            self.listReferenceFiles = [self.selectedRefFile]
            self.currentMeasure = self.listSelectedMethods[0]
            self.selectedProFile = processedFiles[0]
            self.listProcessedFiles = dict([(self.selectedRefFile, processedFiles)])
            self.setData = imageSample.DataSet(self.listReferenceFiles, self.listProcessedFiles, \
                                               self.selectedPackage, self.listSelectedMethods, object=self.logManager)
            self.update_images()
            str2Print = self.data_to_string()
            self.print_message(
                'Package: ' + self.selectedPackage + '\n' + 'Difference map: ' + self.currentMeasure + '\n' + \
                str2Print)
            self.print_message("Process Finished!")
            self.logManager.onLogging(logType='info', message='Package: ' + self.selectedPackage + '\n'\
                                                              + 'Difference map: ' + self.currentMeasure + '\n'\
                                                              + str2Print)
            self.logManager.onLogging(logType='info', message="Process Finished!")
        else:
            self.return_default()
        self.logManager.onLogging(logType='debug', message='Single - Multiple finished')

    # Computing multiple reference multiple processed image fidelity using the selected measures
    def multiple_ref_multiple_pro(self, button=None):
        self.logManager.onLogging(logType='debug', message='Starting new: Multiple - Multiple')
        self.return_default()
        self.print_message('Computing your data!\nThis could take a while depending of your data size.')
        self.on_select_package(doCompute=False)
        self.on_select_measures(doCompute=False)
        pythonFile = myWidgets.load_file(self.window, 'txt', 'Select your .py file')
        if pythonFile:
            try:
                path2Files, listReferences, dataSet = imp.load_source('module.name', pythonFile).myDataBase()
                path2Files = '/'.join(pythonFile.split('/')[:-1]) + path2Files
            except SyntaxError:
                self.print_message('File Syntax corrupted. Please verify your file!')
                self.logManager.onLogging(logType='error', message='File Syntax corrupted. Please verify your file!')
                self.return_default()
                return
        else:
            self.logManager.onLogging(logType='error', message='File Syntax corrupted. Please verify your file!')
            self.return_default()
            return
        self.listProcessedFiles = dict()
        self.listReferenceFiles = []
        for ii in listReferences:
            self.listReferenceFiles.append(path2Files + ii)
            self.listProcessedFiles[path2Files + ii] = []
            for jj in dataSet[ii]:
                self.listProcessedFiles[path2Files + ii].append(path2Files + jj)
        self.setData = imageSample.DataSet(self.listReferenceFiles, self.listProcessedFiles, \
                                           self.selectedPackage, self.listSelectedMethods, object=self.logManager)
        self.currentMeasure = self.listSelectedMethods[0]
        self.selectedRefFile = self.listReferenceFiles[0]
        self.selectedProFile = self.listProcessedFiles[self.selectedRefFile][0]
        self.update_images()
        str2Print = self.data_to_string()
        self.print_message('Package: ' + self.selectedPackage + '\n' + 'Difference map: ' + self.currentMeasure + '\n' + \
                           str2Print)
        self.logManager.onLogging(logType='info', message='Package: ' + self.selectedPackage + '\n'\
                                                          + 'Difference map: ' + self.currentMeasure + '\n' + str2Print)
        self.print_message("Process Finished!")
        self.logManager.onLogging(logType='debug', message='Multiple - Multiple Finished')

    # Save computed data
    def on_save_data(self, button=None):
        self.logManager.onLogging(logType='debug', message='Saving Data')
        popwin = myWidgets.popupWindowWithTextInput('Name your file: ')
        with open(self.workingPath + '/' + popwin.file_name + '.iFAS', 'w') as f:
            pickle.dump([self.selectedPackage, self.listAvailableMethods, self.listSelectedMethods, self.currentMeasure,\
                         self.listReferenceFiles, self.selectedRefFile, self.selectedProFile, self.setData], f)
        self.print_message("Save completed!")
        self.logManager.onLogging(logType='debug', message='Save completed!')

    # Load precomputed data
    def on_load_data(self, button=None):
        self.logManager.onLogging(logType='debug', message='Loading Data')
        temp = myWidgets.load_file(self.window, 'txt', 'Select your .iFAS file')
        try:
            with open(temp) as f:
                self.selectedPackage, self.listAvailableMethods, self.listSelectedMethods, self.currentMeasure,\
                self.listReferenceFiles, self.selectedRefFile, self.selectedProFile, self.setData = pickle.load(f)
            self.xAxis = self.currentMeasure
            self.yAxis = self.currentMeasure
            self.update_images()
            str2Print = self.data_to_string()
            self.print_message('Package: ' + self.selectedPackage + '\n' + 'Difference map: ' + self.currentMeasure \
                               + '\n' + str2Print)
            self.logManager.onLogging(logType='info', message='Package: ' + self.selectedPackage + '\n'\
                                                              + 'Difference map: ' + self.currentMeasure + '\n'\
                                                              + str2Print)
            self.print_message("Load Completed!")
            self.logManager.onLogging(logType='info', message='Load Completed!')
        except IOError:
            self.print_message('File corrupted or not an iFAS file. Please verify your file!')
            self.logManager.onLogging(logType='error', message='File corrupted or not an iFAS file.'\
                                                               + 'Please verify your file!')
            self.return_default()
            return
        self.logManager.onLogging(logType='debug', message='Data loaded')

    # Plotting data on UI for current reference ImageSet
    def plot_current_selection(self):
        self.logManager.onLogging(logType='debug', message='Plotting data on UI for current reference ImageSet')
        self.setImages = self.setData.data[self.selectedRefFile]
        _, xAxisValues = self.setImages.returnVector(measure=self.xAxis)
        _, yAxisValues = self.setImages.returnVector(measure=self.yAxis)
        self.plotAxis.set_data(xAxisValues, yAxisValues)
        self.axis.set_xlim([np.min(xAxisValues) - 0.05 * np.min(xAxisValues), \
                            np.max(xAxisValues) + 0.05 * np.max(xAxisValues)])
        self.axis.set_ylim([np.min(yAxisValues) - 0.05 * np.min(yAxisValues),\
                            np.max(yAxisValues) + 0.05 * np.max(yAxisValues)])
        self.axis.relim()
        self.axis.autoscale_view(True, True, True)
        self.axis.set_xlabel(self.xAxis, fontsize=7)
        self.axis.set_ylabel(self.yAxis, fontsize=7)
        self.figure.canvas.draw()
        p, s, t, pd = myUtilities.compute_1dcorrelatiosn(xAxisValues, yAxisValues)
        self.print_message("Pearson R = " + "%.5f" % p + "\n" + "Spearman R = " + "%.5f" % s + "\n" + \
                          "Distance R = " + "%.5f" % pd)
        # self.printMessage("Pearson R = " + "%.5f" % p + "\n" + "Spearman R = " + "%.5f" % s + "\n" +\
        #                   "Kendall t = " + "%.5f" % t + "\n" + "Distance R = " + "%.5f" % pd)
        self.logManager.onLogging(logType='debug', message='ata on UI for current reference ImageSet plotted')

    # Plotting selected x and y axis
    def plot_x_y(self, button):
        self.logManager.onLogging(logType='debug', message='Plotting selected x and y axis')
        popwin = myWidgets.popupWindowWithList(self.listSelectedMethods, sel_method=Gtk.SelectionMode.SINGLE,\
                                               message="X axis measure")
        self.xAxis = popwin.list_items
        if not self.xAxis:
            self.xAxis = self.listSelectedMethods[0]
        popwin = myWidgets.popupWindowWithList(self.listSelectedMethods, sel_method=Gtk.SelectionMode.SINGLE,\
                                               message="Y axis measure")
        self.yAxis = popwin.list_items
        if not self.yAxis:
            self.yAxis = self.listSelectedMethods[0]
        self.print_message("Current x axis: " + self.xAxis)
        self.print_message("Current y axis: " + self.yAxis)
        self.plot_current_selection()
        self.logManager.onLogging(logType='debug', message='Selected x and y axis plotted')

    # Plotting ALL x and y axis
    def on_scatter_plot(self, button=None):
        self.logManager.onLogging(logType='debug', message='Plotting ALL x and y axis')
        popwin = myWidgets.popupWindowWithList(self.listSelectedMethods, sel_method=Gtk.SelectionMode.SINGLE,\
                                               message="X axis measure")
        xAxis = popwin.list_items
        popwin = myWidgets.popupWindowWithList(self.listSelectedMethods, sel_method=Gtk.SelectionMode.SINGLE,\
                                               message="Y axis measure")
        yAxis = popwin.list_items
        Data = self.setData.getDictTableperRef(xAxis, yAxis)
        _ = myWidgets.popupWindowWithScatterPlot(self, Data, (xAxis, yAxis))
        self.logManager.onLogging(logType='debug', message='ALL x and y axis plotted')

    # Plotting Multiple distortion types according to selected by user
    def on_multiple_distortion_plot(self, button=None, data=None):
        self.logManager.onLogging(logType='debug', message='Plotting Multiple distortion types according to selected by user')
        popwin = myWidgets.popupWindowWithTextInput('Number of distortions: ')
        if not popwin.file_name:
            self.print_message(popwin.file_name + " is not integer!")
            return
        Ndistortions = int(popwin.file_name)
        listProcessedFiles = self.setData.getListProcessed()
        listDistortions = []
        for ii in range(Ndistortions):
            popwin = myWidgets.popupWindowWithList(sorted(listProcessedFiles, key=lambda s: s[-9:-1]),\
                                                   sel_method=Gtk.SelectionMode.MULTIPLE, split_=True,\
                                                   message=str(ii) + "th distortion")
            listDistortions.append(popwin.list_items)
        popwin = myWidgets.popupWindowWithList(self.listSelectedMethods, sel_method=Gtk.SelectionMode.SINGLE,\
                                               message="X axis")
        xAxis = popwin.list_items
        popwin = myWidgets.popupWindowWithList(self.listSelectedMethods, sel_method=Gtk.SelectionMode.SINGLE,\
                                               message="Y axis")
        yAxis = popwin.list_items
        Data = self.setData.getDataLabel(Ndistortions, listDistortions, xAxis, yAxis)
        _ = myWidgets.popupWindowWithScatterPlotWizard(self, np.array(Data), (xAxis, yAxis))
        self.logManager.onLogging(logType='debug',
                                  message='Plotting Multiple distortion types according to selected by user')

    # Adds precomputed measure. Name human is reserved for human scores (DMOS)
    def add_precomputed_measure(self, button=None):
        self.logManager.onLogging(logType='debug', message='Adding precomputed measure')
        temp = myWidgets.load_file(self.window, 'txt', 'Select your measure file')
        if temp:
            popwin = myWidgets.popupWindowWithTextInput('Name your measure [name it \'human\'\nfor subjective scores]: ')
            self.setData.setPrecomputedData(temp, popwin.file_name, self.logManager)
            if not (popwin.file_name in self.listSelectedMethods):
                self.listSelectedMethods.append(popwin.file_name)
            self.print_message("File " + temp + " loaded.")
            self.logManager.onLogging(logType='info', message='File ' + temp + ' loaded.')
        else:
            self.print_message("Error finding data File")
            self.logManager.onLogging(logType='error', message='Error finding data File!')
        self.logManager.onLogging(logType='debug', message='Precomputed measure added')

    # Adds new python script to the configuration file
    def add_new_python_script(self, button=None):
        self.logManager.onLogging(logType='debug', message='Adding new python script to the configuration file')
        popwin = myWidgets.popupWindowWithTextInput()
        self.listAvailablePackages.append(popwin.file_name)
        file('./listPackages', 'w').write("\n".join(self.listAvailablePackages) + "\n")
        self.get_list_packages("./listPackages")
        self.print_message('New fidelity pack add to the default packages.')
        self.logManager.onLogging(logType='info', message='New fidelity pack add to the default packages.')
        self.logManager.onLogging(logType='debug', message='New python script to the configuration file added')

    # Plots the correlation bar for the selected measures and the 'human' data if available
    def global_correlation_bar_plot(self, button=None):
        self.logManager.onLogging(logType='debug', message='Plotting correlation bar for the measures and the human')
        correlations = {}
        for ii in self.listSelectedMethods:
            if ii != 'human':
                Data = self.setData.getDataMatrix(ii, 'human')
                p, s, t, pd = myUtilities.compute_1dcorrelatiosn(Data[:, 0], Data[:, 1])
                correlations[ii] = np.abs(np.asarray([p, s, t, pd]))
                self.print_message(ii + '\n' + 'Pearson R: ' + "%.5f" % p + '\n' + 'Spearman R: ' + "%.5f" % s \
                                   + 'Distance R: ' + "%.5f" % pd)
                # self.printMessage(ii + '\n' + 'Pearson R: ' + "%.5f" % p + '\n' + 'Spearman R: ' + "%.5f" % s \
                #                    + '\n' + 'Kendall t: ' + "%.5f" % t + '\n' + 'Distance R: ' + "%.5f" % pd)
        pmax = 0.
        smax = 0.
        tmax = 0.
        pdmax = 0.
        for ii in correlations:
            if np.abs(correlations[ii][0]) > pmax:
                pmax = correlations[ii][0]
                posMaxP = ii
            if np.abs(correlations[ii][1]) > smax:
                smax = correlations[ii][1]
                posMaxS = ii
            if np.abs(correlations[ii][2]) > tmax:
                tmax = correlations[ii][2]
                posMaxt = ii
            if np.abs(correlations[ii][3]) > pdmax:
                pdmax = correlations[ii][3]
                posMaxD = ii
        message = "Best according to Pearson R is " + posMaxP + ': ' + "%.5f" % pmax
        message += "\nBest according to Spearman R is " + posMaxS + ': ' + "%.5f" % smax
        # message += "\nBest according to Kendall t is " + posMaxt + ': ' + "%.5f" % tmax
        message += "\nBest according to Correlation distance is " + posMaxD + ': ' + "%.5f" % pdmax
        self.print_message(message)
        self.logManager.onLogging(logType='debug', message=message)
        _ = myWidgets.popupWindowWithBarPlot(self, correlations)
        self.logManager.onLogging(logType='debug', message='Correlation bar for the measures and the human plotted')

    # Box plot of the correlations per source sample
    def box_plot_reference(self, button=None):
        self.logManager.onLogging(logType='debug', message='Box plot of the correlations per source sample')
        correlations = {}
        for ii in self.listSelectedMethods:
            if ii != 'human':
                correlations[ii] = []
                for jj in self.listReferenceFiles:
                    Data = self.setData.getDataMatrixReference(ii, 'human', jj)
                    p, s, t, pd = myUtilities.compute_1dcorrelatiosn(Data[:, 0], Data[:, 1])
                    correlations[ii].append([p, s, t, pd])
                    message = ii + '\n' + jj.split('/')[-1] + '\n' + 'Pearson R: ' + "%.5f" % p + '\n' + \
                              'Spearman R: ' + "%.5f" % s + '\n' + 'Kendall t: ' + "%.5f" % t + '\n' + \
                              'Distance R: ' + "%.5f" % pd
                    self.print_message(message=message)
                    self.logManager.onLogging(logType='debug', message=message)
                    # self.printMessage(ii + '\n' + jj.split('/')[-1] + '\n' + 'Pearson R: ' + "%.5f" % p + '\n' + \
                    #                   'Spearman R: ' + "%.5f" % s + '\n' + 'Kendall t: ' + "%.5f" % t + '\n' + \
                    #                   'Distance R: ' + "%.5f" % pd)
                correlations[ii] = np.asarray(correlations[ii])
        _ = myWidgets.popupWindowWithBoxPlot(self, correlations)
        self.logManager.onLogging(logType='debug', message='Correlations per source sample plotted')

    # Histogram of the content features of the references
    def histogram_content_features(self, button=None):
        self.logManager.onLogging(logType='debug', message='Histogram of the content features of the references')
        listContentFunctions = inspect.getmembers(contentFeatures, inspect.isfunction)
        # Extracting the string
        listContentFeatures = []
        for ii in range(len(listContentFunctions)):
            listContentFeatures.append(listContentFunctions[ii][0])
        popwin = myWidgets.popupWindowWithList(listContentFeatures, sel_method=Gtk.SelectionMode.SINGLE,\
                                               message="Content features")
        self.setData.computeContentFeatures(contentFeatures, [popwin.list_items])
        Data = []
        Data_names = []
        for ii in self.listReferenceFiles:
            Data.append(self.setData.getContentFeature(popwin.list_items, ii))
            Data_names.append(ii.split('/')[-1])
        _ = myWidgets.popupWindowWithContentHistogram(Object=self, data=Data)
        self.logManager.onLogging(logType='debug', message='Histogram of the content features of the references plotted')

    # Linear and non linear Regression Wizzard
    def on_regression(self, button=None, data=None):
        self.logManager.onLogging(logType='debug', message='Linear and non linear Regression Wizzard')
        popwin = myWidgets.popupWindowWithList(self.listReferenceFiles, sel_method=Gtk.SelectionMode.MULTIPLE,\
                                               split_=True, message="Reference image TRAINING set")
        listTrain = popwin.list_items
        popwin = myWidgets.popupWindowWithList(self.listReferenceFiles, sel_method=Gtk.SelectionMode.MULTIPLE,\
                                                  split_=True, message="Reference image TEST set")
        listTest = popwin.list_items
        popwin = myWidgets.popupWindowWithList(self.listSelectedMethods, sel_method=Gtk.SelectionMode.SINGLE,\
                                               message="Independent variable [x-axis]")
        xAxis = popwin.list_items
        popwin = myWidgets.popupWindowWithList(self.listSelectedMethods, sel_method=Gtk.SelectionMode.SINGLE,\
                                               message="Dependent variable [y-axis]")
        yAxis = popwin.list_items
        TrainList = []
        TestList = []
        for ii in listTrain:
            for jj in self.listReferenceFiles:
                if ii in jj:
                    TrainList.append(jj)
        for ii in listTest:
            for jj in self.listReferenceFiles:
                if ii in jj:
                    TestList.append(jj)
        aopt, [p, s, t, pd], [p0, s0, t0, pd0] = self.setData.regressionModel(TrainList, TestList, xAxis, yAxis, data)
        messagea = ''
        for ii in range(len(aopt)):
            messagea += 'a' + str(ii) + ": %.5f" % aopt[ii] + '\n'
        self.print_message("The optimal parameters for function\n" + optimizationTools.fun_text(data) + "\n" + messagea)
        self.logManager.onLogging(logType='info', message="The optimal parameters for function\n"\
                                                          + optimizationTools.fun_text(data) + "\n" + messagea)
        self.print_message('Training values: ' + '\n' + 'Pearson R: ' + "%.5f" % p + '\n' + \
                          'Spearman R: ' + "%.5f" % s + '\n' + 'Distance R: ' + "%.5f" % pd)
        self.logManager.onLogging(logType='info', message='Training values: ' + '\n' + 'Pearson R: ' + "%.5f" % p\
                                                          + '\n' + 'Spearman R: ' + "%.5f" % s + '\n' + 'Distance R: '\
                                                          + "%.5f" % pd)
        self.print_message('Testing values: ' + '\n' + 'Pearson R: ' + "%.5f" % p0 + '\n' + \
                          'Spearman R: ' + "%.5f" % s0 + '\n' + 'Distance R: ' + "%.5f" % pd0)
        self.logManager.onLogging(logType='info', message='Testing values: ' + '\n' + 'Pearson R: ' + "%.5f" % p0\
                                                          + '\n' + 'Spearman R: ' + "%.5f" % s0 + '\n' + 'Distance R: '\
                                                          + "%.5f" % pd0)
        _ = myWidgets.popupWindowWithScatterPlotRegression(self, self.setData.data2plot, (xAxis, yAxis))
        self.logManager.onLogging(logType='debug', message='Linear and non linear Regression Wizzard finished')

    # Correlation heat map to combine features using linear model
    def on_heat_map(self, button=None):
        self.logManager.onLogging(logType='debug', message='Correlation heat map to combine features using linear model')
        self.print_message("Heating the map")
        self.logManager.onLogging(logType='info', message='Heating the map')
        popwin = myWidgets.popupWindowWithList(self.listSelectedMethods, sel_method=Gtk.SelectionMode.SINGLE,\
                                               message="X axis measure")
        xAxis = popwin.list_items
        popwin = myWidgets.popupWindowWithList(self.listSelectedMethods, sel_method=Gtk.SelectionMode.SINGLE,\
                                               message="Y axis measure")
        yAxis = popwin.list_items
        Data = self.setData.getHeatMapData(xAxis, yAxis)
        _ = myWidgets.popupWindowWithHeatParameterMap(self, Data, para_rangex=(0.0, 1.), para_rangey=(0.0, 1.),\
                                                      name_axis=(xAxis, yAxis, 'human'))
        self.logManager.onLogging(logType='debug',
                                  message='Correlation heat map to combine features using linear model heated')

    # Opens pdf guide
    def on_guide_clicked(self, button=None):
        self.logManager.onLogging(logType='debug', message='Help?')
        self.print_message("Do you need some help?\nPlease see our help GUIDE in pdf format")
        self.on_about_click()
        opener = "open" if sys.platform == "darwin" else "xdg-open"
        subprocess.call([opener, self.workingPath + '/Help.pdf'])
        self.logManager.onLogging(logType='debug', message='Helpped')

    # Displays the author and version information
    def on_about_click(self, button=None):
        self.logManager.onLogging(logType='debug', message='Aboud click')
        currentTime = myWidgets.getTime()
        self.print_message("iFAS beta-Version Edition 2019-05\n" + \
                          'Copyleft Benhur Ortiz-Jaramillo\n' + \
                          'Postdoctoral researcher at IPI-imec\n' + \
                          'This program is only for research purposes\nand comes with absolutely no warranty')
        self.logManager.onLogging(logType='debug', message='about clicked')

    # updates the window size to the current display
    def update_screen_size(self, button):
        self.logManager.onLogging(logType='debug', message='Updating screen size')
        screen = self.window.get_screen()
        monitors = []
        for m in range(screen.get_n_monitors()):
            monitors.append(screen.get_monitor_geometry(m))
        currentMonitor = screen.get_monitor_at_window(screen.get_active_window())
        monitor_par = monitors[currentMonitor]
        # TODO Screen sizes could be improved by playing with the parameters
        self.videoHeightDefault = int((monitor_par.height - 150) / 2.)
        self.videoWidthDefault = int((monitor_par.width - 50) / 2.)
        self.canvas.set_size_request(int(9. * self.videoWidthDefault / 16.), self.videoHeightDefault)
        self.scrolledWindow.set_property("width-request", int(7. * self.videoWidthDefault / 16.))
        self.scrolledWindow.set_property("height-request", self.videoHeightDefault)
        self.statusBar.set_property("width-request", self.videoWidthDefault)
        self.logManager.onLogging(logType='debug', message='Screen size updated')


# ~ Uncomment the following three lines for standalone running
if __name__ == "__main__":
    Main()
    Gtk.main()
