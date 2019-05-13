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

        # It is a list of strings with the name of the packages
        self.listAvailablePackages = None
        self.getListPackages("./listPackages")
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
        self.selectedRefFile = self.workingPath + '/sample_images/test_ref_0.bmp'
        self.selectedProFile = self.workingPath + '/sample_images/test_pro_0.bmp'
        self.listReferenceFiles = [self.selectedRefFile]
        self.listProcessedFiles = dict([(self.selectedRefFile, [self.selectedProFile])])
        self.setData = imageSample.DataSet(self.listReferenceFiles, self.listProcessedFiles,\
                                           self.selectedPackage, self.listSelectedMethods)
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
        self.createStatusBar()

        # Setting Image Reference space on UI
        drawable_loc = self.builder.get_object("hbox4")
        self.drawableReference = Gtk.Image()
        self.setImageSscrolledWindow(drawable_loc, self.drawableReference)
        # Setting Image Processed space on UI
        self.drawableProcessed = Gtk.Image()
        self.setImageSscrolledWindow(drawable_loc, self.drawableProcessed)
        # Setting Image Difference space on UI
        drawable_loc = self.builder.get_object("box5")
        self.drawableDifference = Gtk.Image()
        self.setImageSscrolledWindow(drawable_loc, self.drawableDifference)

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
        self.createTextView()

        # Setting labels settings
        self.labelProcessedImage = self.builder.get_object("processed_image")
        self.labelProcessedImage.modify_font(Pango.FontDescription('Sans 16'))
        self.labelReferenceImage = self.builder.get_object("label3")
        self.labelReferenceImage.modify_font(Pango.FontDescription('Sans 16'))
        self.labelDifferenceImage = self.builder.get_object("label_dif_image")
        self.labelDifferenceImage.modify_font(Pango.FontDescription('Sans 16'))

        # Setting Stop button
        self.buttonStop = self.builder.get_object("button2")
        self.buttonStop.connect("clicked", self.onStopAll)

        # Setting Start button
        # self.buttonStart = self.builder.get_object("button1")
        # self.buttonStart.connect("clicked", self.executeSelection)

        self.updateImages()
        # Destroying when quiting
        self.window.connect("delete_event", lambda w, e: Gtk.main_quit())
        self.window.connect("destroy", lambda w: Gtk.main_quit())
        # Displaying UI window
        self.window.show_all()
        self.onAboutClick()

    # Getting list of packages aka python scripts, fidelity group
    def getListPackages(self, fileLocation):
        listPackages = []
        with open(fileLocation) as f:
            for line in f:
                line = line.replace('\n', '')
                if self.verifyPackage(line):
                    listPackages.append(line)
                else:
                    try:
                        self.printMessage('Verify your listPackages file. One ore more lines' +\
                                          'could be corrupted or empty: ' + line)
                    except AttributeError:
                        print 'Verify your listPackages file. One ore more lines' + \
                              'could be corrupted or empty: ' + line
        # It is a list of strings with the name of the packages
        self.listAvailablePackages = listPackages

    # Verifying that packages aka python scripts, fidelity groups are located in folder
    def verifyPackage(self, packageName):
        try:
            if not packageName == '':
                result = importlib.import_module(packageName)
                if result:
                    return True
        except ImportError:
            return False

    # function to print messages in the interface textview
    def printMessage(self, message):
        currentTime = myWidgets.getTime()
        self.textBuffer.insert_with_tags(self.textBuffer.get_end_iter(), '\n\n' + currentTime + '\n', self.textTag)
        self.textBuffer.insert(self.textBuffer.get_end_iter(), message + '\n')

    # function to create the status bar
    def createStatusBar(self):
        self.statusBar = self.builder.get_object("levelbar1")
        self.statusBar.set_property("width-request", self.videoWidthDefault)
        self.statusBar.set_min_value(0)
        self.statusBar.set_max_value(100)
        self.statusBar.set_mode(Gtk.LevelBarMode.CONTINUOUS)

    # function to set image on scrolled window
    def setImageSscrolledWindow(self, location, image):
        scrolledwindow = Gtk.ScrolledWindow()
        scrolledwindow.set_hexpand(True)
        scrolledwindow.set_vexpand(True)
        scrolledwindow.set_policy(Gtk.PolicyType.AUTOMATIC, Gtk.PolicyType.AUTOMATIC)
        scrolledwindow.set_property("width-request", self.videoWidthDefault)
        scrolledwindow.set_property("height-request", self.videoHeightDefault)
        scrolledwindow.add_with_viewport(image)
        location.pack_start(scrolledwindow, True, True, 0)

    # function to set the parameters of the text viewer
    def createTextView(self):
        self.scrolledWindow.set_hexpand(True)
        self.scrolledWindow.set_vexpand(True)
        self.scrolledWindow.set_policy(Gtk.PolicyType.AUTOMATIC, Gtk.PolicyType.ALWAYS)
        # TODO Screen sizes could be improved by playing with the parameters
        self.scrolledWindow.set_property("width-request", int(7. * self.videoWidthDefault / 16.))
        self.scrolledWindow.set_property("height-request", self.videoHeightDefault)
        self.scrolledWindowPlace.pack_start(self.scrolledWindow, True, True, 0)
        self.textView.connect("size-allocate", self.autoScroll)
        self.scrolledWindow.add(self.textView)

    # function to autoscroll the textViewer
    def autoScroll(self, *args):
        adj = self.textView.get_vadjustment()
        adj.set_value(adj.get_upper() - adj.get_page_size())

    # function to stop every process
    def onStopAll(self, button=None):
        self.printMessage("Stop Pressed")
        self.stopped = True

    def updateImages(self):
        self.saveTempFiles()
        self.drawableReference.set_from_file('/tmp/temp_ref.png')
        self.drawableProcessed.set_from_file('/tmp/temp_pro.png')
        self.drawableDifference.set_from_file('/tmp/temp_cd.png')
        # Removing temp files
        os.remove("/tmp/temp_ref.png")
        os.remove("/tmp/temp_pro.png")
        os.remove("/tmp/temp_cd.png")
        self.window.show_all()

    # Saving temporary image files to be displayed on the canvas
    def saveTempFiles(self):
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

    # Setting to default UI parameters
    def returnDefault(self):
        self.listAvailablePackages = None
        self.getListPackages("./listPackages")
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
        self.selectedRefFile = self.workingPath + '/sample_images/test_ref_0.bmp'
        self.selectedProFile = self.workingPath + '/sample_images/test_pro_0.bmp'
        self.listReferenceFiles = [self.selectedRefFile]
        self.listProcessedFiles = dict([(self.selectedRefFile, [self.selectedProFile])])
        self.setData = imageSample.DataSet(self.listReferenceFiles, self.listProcessedFiles, \
                                           self.selectedPackage, self.listSelectedMethods)
        self.setImages = self.setData.data[self.selectedRefFile]
        message = 'Returning to default parameters'
        self.printMessage(message)
        self.updateData()
        self.updateImages()

    def updateData(self):
        self.setData.computeData()

    def data2String(self):
        return self.setData.data2String()

    # Changing the list of methods
    def onSelectMeasures(self, button=None, doCompute=True):
        # Measures available in the pyhton script
        popwin = myWidgets.popupWindowWithList(self.listAvailableMethods, sel_method=Gtk.SelectionMode.MULTIPLE,\
                                               message="List of Measures")
        self.listSelectedMethods = popwin.list_items
        if not self.listSelectedMethods:
            self.returnDefault()
        self.currentMeasure = self.listSelectedMethods[0]
        if doCompute:
            self.updateData()
            str2Print = self.data2String()
            self.printMessage('Package: ' + self.selectedPackage + '\n' + 'Difference map: ' + self.currentMeasure +\
                              '\n' + str2Print)
            self.updateImages()

    # Changing the package
    def onSelectPackage(self, button=None, doCompute=True):
        popwin = myWidgets.popupWindowWithList(self.listAvailablePackages, sel_method=Gtk.SelectionMode.SINGLE,\
                                               message="List of Packages")
        self.selectedPackage = popwin.list_items
        if not self.selectedPackage:
            self.returnDefault()
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
                self.updateData()
                self.updateImages()
                self.printMessage('Please select the new set of fidelity measures!')

    # Changing the difference map
    def onChangeDiffMap(self, button):
        popwin = myWidgets.popupWindowWithList(self.listSelectedMethods, sel_method=Gtk.SelectionMode.SINGLE,\
                                               message="List of Measures")
        self.currentMeasure = popwin.list_items
        if not self.currentMeasure:
            self.returnDefault()
        else:
            self.setData.changeDiffMap(self.currentMeasure)
        str2Print = self.data2String()
        self.printMessage('Package: ' + self.selectedPackage + '\n' + 'Difference map: ' + self.currentMeasure +\
                          '\n' + str2Print)
        self.updateImages()

    # Changing the displaying image
    def onImageSetChanged(self, button):
        previousFile = self.selectedRefFile
        popwin = myWidgets.popupWindowWithList(self.listReferenceFiles, sel_method=Gtk.SelectionMode.SINGLE,\
                                               split_=True, message="Reference images")
        self.selectedRefFile = popwin.list_items
        if not self.selectedRefFile:
            self.selectedRefFile = previousFile
            self.printMessage("Error in Reference file selection. Returning to: " + previousFile.split('/')[-1])
        else:
            flag = False
            for ii in self.listReferenceFiles:
                if self.selectedRefFile in ii:
                    newFile = ii
                    flag = True
            if flag:
                self.selectedRefFile = newFile
                self.printMessage("Reference file selection set to: " + newFile.split('/')[-1])
            else:
                self.selectedRefFile = previousFile
                self.printMessage("Error finding Reference. Returning to: " + previousFile.split('/')[-1])
        previousFile = self.selectedProFile
        self.setImages = self.setData.data[self.selectedRefFile]
        listFiles = self.setImages.returnListProcessed()
        popwin = myWidgets.popupWindowWithList(listFiles, sel_method=Gtk.SelectionMode.SINGLE, split_=True,\
                                               message="Processed images")
        self.selectedProFile = popwin.list_items
        if not self.selectedProFile:
            self.selectedProFile = previousFile
            self.printMessage("Error in processed file selection. Returning to: " + previousFile.split('/')[-1])
        else:
            flag = False
            for ii in listFiles:
                if self.selectedProFile in ii:
                    newFile = ii
                    flag = True
            if flag:
                self.selectedProFile = newFile
                self.printMessage("Processed file selection set to: " + newFile.split('/')[-1])
            else:
                self.selectedProFile = previousFile
                self.printMessage("Error finding selection. Returning to sample: " + previousFile.split('/')[-1])
        self.setData.changeProcessedImage(self.selectedRefFile, self.selectedProFile)
        self.updateImages()
        # TODO modify to clear plot place

    # Computing single reference single processed image fidelity using the selected measures
    def singleRefsinglePro(self, button):
        self.returnDefault()
        self.onSelectPackage(doCompute=False)
        self.onSelectMeasures(doCompute=False)
        self.selectedRefFile = myWidgets.load_file(self.window, 'img', 'Select your Reference Image')
        self.selectedProFile = myWidgets.load_file(self.window, 'img', 'Select your Processed Image')
        if self.selectedRefFile and self.selectedProFile:
            self.listReferenceFiles = [self.selectedRefFile]
            self.currentMeasure = self.listSelectedMethods[0]
            self.listProcessedFiles = dict([(self.selectedRefFile, [self.selectedProFile])])
            self.setData = imageSample.DataSet(self.listReferenceFiles, self.listProcessedFiles, \
                                               self.selectedPackage, self.listSelectedMethods)
            self.updateImages()
            str2Print = self.data2String()
            self.printMessage('Package: ' + self.selectedPackage + '\n' + 'Difference map: ' + self.currentMeasure +\
                              '\n' + str2Print)
            self.printMessage("Process Finished!")
        else:
            self.returnDefault()

    # Computing single reference multiple processed image fidelity using the selected measures
    def singleRefmultiplePro(self, button):
        self.returnDefault()
        self.onSelectPackage(doCompute=False)
        self.onSelectMeasures(doCompute=False)
        self.selectedRefFile = myWidgets.load_file(self.window, 'img', 'Select your Reference Image')
        processedFiles = myWidgets.load_file(self.window, 'img', 'Select your Processed Images', multiple=True)
        if self.selectedRefFile and self.selectedProFile:
            self.listReferenceFiles = [self.selectedRefFile]
            self.currentMeasure = self.listSelectedMethods[0]
            self.selectedProFile = processedFiles[0]
            self.listProcessedFiles = dict([(self.selectedRefFile, processedFiles)])
            self.setData = imageSample.DataSet(self.listReferenceFiles, self.listProcessedFiles, \
                                               self.selectedPackage, self.listSelectedMethods)
            self.updateImages()
            str2Print = self.data2String()
            self.printMessage(
                'Package: ' + self.selectedPackage + '\n' + 'Difference map: ' + self.currentMeasure + '\n' + \
                str2Print)
            self.printMessage("Process Finished!")
        else:
            self.returnDefault()

    # Computing multiple reference multiple processed image fidelity using the selected measures
    def multipleRefmultiplePro(self, button=None):
        self.returnDefault()
        self.printMessage('Computing your data!\nThis could take a while depending of your data size.')
        self.onSelectPackage(doCompute=False)
        self.onSelectMeasures(doCompute=False)
        pythonFile = myWidgets.load_file(self.window, 'txt', 'Select your .py file')
        if pythonFile:
            try:
                path2Files, listReferences, dataSet = imp.load_source('module.name', pythonFile).myDataBase()
                path2Files = '/'.join(pythonFile.split('/')[:-1]) + path2Files
            except SyntaxError:
                self.printMessage('File Syntax corrupted. Please verify your file!')
                self.returnDefault()
                return
        else:
            self.returnDefault()
            return
        self.listProcessedFiles = dict()
        self.listReferenceFiles = []
        for ii in listReferences:
            self.listReferenceFiles.append(path2Files + ii)
            self.listProcessedFiles[path2Files + ii] = []
            for jj in dataSet[ii]:
                self.listProcessedFiles[path2Files + ii].append(path2Files + jj)
        self.setData = imageSample.DataSet(self.listReferenceFiles, self.listProcessedFiles, \
                                           self.selectedPackage, self.listSelectedMethods)
        self.currentMeasure = self.listSelectedMethods[0]
        self.selectedRefFile = self.listReferenceFiles[0]
        self.selectedProFile = self.listProcessedFiles[self.selectedRefFile][0]
        self.updateImages()
        str2Print = self.data2String()
        self.printMessage('Package: ' + self.selectedPackage + '\n' + 'Difference map: ' + self.currentMeasure + '\n' + \
                          str2Print)
        self.printMessage("Process Finished!")

    # Save computed data
    def onSaveData(self, button=None):
        popwin = myWidgets.popupWindowWithTextInput('Name your file: ')
        with open(self.workingPath + '/' + popwin.file_name + '.iFAS', 'w') as f:
            pickle.dump([self.selectedPackage, self.listAvailableMethods, self.listSelectedMethods, self.currentMeasure,\
                         self.listReferenceFiles, self.selectedRefFile, self.selectedProFile, self.setData], f)
        self.printMessage("Save completed!")

    # Load precomputed data
    def onLoadData(self, button=None):
        temp = myWidgets.load_file(self.window, 'txt', 'Select your .iFAS file')
        try:
            with open(temp) as f:
                self.selectedPackage, self.listAvailableMethods, self.listSelectedMethods, self.currentMeasure,\
                self.listReferenceFiles, self.selectedRefFile, self.selectedProFile, self.setData = pickle.load(f)
            self.xAxis = self.currentMeasure
            self.yAxis = self.currentMeasure
            self.updateImages()
            str2Print = self.data2String()
            self.printMessage('Package: ' + self.selectedPackage + '\n' + 'Difference map: ' + self.currentMeasure + '\n' + \
                              str2Print)
            self.printMessage("Load Completed!")
        except IOError:
            self.printMessage('File corrupted or not an iFAS file. Please verify your file!')
            self.returnDefault()
            return

    # Plotting data on UI for current reference ImageSet
    def plotCurrentSelection(self):
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
        self.printMessage("Pearson R = " + "%.5f" % p + "\n" + "Spearman R = " + "%.5f" % s + "\n" + \
                          "Distance R = " + "%.5f" % pd)
        # self.printMessage("Pearson R = " + "%.5f" % p + "\n" + "Spearman R = " + "%.5f" % s + "\n" +\
        #                   "Kendall t = " + "%.5f" % t + "\n" + "Distance R = " + "%.5f" % pd)

    # Plotting selected x and y axis
    def plotXmeasureYmeasure(self, button):
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
        self.printMessage("Current x axis: " + self.xAxis)
        self.printMessage("Current y axis: " + self.yAxis)
        self.plotCurrentSelection()

    # Plotting ALL x and y axis
    def onScatterPlot(self, button=None):
        popwin = myWidgets.popupWindowWithList(self.listSelectedMethods, sel_method=Gtk.SelectionMode.SINGLE,\
                                               message="X axis measure")
        xAxis = popwin.list_items
        popwin = myWidgets.popupWindowWithList(self.listSelectedMethods, sel_method=Gtk.SelectionMode.SINGLE,\
                                               message="Y axis measure")
        yAxis = popwin.list_items
        Data = self.setData.getDictTableperRef(xAxis, yAxis)
        _ = myWidgets.popupWindowWithScatterPlot(self, Data, (xAxis, yAxis))

    # Plotting Multiple distortion types according to selected by user
    def onMultipleDistortionPlot(self, button=None, data=None):
        popwin = myWidgets.popupWindowWithTextInput('Number of distortions: ')
        if not popwin.file_name:
            self.printMessage(popwin.file_name + " is not integer!")
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

    # Adds precomputed measure. Name scores is reserved for human scores
    def addPrecomputedMeasure(self, button=None):
        temp = myWidgets.load_file(self.window, 'txt', 'Select your measure file')
        if temp:
            popwin = myWidgets.popupWindowWithTextInput('Name your measure [name it \'human\'\nfor subjective scores]: ')
            self.setData.setPrecomputedData(temp, popwin.file_name)
            if not (popwin.file_name in self.listSelectedMethods):
                self.listSelectedMethods.append(popwin.file_name)
            self.printMessage("File " + temp + " loaded.")
        else:
            self.printMessage("Error finding data File. Please Try again")

    # Adds new python script to the configuration file
    def addNewPythonScript(self, button=None):
        popwin = myWidgets.popupWindowWithTextInput()
        self.listAvailablePackages.append(popwin.file_name)
        file('./listPackages', 'w').write("\n".join(self.listAvailablePackages) + "\n")
        self.getListPackages("./listPackages")
        self.printMessage('New fidelity pack add to the default packages.')

    # Plots the correlation bar for the selected measures and the 'human' data if available
    def globalCorrelationBarPlot(self, button=None):
        correlations = {}
        for ii in self.listSelectedMethods:
            if ii != 'human':
                Data = self.setData.getDataMatrix(ii, 'human')
                p, s, t, pd = myUtilities.compute_1dcorrelatiosn(Data[:, 0], Data[:, 1])
                correlations[ii] = np.abs(np.asarray([p, s, t, pd]))
                self.printMessage(ii + '\n' + 'Pearson R: ' + "%.5f" % p + '\n' + 'Spearman R: ' + "%.5f" % s \
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
        self.printMessage(message)
        _ = myWidgets.popupWindowWithBarPlot(self, correlations)

    # Box plot of the correlations per source sample
    def boxPlotxReference(self, button=None):
        correlations = {}
        for ii in self.listSelectedMethods:
            if ii != 'human':
                correlations[ii] = []
                for jj in self.listReferenceFiles:
                    Data = self.setData.getDataMatrixReference(ii, 'human', jj)
                    p, s, t, pd = myUtilities.compute_1dcorrelatiosn(Data[:, 0], Data[:, 1])
                    correlations[ii].append([p, s, t, pd])
                    self.printMessage(ii + '\n' + jj.split('/')[-1] + '\n' + 'Pearson R: ' + "%.5f" % p + '\n' + \
                                      'Spearman R: ' + "%.5f" % s + '\n' + 'Kendall t: ' + "%.5f" % t + '\n' + \
                                      'Distance R: ' + "%.5f" % pd)
                    # self.printMessage(ii + '\n' + jj.split('/')[-1] + '\n' + 'Pearson R: ' + "%.5f" % p + '\n' + \
                    #                   'Spearman R: ' + "%.5f" % s + '\n' + 'Kendall t: ' + "%.5f" % t + '\n' + \
                    #                   'Distance R: ' + "%.5f" % pd)
                correlations[ii] = np.asarray(correlations[ii])
        _ = myWidgets.popupWindowWithBoxPlot(self, correlations)

    # Histogram of the content features of the references
    def histogramContentFeatures(self, button=None):
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

    # Linear and non linear Regression Wizzard
    def onRegression(self, button=None, data=None):
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
        self.printMessage("The optimal parameters for function\n" + optimizationTools.fun_text(data) + "\n" + messagea)
        self.printMessage('Training values: ' + '\n' + 'Pearson R: ' + "%.5f" % p + '\n' + \
                          'Spearman R: ' + "%.5f" % s + '\n' + 'Distance R: ' + "%.5f" % pd)
        self.printMessage('Testing values: ' + '\n' + 'Pearson R: ' + "%.5f" % p0 + '\n' + \
                          'Spearman R: ' + "%.5f" % s0 + '\n' + 'Distance R: ' + "%.5f" % pd0)
        _ = myWidgets.popupWindowWithScatterPlotRegression(self, self.setData.data2plot, (xAxis, yAxis))

    # Correlation heat map to combine features using linear model
    def onHeatMap(self, button=None):
        self.printMessage("Heating the map")
        popwin = myWidgets.popupWindowWithList(self.listSelectedMethods, sel_method=Gtk.SelectionMode.SINGLE,\
                                               message="X axis measure")
        xAxis = popwin.list_items
        popwin = myWidgets.popupWindowWithList(self.listSelectedMethods, sel_method=Gtk.SelectionMode.SINGLE,\
                                               message="Y axis measure")
        yAxis = popwin.list_items
        Data = self.setData.getHeatMapData(xAxis, yAxis)
        _ = myWidgets.popupWindowWithHeatParameterMap(self, Data, para_rangex=(0.0, 1.), para_rangey=(0.0, 1.),\
                                                      name_axis=(xAxis, yAxis, 'human'))

    # Opens pdf guide
    def onGuideClicked(self, button=None):
        self.printMessage("Do you need some help?\nPlease see our help GUIDE in pdf format")
        self.onAboutClick()
        opener = "open" if sys.platform == "darwin" else "xdg-open"
        subprocess.call([opener, self.workingPath + '/Help.pdf'])

    # Displays the author and version information
    def onAboutClick(self, button=None):
        currentTime = myWidgets.getTime()
        self.printMessage("iFAS beta-Version Edition 2019-05\n" + \
                          'Copyleft Benhur Ortiz-Jaramillo\n' + \
                          'Postdoctoral researcher at IPI-imec\n' + \
                          'This program is only for research purposes\nand comes with absolutely no warranty')

    # updates the window size to the current display
    def updateScreenSize(self, button):
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


# ~ Uncomment the following three lines for standalone running
if __name__ == "__main__":
    Main()
    Gtk.main()
