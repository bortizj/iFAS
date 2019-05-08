# !/usr/bin/env python2.7
# Importing necessary packages
import numpy as np
from scipy import ndimage
import importlib


# Python object contains the reference image, processed image and the necessary functions to compute the
# values
class ImageSet(object):
    # Variable initialization
    # referenceImageLocation: string with folder location to the reference image
    # listProcessedImageLocations: list of strings with folder locations to the reference images
    # listFunctions: list of strings with the selected functions
    def __init__(self, referenceImageLocation, listProcessedImageLocations, package, listFunctions):
        # Initializing the current image file locations and the list of processed images
        self.refLocation = referenceImageLocation
        self.proLocation = listProcessedImageLocations[0]
        self.proFileLocations = listProcessedImageLocations
        # Initializing the current images and total the number of images as well the image differences
        self.NprocessedImages = int(len(listProcessedImageLocations))
        self.imageReference = np.double(ndimage.imread(referenceImageLocation))
        self.imageProcessed = np.double(ndimage.imread(self.proLocation))
        self.imageDifference = np.sum(np.abs(self.imageReference - self.imageProcessed), axis=2)
        # Initializing the list of functions and its python file script
        self.package = package
        self.listMeasures = listFunctions
        self.Nmeasures = int(len(listFunctions))
        self.currentMeasure = listFunctions[0]
        # Initializing data dictionary
        self.data = dict()
        self.contentFeatures = dict()
        self.regressionModel = None
        for ii in self.listMeasures:
            self.data[ii] = np.zeros((self.NprocessedImages, 1))
        self.computeData()

    def computeData(self):
        # package is actually the python module not only the string
        package = importlib.import_module(self.package)
        # Computing the list of measures between the reference and the list of processed images
        for ii in self.listMeasures:
            # Verifying that package has the function ii from the list of measures
            if hasattr(package, ii):
                for jj in range(self.NprocessedImages):
                    currentFile = self.proFileLocations[jj]
                    # compute current values for the current image and measure selection and store the image difference
                    if ii == self.currentMeasure and currentFile == self.proLocation:
                        self.imageProcessed = np.double(ndimage.imread(currentFile))
                        self.imageDifference, val = getattr(package, ii)(self.imageReference, self.imageProcessed)
                    # else compute current values and discard the image difference
                    else:
                        _, val = getattr(package, ii)(self.imageReference, np.double(ndimage.imread(currentFile)))
                    self.data[ii][jj] = val
            else:
                print "Current package named " + self.package + " has not attribute " + ii

    # ImageLocation is a string
    def changeProcessedImage(self, ImageLocation):
        if ImageLocation in self.proFileLocations:
            self.proLocation = ImageLocation
            self.imageProcessed = np.double(ndimage.imread(ImageLocation))
            self.changeDifferenceImage(self.currentMeasure)
            print "Processed image set to " + ImageLocation
        else:
            print "Current Image set has not processed image named " + ImageLocation

    # DifferenceImage is a string
    def changeDifferenceImage(self, DifferenceImage):
        package = importlib.import_module(self.package)
        if hasattr(package, DifferenceImage):
            self.currentMeasure = DifferenceImage
            self.imageDifference, _ = getattr(package, self.currentMeasure)(self.imageReference, self.imageProcessed)
            self.imageDifference[np.isnan(self.imageDifference)] = 0.
            self.imageDifference[np.isinf(self.imageDifference)] = 0.
            print "Image difference set to " + self.currentMeasure
        else:
            print "Current package named " + self.package + " has not attribute " + DifferenceImage

    # Adds a numpy array of data (n x 1) to the dictionary data with string name measureName
    def addPrecomputedData(self, measureName, data):
        if not (self.data.has_key(measureName) or self.NprocessedImages != data.size):
            self.data[measureName] = data.reshape((data.size, 1))
            if not (measureName in self.listMeasures):
                self.listMeasures.append(measureName)
            print "Attribute named " + measureName + " has been added. This data has not been saved!"
        elif self.NprocessedImages != data.size:
            print "Data size does not agree with input size " + str(self.NprocessedImages) + " != " + str(data.size)
        else:
            print "Current Image set has already an attribute named " + self.currentMeasure

    # changing and recomputing the data based on the selection
    def changeMeasures(self, package, listFunctions):
        self.package = package
        self.listMeasures = listFunctions
        self.Nmeasures = int(len(listFunctions))
        self.currentMeasure = listFunctions[0]
        # Initializing data dictionary
        self.data = ()
        self.contentFeatures = dict()
        for ii in self.listMeasures:
            self.data[ii] = np.zeros((self.NprocessedImages, 1))
        self.computeData()

    # Return data as numpy array
    def returnVector(self, measure=None):
        if measure is None:
            data = []
            name = []
            for ii in self.listMeasures:
                data.append(self.data[ii])
                name.append(self.proFileLocations)
        else:
            data = self.data[measure]
            name = self.proFileLocations
        return name, data

    # Return data as numpy array for specific list of processed images
    def returnVectorProcessed(self, measure, processedList):
        data = []
        for ii in processedList:
            data.append(self.data[measure][ii])
        return np.array(data)

    # Return a value for specific processed image
    def returnValueProcessed(self, measure, processedFile):
        idx = self.proFileLocations.index(processedFile)
        return self.data[measure][idx]

    # Return the list of processed images
    def returnListProcessed(self):
        return self.proFileLocations

    # Content related features to the reference image
    def computeContentFeatures(self, package, features):
        self.contentFeatures = dict()
        for ii in features:
            self.contentFeatures[ii] = getattr(package, ii)(self.imageReference)

    # regression model
    def setRegressionModel(self, model=None):
        self.regressionModel = model
