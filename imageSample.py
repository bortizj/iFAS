# !/usr/bin/env python2.7
# Importing necessary packages
import numpy as np
from scipy import ndimage
import importlib
import optimizationTools
import myUtilities


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

    # Computes the data for the reference image and its corresponding processed images
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
        self.data = dict()
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
            if ii == 'colorfulness':
                _, self.contentFeatures[ii] = getattr(package, ii)(self.imageReference)
            else:
                self.contentFeatures[ii] = getattr(package, ii)(self.imageReference)

    # regression model
    def setRegressionModel(self, model=None):
        self.regressionModel = model


# Python object contains the database of images reference and processed images and the necessary functions to compute
# the values
class DataSet(object):
    # Variable initialization
    # referenceImageLocation: string with folder location to the current reference image
    # listReferenceImages: list of reference images
    # listProcessedImageLocations: dictionary with folder locations to the processed images
    # listFunctions: list of strings with the selected functions
    def __init__(self, listreferenceImageLocations, listProcessedImageLocations, package, listFunctions):
        # Initializing the current image file locations and the list of reference as well as the processed images
        self.refLocation = listreferenceImageLocations[0]
        self.proLocation = listProcessedImageLocations[self.refLocation][0]
        self.refFileLocations = listreferenceImageLocations
        self.proFileLocations = listProcessedImageLocations
        # Initializing the current images and total the number of images as well the image differences
        self.NreferenceImages = int(len(listreferenceImageLocations))
        self.NprocessedImages = int(len(listProcessedImageLocations))
        self.imageReference = np.double(ndimage.imread(self.refLocation))
        self.imageProcessed = np.double(ndimage.imread(self.proLocation))
        self.imageDifference = np.sum(np.abs(self.imageReference - self.imageProcessed), axis=2)
        # Initializing the list of functions and its python file script
        self.package = package
        self.listMeasures = listFunctions
        self.Nmeasures = int(len(listFunctions))
        self.currentMeasure = listFunctions[0]
        # Initializing data dictionary
        self.data = dict()
        self.model = None
        self.modelName = None
        self.data2plot = dict()
        self.computeData()

    # Computes the data for thw whole dataset
    def computeData(self):
        count = 0
        for ii in self.refFileLocations:
            self.data[ii] = ImageSet(ii, self.proFileLocations[ii], self.package, self.listMeasures)
            count += 1
            print 100. * count / len(self.refFileLocations)

    # Convert the computed data to a string to be displayed by reference file
    def data2String(self):
        str2Print = ''
        for ii in self.refFileLocations:
            str2Print += ii.split('/')[-1] + '\n'
            name, data = self.data[ii].returnVector()
            strOut = myUtilities.conver2String(name, data, self.listMeasures)
            str2Print += strOut
        return str2Print

    # Changes the difference map for each reference image
    def changeDiffMap(self, measure):
        for ii in self.refFileLocations:
            self.data[ii].changeDifferenceImage(measure)

    # Changes the processed image for the selected reference
    def changeProcessedImage(self, refImageLocation, proImageLocation):
        self.data[refImageLocation].changeProcessedImage(proImageLocation)

    # Gets Dictionary of a numpy array for each reference given the measure names xAxis and yAxis
    def getDictTableperRef(self, xAxis, yAxis):
        Data = {}
        for ii in self.refFileLocations:
            _, xAxisValues = self.data[ii].returnVector(measure=xAxis)
            _, yAxisValues = self.data[ii].returnVector(measure=yAxis)
            Data[ii.split('/')[-1]] = np.hstack((xAxisValues, yAxisValues))
        return Data

    # Returns the list of processed files for all references
    def getListProcessed(self):
        listProcessedFiles = []
        for ii in sorted(self.refFileLocations):
            subList = self.data[ii].returnListProcessed()
            for jj in sorted(subList):
                listProcessedFiles.append(jj)
        return listProcessedFiles

    # returns a numpy array with the measures xAxis and yAxis and the las colu;n is the label in image distortion
    def getDataLabel(self, Ndistortions, listImagesbyDistortions, xAxis, yAxis):
        Data = []
        for ii in self.refFileLocations:
            for jj in self.data[ii].returnListProcessed():
                for kk in range(Ndistortions):
                    for ll in listImagesbyDistortions[kk]:
                        if ll in jj:
                            xValue = self.data[ii].returnValueProcessed(xAxis, jj)[0]
                            yValue = self.data[ii].returnValueProcessed(yAxis, jj)[0]
                            Data.append([xValue, yValue, kk])
        return Data

    # Set precompputed data such as human data in the whole database
    def setPrecomputedData(self, datafile, measureName):
        for ii in self.refFileLocations:
            listProcessed = self.data[ii].returnListProcessed()
            data = np.nan * np.ones((len(listProcessed), 1))
            with open(datafile) as f:
                for line in f:
                    temp_string = line.split()
                    if temp_string[0] in ii:
                        for jj in range(len(listProcessed)):
                            if temp_string[1] in listProcessed[jj]:
                                data[jj] = float(temp_string[2])
                if np.any(np.isnan(data)):
                    print str(int(np.sum(np.isnan(data)))) + ' images not found. Verify your file.'
                    return
                else:
                    self.data[ii].addPrecomputedData(measureName, data)
        if not (measureName in self.listMeasures):
            self.listMeasures.append(measureName)
            self.Nmeasures = int(len(self.listMeasures))
            self.currentMeasure = self.listMeasures[0]
        print "Measure " + measureName + " loaded."

    # get the matrix of the whole dta for specific pair of measures xAxis and yAxis
    def getDataMatrix(self, xAxis, yAxis):
        Data = []
        for jj in self.refFileLocations:
            for kk in self.data[jj].returnListProcessed():
                xValue = self.data[jj].returnValueProcessed(xAxis, kk)[0]
                yValue = self.data[jj].returnValueProcessed(yAxis, kk)[0]
                Data.append([xValue, yValue])
        Data = np.asarray(Data)
        return Data

    # get the matrix for specific measures xAxis and yAxis and reference
    def getDataMatrixReference(self, xAxis, yAxis, reference):
        _, xValues = self.data[reference].returnVector(measure=xAxis)
        _, yValues = self.data[reference].returnVector(measure=yAxis)
        Data = np.hstack((xValues, yValues))
        Data = np.asarray(Data)
        return Data

    # Compute the content features for all references
    def computeContentFeatures(self, contentFeatures, listFeatures):
        for ii in self.refFileLocations:
            self.data[ii].computeContentFeatures(contentFeatures, listFeatures)

    # get the cintent features for specific reference and content feature
    def getContentFeature(self, contentFeature, reference):
        return self.data[reference].contentFeatures[contentFeature]

    # Computes a regression model given a list of train and test samples, measures xAxis and yAxis and a model
    def regressionModel(self, listTrain, listTest, xAxis, yAxis, model):
        self.modelName = model
        x = []
        y = []
        for ii in listTrain:
            _, xValues = self.data[ii].returnVector(measure=xAxis)
            _, yValues = self.data[ii].returnVector(measure=yAxis)
            x += xValues.T.tolist()[0]
            y += yValues.T.tolist()[0]
        x = np.array(x)
        y = np.array(y)
        self.model = optimizationTools.optimize_function(x, y, fun_type=model)
        y_est = optimizationTools.gen_data(x, self.model, fun_type=model)
        xTest = []
        yTest = []
        for ii in listTest:
            _, xValues = self.data[ii].returnVector(measure=xAxis)
            _, yValues = self.data[ii].returnVector(measure=yAxis)
            xTest += xValues.T.tolist()[0]
            yTest += yValues.T.tolist()[0]
        x_test = np.array(xTest)
        y_test = np.array(yTest)
        y_est_test = optimizationTools.gen_data(x_test, self.model, fun_type=model)
        for ii in listTrain:
            self.data[ii].setRegressionModel(self.model)
        for ii in listTest:
            self.data[ii].setRegressionModel(self.model)
        self.data2plot = dict()
        xfull = np.linspace(np.minimum(np.min(x), np.min(x_test)), np.maximum(np.max(x), np.max(x_test)), 100)
        # Regression line data
        self.data2plot['rl'] = np.transpose(np.vstack((xfull, optimizationTools.gen_data(xfull, self.model, fun_type=model))))
        # Train data
        self.data2plot['t'] = np.transpose(np.vstack((x, y)))
        # Test data
        self.data2plot['s'] = np.transpose(np.vstack((x_test, y_test)))
        return self.model, myUtilities.compute_1dcorrelatiosn(y, y_est), myUtilities.compute_1dcorrelatiosn(y_test, y_est_test)

    # estimates values given the model and the input data
    def estimateModelValues(self, x, model):
        return optimizationTools.gen_data(x, self.model, fun_type=self.modelName)

    # returns dictionary of data to be plotted as heat map
    def getHeatMapData(self, xAxis, yAxis):
        x = []
        y = []
        h = []
        for ii in self.refFileLocations:
            _, xValues = self.data[ii].returnVector(measure=xAxis)
            _, yValues = self.data[ii].returnVector(measure=yAxis)
            _, hValues = self.data[ii].returnVector(measure='human')
            x += xValues.T.tolist()[0]
            y += yValues.T.tolist()[0]
            h += hValues.T.tolist()[0]
        values = np.vstack((np.array(x), np.array(y), np.array(h))).T
        return {xAxis: values[:, 0], yAxis: values[:, 1], 'human': values[:, 2]}