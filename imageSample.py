# !/usr/bin/env python2.7
# Importing necessary packages
import numpy as np
from scipy import ndimage
import importlib


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
        self.data = {}
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
            print "Image difference set to " + self.currentMeasure
        else:
            print "Current package named " + self.package + " has not attribute " + DifferenceImage

    # Adds a numpy array of data (n x 1) to the dictionary data with string name measureName
    def addPrecomputedData(self, measureName, data):
        if not (self.data.has_key(measureName) or self.NprocessedImages != data.size):
            self.data[measureName] = data
            print "Attribute named " + self.currentMeasure + " has been set. This has data has not been saved!"
        elif self.NprocessedImages != data.size:
            print "Data size does not agree with input size " + str(self.NprocessedImages) + " != " + str(data.size)
        else:
            print "Current Image set has already an attribute named " + self.currentMeasure


# ~ Uncomment the following three lines for standalone running
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    refimage = "/media/bortiz/Data/Full_Papers/IFAApplications/documents/iFAS/sample_images/i03.bmp"
    proimages = ["/media/bortiz/Data/Full_Papers/IFAApplications/documents/iFAS/sample_images/i03/i03_07_1.bmp",\
                 "/media/bortiz/Data/Full_Papers/IFAApplications/documents/iFAS/sample_images/i03/i03_07_2.bmp",\
                 "/media/bortiz/Data/Full_Papers/IFAApplications/documents/iFAS/sample_images/i03/i03_07_3.bmp",\
                 "/media/bortiz/Data/Full_Papers/IFAApplications/documents/iFAS/sample_images/i03/i03_07_4.bmp",\
                 "/media/bortiz/Data/Full_Papers/IFAApplications/documents/iFAS/sample_images/i03/i03_07_5.bmp"]
    listfunctions = ["psnr", "ssim", "SSIM", "blocking_diff", "noise_diff", "blur_diff", "epsnr"]
    imgSet = ImageSet(refimage, proimages, "miselaneusPack", listfunctions)

    imgSet.changeProcessedImage("dildo")
    imgSet.changeProcessedImage("/media/bortiz/Data/Full_Papers/IFAApplications/documents/iFAS/sample_images/i03/i03_07_3.bmp")
    imgSet.changeDifferenceImage("ssim")
    imgSet.changeDifferenceImage("dildo")
    imgSet.addPrecomputedData("dmos", np.ones((4, 1)))
    imgSet.addPrecomputedData("psnr", np.ones((5, 1)))
    imgSet.addPrecomputedData("dmos", np.ones((5, 1)))
