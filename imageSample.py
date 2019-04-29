# !/usr/bin/env python2.7
# Importing necessary packages
import numpy as np
from scipy import ndimage
from scipy import misc

class ImageSet(object):
    # Variable initialization
    # referenceImageLocation: string with folder location to the reference image
    # listProcessedImageLocations: list of strings with folder locations to the reference images
    def __init__(self, referenceImageLocation, listProcessedImageLocations):
        self.refLocation = referenceImageLocation
        self.proLocation = listProcessedImageLocations
        self.NprocessedImages = len(listProcessedImageLocations)