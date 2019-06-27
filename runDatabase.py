#!/usr/bin/env python2.7
# Importing necessary packages
import importlib, imp, os, pickle, inspect
import imageSample

def multipleRefmultiplePro(pythonFile, selectedPackage):
    workingPath = os.path.dirname(os.path.abspath(__file__))
    print 'Computing your data!\nThis could take a while depending of your data size.'
    package = importlib.import_module(selectedPackage)
    listAvailableMethods = []
    for ii in range(len(inspect.getmembers(package, inspect.isfunction))):
        listAvailableMethods.append(inspect.getmembers(package, inspect.isfunction)[ii][0])
    try:
        path2Files, listReferences, dataSet = imp.load_source('module.name', pythonFile).myDataBase()
        path2Files = '/'.join(pythonFile.split('/')[:-1]) + path2Files
    except SyntaxError:
        print 'File Syntax corrupted. Please verify your file!'
        return
    listProcessedFiles = dict()
    listReferenceFiles = []
    for ii in listReferences:
        listReferenceFiles.append(path2Files + ii)
        listProcessedFiles[path2Files + ii] = []
        for jj in dataSet[ii]:
            listProcessedFiles[path2Files + ii].append(path2Files + jj)
    setData = imageSample.DataSet(listReferenceFiles, listProcessedFiles, selectedPackage, listAvailableMethods)
    currentMeasure = listAvailableMethods[0]
    selectedRefFile = listReferenceFiles[0]
    selectedProFile = listProcessedFiles[selectedRefFile][0]
    with open(workingPath + '/' + pythonFile.split('/')[-1].split('.')[0] + '.iFAS', 'w') as f:
        pickle.dump([selectedPackage, listAvailableMethods, listAvailableMethods, currentMeasure, \
                     listReferenceFiles, selectedRefFile, selectedProFile, setData], f)
    print "Process Finished!"


# ~ Uncomment the following three lines for standalone running
if __name__ == "__main__":
    selectedPackage = 'texturePack'
    pythonFile = '/media/bortiz/DATA/gitProjects/multipleRefmultipleProTextureTest.py'
    multipleRefmultiplePro(pythonFile, selectedPackage)
    x = raw_input('Press Enter to Finish')
