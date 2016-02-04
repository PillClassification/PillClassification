# -*- coding: utf-8 -*-
import pandas as pd
import csv
import numpy as np
import cPickle as pickle
import compInfo as ci
from os import path
from dataInfo import columnInfo
from sklearn.feature_extraction import DictVectorizer as DV
from compInfo import outputDataDirectory
from os.path import splitext
def preprocess(fileName, train=True):
    df = file2Dataframe(fileName)
    inputArr, outputArr = processDataframeForNP(df, train)
    inputFileName, outputFileName = makeFileName(fileName)
    
    if train:
        save(outputFileName, outputArr)
        
    save(inputFileName, inputArr)
    
    return inputArr, outputArr

def processDiscDataframe(df):    
    vectorizer = DV( sparse = False )
    df = df.to_dict('records')
    numpyData = vectorizer.fit_transform(df)
    return numpyData

def makeFitFromDF(df):
    vectorizer = DV(sparse = False)
    df = df.to_dict('records')
    return vectorizer.fit(df)

def processTesting(dfTesting, dfTraining):
    testInputDiscrete = dfTesting[columnInfo['discrete']]
    testInputContinuous = dfTesting[columnInfo['continous']]
    
    fit = makeFitFromDF(dfTraining[columnInfo['discrete']])
    testInputDiscreteDict = testInputDiscrete.to_dict('records')

    testInputDiscrete = fit.transform(testInputDiscreteDict)
    testInputContinuous = testInputContinuous.as_matrix()

    inputData = np.concatenate(( testInputContinuous, testInputDiscrete ), axis=1)
    return inputData

def makeDFFromCSV():
    print "working"
    dfTraining = file2Dataframe(ci.originalDataDirectory + ci.trainingPathCSV)
    dfTesting = file2Dataframe(ci.originalDataDirectory + ci.testingInputPathCSV)
    print "still working"
    save(outputDataDirectory + "dataFrameTraining.df", dfTraining)
    save(outputDataDirectory + "dataFrameTesting.df", dfTesting)
    print "stilll work"    
    
def processDataframeForNP(df, train=True):
    df = df.drop(columnInfo['extraneous'], axis=1) #Unecessary Info
    
    outputData = None
    if train:
        outputData = df[columnInfo['target']]
        outputData = outputData.as_matrix()
    df = df.drop(columnInfo['target'], axis=1)
    
    inputDiscrete = df[columnInfo['discrete']]
    inputContinuous = df[columnInfo['continous']]
    
    inputDiscrete = processDiscDataframe(inputDiscrete)
    inputContinuous = inputContinuous.as_matrix()
    
    inputData = np.concatenate(( inputContinuous, inputDiscrete ), axis=1)
    return inputData, outputData

def file2Dataframe(fileName):
    df = pd.read_csv(fileName, engine='c')
    return df

def save(fileName, obj):
    pickle.dump( obj, open( fileName, "wb" ) )
    pass

def load(fileName):
    return pickle.load( open( fileName, "rb" ) )

def seperateTestInputOutput(df):
    pass

def makeFileName(inFileName, train=True):
    base=path.basename(inFileName)
    fileName, ext = path.splitext(base)
    
    outputFileName = None
    if train:
        outputFileName = outputDataDirectory + fileName + '_output.np'
    
    inputFileName = outputDataDirectory + fileName + '_input.np'
    
    return inputFileName, outputFileName

def writeToFile(data, writer):
    for el in data:
        writer.writerow([el])

def writeToFilePath(data, filePath):
    writer = csv.writer(open(filePath, 'wb'))
    writeToFile(data, writer)

def saveInputOutput(inA, outA, inFilePath, outFilePath):
    inputFile = csv.writer(open(inFilePath, 'wb'))
    outputFile = csv.writer(open(outFilePath, 'wb'))
    writeToFile(inA, inputFile)
    writeToFile(outA, outputFile)

def file2Data(filePath):
    fileData = []
    with open(filePath, 'rb') as fp:
        reader = csv.reader(fp)
        for row in reader:
            fileData.append(row)
    return fileData
    
def saveModelInfo(fileName, modelInfo):
    #modelInfo is a dict with the following keys:
    #modelName, modelParameters, percentYesCorrect 
    #percentNoCorrect, numInstances, savedModelFileName
    #it will be printed in that order
    order = ['modelName', 'modelParameters', 'percentYesCorrect',
    'percentNoCorrect', 'numInstances', 'savedModelFileName']
    csvRow = [modelInfo[x] for x in order]
    with open(fileName, 'ab') as f:
        f.write(csvRow)

def makeTestingDF():
    makeDFFromCSV()
    trainingDF = load(ci.outputDataDirectory + 'dataFrameTraining.df')
    testingDF = load(ci.outputDataDirectory + 'dataFrameTesting.df')
    testingData = processTesting(testingDF, trainingDF)
    save(ci.outputDataDirectory + ci.testingInputPathFitted, testingData)
    print testingData.shape

def saveCSVasNP(csvFile):
    csv = np.genfromtxt (csvFile, delimiter=",")
    fileName = splitext(csvFile)[0] + ".np"
    save(fileName, csv)