# -*- coding: utf-8 -*-
import dataMngr as dm
import compInfo as ci
import machineLearning as ML

def makeTestOutput(modelfp, inTestFileName, outTestFileName):
  ML.generateTestOutput(dm.load(modelfp), dm.load(inTestFileName), outTestFileName)


#dm.insertFirstRow(ci.originalDataDirectory + "data_HIV_5class.csv")
inData, outData = dm.preprocess(ci.originalDataDirectory + "data_HIV_5class_processed.csv")
ML.generateTestOutput(dm.load(ci.modelsDirectory + "rForest_n_10.mod"), inData, ci.outputTestingDirectory + "testingOutput.csv")
