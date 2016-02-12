# -*- coding: utf-8 -*-
import dataMngr as dm
import compInfo as ci
import machineLearning as ML

def makeTestOutput(modelfp, inTestFileName, outTestFileName):
  ML.generateTestOutput(dm.load(modelfp), dm.load(inTestFileName), outTestFileName)


#------- Preprocess and insert first row ---------------#
dm.insertFirstRow(ci.originalDataDirectory + "data_HIV_5class.csv")

#--------- Data preprocess to python object and pickle object ------------#
inData, outData = dm.preprocess(ci.originalDataDirectory + "data_HIV_5class_processed.csv")

#----------- Make Testing output with model and data ------------------#
ML.generateTestOutput(dm.load(ci.modelsDirectory + "rForest_n_10.mod"), inData, ci.outputTestingDirectory + "testingOutput.csv")

#------------ Compare Score to actual data ------------------------------#
print ML.percentCorrect(dm.load(ci.outputTestingDirectory + 'data_HIV_5class_processed_output.np'), dm.file2Data(ci.outputTestingDirectory + 'testingOutput.csv'))
