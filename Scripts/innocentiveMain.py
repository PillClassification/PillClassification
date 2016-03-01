# -*- coding: utf-8 -*-
import dataMngr as dm
import compInfo as ci
import machineLearning as ML
import sklearn.metrics as stats

def makeTestOutput(modelfp, inTestFileName, outTestFileName):
  ML.generateTestOutput(dm.load(modelfp), dm.load(inTestFileName), outTestFileName)

def addHeaders(fp):
  dm.insertFirstRow(fp)

def pickleData(fp):
  inData, outData = dm.preprocess(fp)

def generateTestOutput(modelfp, testOutfp, inData):
  ML.generateTestOutput(dm.load(modelfp), inData, testOutfp)

#------- Preprocess and insert first row ---------------#
# dm.insertFirstRow(ci.originalDataDirectory + "data_HIV_5class.csv")

#--------- Data preprocess to python object and pickle object ------------#
# inData, outData = dm.preprocess(ci.originalDataDirectory + "data_HIV_5class_processed.csv")
#inData = dm.load(ci.outputTestingDirectory + "data_HIV_5class_processed_input.np")
#outData = dm.load(ci.outputTestingDirectory + "data_HIV_5class_processed_output.np")

#----------- Make Testing output with model and data ------------------#
#ML.generateTestOutput(dm.load(ci.modelsDirectory + "rForest_n_10.mod"), inData, ci.outputTestingDirectory + "testingOutput.csv")
#ML.generateTestOutput(dm.load(ci.modelsDirectory + "adaboostrfd_10_t_200_f_10.mod"), inData, ci.outputTestingDirectory + "testingOutputAdaboost.csv")

#------------ Compare Score to actual data ------------------------------#
print stats.accuracy_score(dm.load(ci.outputTestingDirectory + 'data_HIV_5class_processed_output.np'), dm.file2Data(ci.outputTestingDirectory + 'testingOutputAdaboost.csv'))
# can just do sklearn.metrics.accuracy_score
