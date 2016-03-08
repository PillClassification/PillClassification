# -*- coding: utf-8 -*-
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import cross_val_score
import dataMngr as dm
import compInfo as ci
import sklearn
from sklearn import metrics
from sklearn import cross_validation as cv

def generateUnfittedModel(model, modelArrParams, modelDictParams):
    if modelArrParams is None:
      modelArrParams = []
    if modelDictParams is None:
      modelDictParams = {}
    return model(*modelArrParams, **modelDictParams)

# only generated model, doesn't attempt to fit model
def generateModel(model, modelArrParams, modelDictParams, inData, outData):
    if modelArrParams is None:
        modelArrParams = []
    if modelDictParams is None:
        modelDictParams = {}
    standardModel = model(*modelArrParams, **modelDictParams)
    return standardModel.fit(inData, outData.ravel())

def percentCorrect(modelData, outputData):
    labeledCorrectlyCount = 0
    labeledIncorrectlyCount = 0
    for i in xrange(len(modelData)):
        if modelData[i] == outputData[i]:
            labeledCorrectlyCount += 1
        else:
            labeledIncorrectlyCount += 1
    return float(labeledCorrectlyCount)/len(modelData)

def generateModelScores(generatedModel, inData, outData, modelFileName, statisticsFileName):
  predicted = generatedModel.predict(inData)
  pc = percentCorrect(predicted, outData)
  saveModelAndScores(modelFileName, generatedModel, statisticsFileName, pc)
  return pc

def saveModelAndScores(modelFileName, generatedModel, statisticsFileName, scores):
  dm.save(modelFileName, generatedModel)
  dm.save(statisticsFileName, scores)

def modelDriver(model, modelArrParams, modelDictParams, inData, outData):
  generatedModel = generateModel(model, modelArrParams, modelDictParams, inData, outData)
  pc = generateModelScores(generatedModel, inData, outData, "RF.np", "RF_Stats.np")
  print pc

def generateTestOutput(generatedModel, inData, outDataName):
  outData = generatedModel.predict(inData)
  dm.writeToFilePath(outData, outDataName)

def checkCrossValidation(inDatafp, outDatafp, modelInfo, numFolds):
  inData = dm.load(inDatafp)
  outData = dm.load(outDatafp).ravel()
  model = generateUnfittedModel(modelInfo["model"], modelInfo["modelArrParameters"], modelInfo["modelDictParameters"])
  return cv.cross_val_score(model, inData, outData, cv=numFolds)
