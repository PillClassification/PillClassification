# -*- coding: utf-8 -*-
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import cross_val_score
import dataMngr as dm
import compInfo as ci
import sklearn

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

def fixContNaNColoumns(data, col, rows):
  for row in rows:
    data[row][col] = 0
  sumCol = sum(data[:, col])
  numRows, _ = data.shape
  numRows = numRows - len(rows)
  avg = sumCol/numRows
  for row in rows:
    data[row][col] = avg
  return data

def oneShotFixContNaNColoumns():
  print 'starting'
  fileName = ci.outputDataDirectory + ci.testingInputPathFitted
  testInput = dm.load(fileName)
  print 'loaded'
  print testInput.shape
  testInput = fixContNaNColoumns(testInput, 6, [425936, 617750, 734618])
  dm.save(ci.outputDataDirectory + ci.testingInputPathFixed, testInput)
  print 'done'

