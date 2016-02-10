from dataMngr import save, load
import dataMngr as dm
import compInfo as ci
import dataInfo as di
import machineLearning as ML
from modelsInfo import models

def createAndSaveModel(modelDict, indata, outdata, fp):
    model = ML.generateModel(modelDict['model'], modelDict['modelArrParameters'],
                          modelDict['modelDictParameters'], indata, outdata)
    save(fp + modelDict['savedModelFileName'], model)
    return model


def generateModels(modelDicts, indata, outdata, fp):
  for modelDict in modelNames:
    createAndSaveModel(modelDict, indata, outdata, fp)

#inData,outData = dm.preprocess(ci.originalDataDirectory + "fullData.csv")
#createAndSaveModel(models["RandomForest10"], inData, outData, ci.outputTestingDirectory)

#fullData = dm.changeClassNames(ci.originalDataDirectory + "fullData.csv", ci.originalDataDirectory + "fullDataNorm.csv", di.pillClasses)
#outputData = dm.changeClassNames(ci.originalDataDirectory + "outputData.csv", ci.originalDataDirectory + "outputDataNorm.csv", di.pillClasses)

# inData = dm.load(ci.originalDataDirectory + "inputData.np")
# outData = dm.load(ci.originalDataDirectory + "outputData.np")

# mod = createAndSaveModel(models["RandomForest10"], inData, outData)