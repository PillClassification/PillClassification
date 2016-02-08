from dataMngr import save, load
import dataMngr as dm
import compInfo as ci
import dataInfo as di
import machineLearning as ML
from modelsInfo import models

def createAndSaveModel(modelDict, indata, outdata, outDict = ''):
    model = ML.generateModel(modelDict['model'], modelDict['modelArrParameters'],
                          modelDict['modelDictParameters'], indata, outdata)
    save(outDict + modelDict['savedModelFileName'], model)
    return model

inData = dm.load(ci.originalDataDirectory + "inputData.np")
outData = dm.load(ci.originalDataDirectory + "outputData.np")

createAndSaveModel(models[""])
#sample data
#sampleInData = dm.load(ci.input500DataPoints)
#sampleOutData = dm.load(ci.output500DataPoints)

# true data
#indata = load(ci.dataDirectory + ci.inputDataPath)
#outdata = load(ci.dataDirectory + ci.outputDataPath)


# testingdata = load(ci.dataDirectory + ci.testingInputPathFixed)

# mArr = [load(ci.modelsDirectory + 'adaboostrfd_10_t_10_f_50.mod'), load(ci.modelsDirectory + 'adaboostrfd_10_t_200_f_10.mod')]


# for i in xrange(len(mArr)):
#     m = models[mArr[i]]
#     model = modelArr[i]
#     ML.generateTestOutput(model, testingdata, ci.outputTestingDirectory + m['savedModelFileName'] + '.csv')
