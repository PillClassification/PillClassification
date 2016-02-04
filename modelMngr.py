from dataMngr import save, load
import compInfo as ci
import machineLearning as ML
from modelsInfo import models

def createAndSaveModel(modelDict, indata, outdata, outDict = ''):
    model = ML.generateModel(modelDict['model'], modelDict['modelArrParameters'],
                          modelDict['modelDictParameters'], indata, outdata)
    save(outDict + modelDict['savedModelFileName'], model)
    return model

#sample data
#sampleInData = dm.load(ci.input500DataPoints)
#sampleOutData = dm.load(ci.output500DataPoints)

# true data
#indata = load(ci.dataDirectory + ci.inputDataPath)
#outdata = load(ci.dataDirectory + ci.outputDataPath)




"""
# createAndSaveModel(models['RandomForest200'], indata, outdata, 
#                   ci.outputDataDirectory)
                   
# createAndSaveModel(models['NaiveBayesGaussian'], indata, outdata, 
#                   ci.outputDataDirectory)

multilayerPerceptrons = ['MultilayerPerceptronStandard', 'MultilayerPerceptron10050',
       'MultilayerPerceptron20050', 'MultilayerPerceptron200100']
       
for mlp in multilayerPerceptrons:
    createAndSaveModel(models[mlp], indata, outdata, ci.outputDataDirectory)
    



GDmodel = createAndSaveModel(models['GradientDescent'], indata, outdata, 
                   ci.outputDataDirectory)
print "done with GDmodel"
KMeans = createAndSaveModel(models['KMeans'], indata, outdata, ci.outputDataDirectory)
"""

#mArr = ['MultilayerPerceptronStandard', 'MultilayerPerceptron10050', 
#         'MultilayerPerceptron20050', 'MultilayerPerceptron200100']

# mArr = ['AdaboostRFd_10_t_10_f_50', 'AdaboostRFd_10_t_200_f_10']
# modelArr = []

# for modelDict in mArr:
#     m = models[modelDict]
#     model = createAndSaveModel( m, indata, outdata, 
#                    ci.modelsDirectory)
#     modelArr.append(model)

testingdata = load(ci.dataDirectory + ci.testingInputPathFixed)

mArr = [load(ci.modelsDirectory + 'adaboostrfd_10_t_10_f_50.mod'), load(ci.modelsDirectory + 'adaboostrfd_10_t_200_f_10.mod')]


for i in xrange(len(mArr)):
    m = models[mArr[i]]
    model = modelArr[i]
    ML.generateTestOutput(model, testingdata, ci.outputTestingDirectory + m['savedModelFileName'] + '.csv')
