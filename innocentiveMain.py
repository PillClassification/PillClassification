# -*- coding: utf-8 -*-
import dataMngr as dm
import compInfo as ci
import machineLearning as ML

def makeTestOutput(modelfp, inTestFileName, outTestFileName):
  ML.generateTestOutput(dm.load(modelfp), dm.load(inTestFileName), outTestFileName)

ML.generateTestOutput(dm.load(ci.modelsDirectory + 'rForest_n_200Ent.mod'), dm.load(ci.dataDirectory + ci.testingInputPathFixed), ci.outputDataDirectory + "OutputDataEntropy200.csv")
