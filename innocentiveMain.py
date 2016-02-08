# -*- coding: utf-8 -*-
import dataMngr as dm
import compInfo as ci
import machineLearning as ML

def makeTestOutput(modelfp, inTestFileName, outTestFileName):
  ML.generateTestOutput(dm.load(modelfp), dm.load(inTestFileName), outTestFileName)



# dm.saveCSVasNP(ci.originalDataDirectory + "processed.csv")