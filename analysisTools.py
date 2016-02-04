# -*- coding: utf-8 -*-
import numpy as np
import dataMngr as dm
import compInfo as ci
import os

def similarity(testOut1, testOut2):
    np.sum((np.bitwise_xor(testOut1, testOut2)))
    pass

def vote(testingDataArr):
    numTest = len(testingDataArr)
    numpyArr = []
    for npFile in testingDataArr:
        numpyArr.append(dm.load(npFile)[1:])
        
    vote = numpyArr[0]
    
    for index in xrange(1, numTest):
        vote += numpyArr[index]
    
    vote /= numTest
    vote = (vote > 0.5).astype(int)
    return vote


directory = '/Users/JuanDa/Google Drive/Innocentive Marketing ML Challenge/OutputTestingData/currentVotes/'
dataArr = []

for f in os.listdir(directory):
    if f.endswith(".np"):
        dataArr.append(directory + f)
        
v = vote(dataArr)
dm.writeToFilePath(v, ci.testingDirectory + 'vote3.csv')