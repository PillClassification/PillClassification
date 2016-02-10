# -*- coding: utf-8 -*-

columnInfo = {'extraneous':[],
            'target':['target_pill'],
            'continous': [],
            'discrete': []}

pillClasses = {"Atripla":0,
              "Cymbalta":1,
              "Epzicom":2,
              "Lexapro":3,
              "Prezista":4,
              "Tivicay":5,
              "Truvada":6}


def setContinuousInfo(numCols):
  for x in xrange(numCols):
    columnInfo['continous'].append("C" + str(x))

setContinuousInfo(516)


