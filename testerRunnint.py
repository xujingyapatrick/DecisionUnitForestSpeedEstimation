'''
Created on Apr 25, 2017

@author: patrick
'''
from DecisionUnitForestSpeedEstimation.Classifier import Classifier
from DecisionUnitForestSpeedEstimation.RawDataProcessor import DataProcessor
from ModingStatusClassification.ClassificationManager import ClassificationManager
from time import sleep
import numpy as np

dataProcessor=DataProcessor()
# a=dataProcessor.getWalkingData()
a=dataProcessor.getRunningData()

# a=dataProcessor.dropLowFrequencyFeatures(a)
# b,c,d=dataProcessor.splitDataToThree(a)
# b=dataProcessor.enlargeDataSet(b)
b,d = dataProcessor.splitData(a)
b=dataProcessor.enlargeDataSet(b)

print("number of training records: "+str(len(b)))
# print("number of optimizing records: "+str(len(c)))
print("number of testing records: "+str(len(d)))
cs=Classifier()
sleep(1)
cm=ClassificationManager()

walkingTypes=[25,30,35,40,45,50]
cs.trainData(b, walkingTypes, 8, numberOfUnits=800)
cm.trainRandomForest(b)
# distNew1, distNew2=cs.testClassificationByDistributeAvarageAndPlot(d)
distNew1, distNew2=cs.testClassificationByDistributeAvarageAndPlotForRunning(d)
print("Before Decision Unit Opt: distNew2= "+str(np.sqrt(distNew2)))
# cs.getTestResultRMSValuesForEachSpeed(d)
# cs.plotAverageSpeedsForAllDecisionUnits(d)
cs.plotAccuracyInEachSpeedsForAllDecisionUnits(d)
# # cs.trainData(b, walkingTypes, 8, numberOfUnits=800)
# distNew1, distNew2=cs.testClassificationForDifferentPredict(c)
# allMsrs=cs.getAllMSRsForDecisionUnits(c)
# cs.deleteUnusefulDecisionUnitsByMsr(allMsrs, distNew2)
# distNew1, distNew2=cs.testClassificationForDifferentPredict(d)
# distCla1, distCla2=cm.testClassification(d)
# cm.getTestResultRMSValuesForEachSpeed(d)
# print("After Decision Unit Opt: distNew2= "+str(distNew2))
# print("distCla2= "+str(distCla2))
