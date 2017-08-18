'''
Created on Apr 25, 2017

@author: patrick
'''
from DecisionUnitForestSpeedEstimation.Classifier import Classifier
from DecisionUnitForestSpeedEstimation.RawDataProcessor import DataProcessor
from ModingStatusClassification.ClassificationManager import ClassificationManager
from time import sleep

dataProcessor=DataProcessor()
a=dataProcessor.getWalkingData()
# a=dataProcessor.dropLowFrequencyFeatures(a)
b,c,d=dataProcessor.splitDataToThree(a)
b=dataProcessor.enlargeDataSet(b)

print("number of training records: "+str(len(b)))
print("number of optimizing records: "+str(len(c)))
print("number of testing records: "+str(len(d)))
cs=Classifier()
sleep(1)
cm=ClassificationManager()

walkingTypes=[10,15,20,25,30,35]
cs.trainData(b, walkingTypes, 8, numberOfUnits=800)
cm.trainRandomForest(b)
distNew1, distNew2=cs.testClassificationForDifferentPredict(d)
print("Before Decision Unit Opt: distNew2= "+str(distNew2))

# cs.trainData(b, walkingTypes, 8, numberOfUnits=800)
correctRate=cs.optimizeDecisionUnits(c)
cs.deleteUnusefulDecisionUnits(correctRate)
distNew1, distNew2=cs.testClassificationForDifferentPredict(d)
distCla1, distCla2=cm.testClassification(d)

print("After Decision Unit Opt: distNew2= "+str(distNew2))
print("distCla2= "+str(distCla2))
