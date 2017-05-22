'''
Created on Apr 25, 2017

@author: patrick
'''
from Classifier import Classifier
from RawDataProcessor import DataProcessor
from ModingStatusClassification.ClassificationManager import ClassificationManager
from time import sleep

dataProcessor=DataProcessor()
a=dataProcessor.getWalkingData()
b,c=dataProcessor.splitData(a)

cs=Classifier()
sleep(1)
cm=ClassificationManager()

walkingTypes=[10,15,20,25,30,35]
cs.trainData(b, walkingTypes, 8, numberOfUnits=200)
cm.trainRandomForest(b)

distNew1, distNew2=cs.testClassificationForDifferentPredict(c)
distCla1, distCla2=cm.testClassification(c)

