'''
Created on Mar 13, 2017

@author: patrick
'''
from ModingStatusClassification.ClassificationManager import ClassificationManager 
# 
# mng=ClassificationManager()
# pd=mng.getAllDataFromDynamoDB()
# print(pd)
mng=ClassificationManager()

res=mng.decideStatusAndSpeed([[121.776,-1232.592,99.532,1528.0,596.0,1524.0,-1744.0,-6204.0,-600.0,760.199936,1409.291392,330.266144,3597.75456103,3272.0,6800.0,2124.0,1254.836,-1453.9,1210.348,-979.3576,318.825312,-1839.225184,-5652.0,1404.0,-7948.0,13.0,24.0,27.0]])
print(res)