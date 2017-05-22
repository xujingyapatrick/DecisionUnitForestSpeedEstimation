'''
Created on Mar 13, 2017

@author: patrick
'''
from __future__ import print_function # Python 2/3 compatibility
import boto3
import json
import decimal
import numpy as np
# from boto3.dynamodb.conditions import Key, Attr
from botocore.exceptions import ClientError
import pandas as pd
import pickle
from ModingStatusClassification.RandomForestGene import RandomForestGene

# from database_setup import Base, Music
# Helper class to convert a DynamoDB item to JSON.
class DecimalEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, decimal.Decimal):
            if o % 1 > 0:
                return float(o)
            else:
                return int(o)
        return super(DecimalEncoder, self).default(o)

############################################################################

class ClassificationManager():
    def __init__(self):
#         self.dynamodb=boto3.resource('dynamodb', region_name='us-west-2', endpoint_url="http://localhost:8000")
        self.dynamodb=boto3.resource('dynamodb', region_name='us-west-2')
        self.table=self.dynamodb.Table("AcceleratorData200Hz")
        self.classifier = pickle.load(open('ModingStatusClassification/speedEstimation.pkl', 'rb'))
#         self.classifier =''
#         self.walkSpeedClassifier =''
#         self.runSpeedClassifier =''
    
    
    def getAllDataFromDynamoDB(self):
        response = self.table.scan()
        arr=[]
        for item in response['Items']:
            line=item['info']['features']
            line=line.split(',')
            line.remove('')
            line.append(int(item['info']['speed']))
            if item['info']['type']=='sitting':
                line.append(0)
            elif item['info']['type']=='walking':
                line.append(1)
            else:
                line.append(2)
            arr.append(line)
        fr=pd.DataFrame(data=arr)
        
        col=[]
        for i in range(28):
            col.append(str(i))
        col.append('speed')
        col.append('types')
        fr.columns=col
        print(fr)
        return fr

    
    def trainRandomForest(self,pddata):
        train=RandomForestGene()
        forest=train.datatrainForType(pddata,numoftrees=200)
        with open('ModingStatusClassification/speedEstimation.pkl', 'wb') as f:
            pickle.dump(forest, f)
        print("status classification training finished")
        self.classifier = pickle.load(open('ModingStatusClassification/speedEstimation.pkl', 'rb'))
        
        print("****************Forest Generation finished!*****************")

    def decideMovingStatus(self,data):
        preds = self.classifier.predict(data)
        if preds==0:
            return {"status":"sitting"}
        elif preds==1:
            return {"status":"walking"}
        elif preds==2:
            return {"status":"running"}
        else:
            return {"status":"error"}

    def decideWalkingSpeedWithDistribute(self,data):
        walkspeeds=[10,15,20,25,30,35]
        prob=self.classifier.predict_proba(data)
        speed=0
        for i in range(len(prob[0])):
            speed=speed+prob[0][i]*walkspeeds[i]
        return speed

    def decideWalkingSpeedWithTowHighestProb(self,data):
        walkspeeds=[10,15,20,25,30,35]
        prob=self.classifier.predict_proba(data)
        loc1=0
        loc2=0
        for i in range(len(prob[0])):
            if prob[0][i]>prob[0][loc1]:
                loc1=i
        
        max=0

        for i in range(len(prob[0])):
            if i!=loc1 and prob[0][i]>=max:
                max=prob[0][i]
                loc2=i
        speed=walkspeeds[loc1]*(prob[0][loc1]/(prob[0][loc1]+prob[0][loc2]))+walkspeeds[loc2]*(prob[0][loc2]/(prob[0][loc1]+prob[0][loc2]))
        return speed


    def decideWalkingSpeedStright(self,data):
#         walkspeeds=[10,15,20,25,30,35]
#         prob=self.classifier.predict_proba(data)
#         speed=0
#         for i in range(len(prob[0])):
#             speed=speed+prob[0][i]*walkspeeds[i]
        speed = self.classifier.predict(data)
        
        return speed[0]
     
    
    
    
    
    def decideRunningSpeed(self,data):
        runspeeds=[25,30,35,40,45,50]
        prob=self.runSpeedClassifier.predict_proba(data)
        speed=0
        for i in range(len(prob[0])):
            speed=speed+prob[0][i]*runspeeds[i]
        return {"speed":speed}
    
    def testClassification(self, featuresFrame):
        featuresTable=list(featuresFrame.values)
        dist1=0
        dist2=0
        for features in featuresTable:
            pred=self.decideWalkingSpeedWithTowHighestProb([features[0:-1]])
            real=int(features[-1])
            dist1=dist1+(real/10.0-pred/10.0)
            dist2=dist2+np.abs(real/10.0-pred/10.0)
        dist1=dist1/float(len(featuresTable))
        dist2=dist2/float(len(featuresTable))
        
        return dist1, dist2
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
             