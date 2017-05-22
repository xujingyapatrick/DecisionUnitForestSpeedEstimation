'''
Created on Apr 11, 2017

@author: patrick
'''
import boto3
import pandas as pd
import random
class DataProcessor():
    def __init__(self):
        self.dynamodb=boto3.resource('dynamodb', region_name='us-west-2')
        self.table=self.dynamodb.Table("AcceleratorData200Hz")
        
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
        return fr
    
    def getWalkingData(self):
        data=self.getAllDataFromDynamoDB()
        dt=data[data['types']==1]
        dt.pop('types')
        ls=list(dt.columns)
        ls.remove('speed')
        ls.append('types')
        dt.columns=ls
        return dt

    def getRunningData(self):
        data=self.getAllDataFromDynamoDB()
        dt=data[data['types']==2]
        dt.pop('types')
        ls=list(dt.columns)
        ls.remove('speed')
        ls.append('types')
        dt.columns=ls
        return dt
    
    def splitData(self,data):
        totalNumber=len(data)
        indexPool=list(data.index)
        trainIndexPoll=random.sample(indexPool,int(totalNumber*0.8))
        trainIndexPoll.sort()
        testIndexPoll=[]
        for ind in indexPool:
            if ind not in trainIndexPoll:
                testIndexPoll.append(ind)
        
        trainData=data.loc[trainIndexPoll,:]
        testData=data.loc[testIndexPoll,:]
        print("hahahahha")
        return trainData, testData
        
        
        
        
        
        
        
        
        
        
        
        

