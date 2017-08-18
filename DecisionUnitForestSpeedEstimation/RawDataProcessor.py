'''
Created on Apr 11, 2017

@author: patrick
'''
import boto3
import pandas as pd
import random
import numpy as np
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
    
    def dropLowFrequencyFeatures(self, data):
        data=data.drop(data.columns[[1,2,6,11,25,26,27]],axis=1)
        print("Drop columns performed!!")
        print(len(list(data.columns)))
        return data
        
    
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
#         print("hahahahha")
        return trainData, testData

    def splitDataToThree(self,data):
        totalNumber=len(data)
        indexPool=list(data.index)
        trainAndBoostIndexPool=random.sample(indexPool,int(totalNumber*0.8))
        trainIndexPool=trainAndBoostIndexPool[:int(len(trainAndBoostIndexPool)*0.75)]
        boostIndexPool=trainAndBoostIndexPool[(int(len(trainAndBoostIndexPool)*0.75)):int(len(trainAndBoostIndexPool))]
        trainIndexPool.sort()
        boostIndexPool.sort()
        trainAndBoostIndexPool.sort()
        testIndexPool=[]
        for ind in indexPool:
            if ind not in trainAndBoostIndexPool:
                testIndexPool.append(ind)
        trainData=data.loc[trainIndexPool,:]
        boostData=data.loc[boostIndexPool,:]
        testData=data.loc[testIndexPool,:]
#         print("hahahahha")
        return trainData, boostData, testData
        
    def enlargeDataSet(self,data):
        types=set(data["types"])
        res=[]
        for type in types:
            recordForType=list(data[data["types"]==type].index)
            for i in range(0,100):
                indexPool=random.sample(recordForType,5)
                ave=np.zeros(len(data.columns))
                for ind in indexPool:
                    for j in range(0,len(data.columns)-1):
                        ave[j]=ave[j]+(float(data.loc[ind][j]))/5.0
                ave[-1]=type
                res.append(ave)
        df=pd.DataFrame(res,columns=list(data.columns))
#         print("DF:!!!!!!!!!")
        df
        data = data.append(df, ignore_index=True)
        return data
    
        
        
        
        
        
        
        
        
        
        

