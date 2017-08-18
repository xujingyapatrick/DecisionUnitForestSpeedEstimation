'''
Created on Apr 11, 2017

@author: patrick
'''
import random
import pandas as pd
import numpy as np
import pickle
import pylab as pl

class Unit():
    def __init__(self):
        self.columns=[]
        self.cube=[]
        self.records=[]

class Classifier():
    def __init__(self):
        self.decisionUnits=None
        self.types=None
        self.featureCount=np.zeros(28)
    
    ##in the dataFrame, features should be float/int, a column called "types" should be included while the values of the types should be String 
    ##typeList should be a list of int types
    def trainData(self,dataFrame,typeList,numberOfFeatures, numberOfUnits=200):
        self.decisionUnits=[]
        self.types=typeList
        dicTypes={}
        for k in typeList:
            dicTypes[str(k)]=list(dataFrame[dataFrame["types"]==k].index)
        while numberOfUnits!=0:
            unit=Unit()
            columns=[a for a in (range(0,len(dataFrame.columns)-1))]
            unit.columns=random.sample(columns,numberOfFeatures)
            
            for i in range(0,len(typeList)):
                rr=random.choice(dicTypes[str(typeList[i])])
                record=[]
                for j in unit.columns:
                    record.append(float(dataFrame.loc[rr][j]))
                unit.records.append(record)
            
            for rec1 in unit.records:
                d2=[]
                for rec2 in unit.records:
                    d3=[]
                    for i in range(0,len(unit.columns)):
                        d3.append(np.abs(rec1[i]-rec2[i]))
                    d2.append(d3)
                unit.cube.append(d2)
                
            self.decisionUnits.append(unit)
            numberOfUnits=numberOfUnits-1
        
        print("Train data finished!")
##in the dataFrame, features should be float/int, a column called "types" should be included while the values of the types should be String 
##typeList should be a list of int types
    def trainDataNormalized(self,dataFrame,typeList,numberOfFeatures, numberOfUnits=200):
        self.decisionUnits=[]
        self.types=typeList
        dicTypes={}
        for k in typeList:
            dicTypes[str(k)]=list(dataFrame[dataFrame["types"]==k].index)
        while numberOfUnits!=0:
            unit=Unit()
            columns=[a for a in (range(0,len(dataFrame.columns)-1))]
            unit.columns=random.sample(columns,numberOfFeatures)
            
            for i in range(0,len(typeList)):
                rr=random.choice(dicTypes[str(typeList[i])])
                record=[]
                for j in unit.columns:
                    record.append(float(dataFrame.loc[rr][j]))
                unit.records.append(record)
            averages=[]
            for i in range(0,len(unit.records[0])):
                ave= sum(row[i] for row in unit.records)
                ave=ave/len(unit.records)
                for j in range(0,len(unit.records)):
                    unit.records[j][i]=unit.records[j][i]/ave
                averages.append(ave)
                
            for rec1 in unit.records:
                d2=[]
                for rec2 in unit.records:
                    d3=[]
                    for i in range(0,len(unit.columns)):
                        d3.append(np.abs(rec1[i]-rec2[i]))
                    d2.append(d3)
                unit.cube.append(d2)
            self.decisionUnits.append(unit)
            
            for i in range(0,len(unit.records[0])):
                for j in range(0,len(unit.records)):
                    unit.records[j][i]=unit.records[j][i]*averages[i]
            
            numberOfUnits=numberOfUnits-1
        
        print("Train data finished!")
            
    #features should be a list, while returning a Sting    
    def predict(self,features):
        predictTypesDic={}
        for tp in self.types:
            predictTypesDic[str(tp)]=0
        
        for unit in self.decisionUnits:
            typePool=[a for a in range(0,len(self.types))]
            while len(typePool)>1:
                ##find max 
                maxDist=0
                maxi=0
                maxj=0
                maxk=0;
                for i in typePool:
                    for j in typePool:
                        for k in range(0,len(unit.columns)):
                            if unit.cube[i][j][k]>=maxDist:
                                maxDist=unit.cube[i][j][k]
                                maxi=i
                                maxj=j
                                maxk=k
                disti=np.abs(unit.records[maxi][maxk]-float(features[unit.columns[maxk]]))
                distj=np.abs(unit.records[maxj][maxk]-float(features[unit.columns[maxk]]))
                if disti>distj:
                    typePool.remove(maxi)
                else:
                    typePool.remove(maxj)
            typeForUnit=self.types[typePool[0]]
            predictTypesDic[str(typeForUnit)]=predictTypesDic[str(typeForUnit)]+1
        resType=""
        countmax=0
        for tp in predictTypesDic:
            if predictTypesDic[tp]>countmax:
                countmax=predictTypesDic[tp]
                resType=tp
        return int(resType)



    #features should be a list, while returning a Sting, predict the final speed by the speed distribution of decision units    
    def predictWithDistribute(self,features):
        predictTypesDic={}
        for tp in self.types:
            predictTypesDic[str(tp)]=0
        
        for unit in self.decisionUnits:
            typePool=[a for a in range(0,len(self.types))]
            while len(typePool)>1:
                ##find max 
                maxDist=0
                maxi=0
                maxj=0
                maxk=0;
                for i in typePool:
                    for j in typePool:
                        for k in range(0,len(unit.columns)):
                            if unit.cube[i][j][k]>=maxDist:
                                maxDist=unit.cube[i][j][k]
                                maxi=i
                                maxj=j
                                maxk=k
                disti=np.abs(unit.records[maxi][maxk]-float(features[unit.columns[maxk]]))
                distj=np.abs(unit.records[maxj][maxk]-float(features[unit.columns[maxk]]))
                if disti>distj:
                    typePool.remove(maxi)
                else:
                    typePool.remove(maxj)
            typeForUnit=self.types[typePool[0]]
            predictTypesDic[str(typeForUnit)]=predictTypesDic[str(typeForUnit)]+1
        finalSpeed=0
        for tp in predictTypesDic:
            finalSpeed=finalSpeed+int(tp)*1.0*predictTypesDic[tp]/len(self.decisionUnits)
        return finalSpeed

    def predictStright(self,features):
        predictTypesDic={}
        for tp in self.types:
            predictTypesDic[str(tp)]=0
        for unit in self.decisionUnits:
            typePool=[a for a in range(0,len(self.types))]
            while len(typePool)>1:
                ##find max 
                maxDist=0
                maxi=0
                maxj=0
                maxk=0;
                for i in typePool:
                    for j in typePool:
                        for k in range(0,len(unit.columns)):
                            if unit.cube[i][j][k]>=maxDist:
                                maxDist=unit.cube[i][j][k]
                                maxi=i
                                maxj=j
                                maxk=k
                disti=np.abs(unit.records[maxi][maxk]-float(features[unit.columns[maxk]]))
                distj=np.abs(unit.records[maxj][maxk]-float(features[unit.columns[maxk]]))
                if disti>distj:
                    typePool.remove(maxi)
                else:
                    typePool.remove(maxj)
            typeForUnit=self.types[typePool[0]]
            predictTypesDic[str(typeForUnit)]=predictTypesDic[str(typeForUnit)]+1
        finalSpeed=0
        max=0
        for tp in predictTypesDic:
            if predictTypesDic[tp]>max:
                max=predictTypesDic[tp]
                finalSpeed=int(tp)
                
        return finalSpeed

    def predictStrightWith500Limit(self,features):
        predictTypesDic={}
        for tp in self.types:
            predictTypesDic[str(tp)]=0
        count = 0
        for unit in self.decisionUnits:
            count = count+1
            if(count > 500):
                break
            typePool=[a for a in range(0,len(self.types))]
            while len(typePool)>1:
                ##find max 
                maxDist=0
                maxi=0
                maxj=0
                maxk=0;
                for i in typePool:
                    for j in typePool:
                        for k in range(0,len(unit.columns)):
                            if unit.cube[i][j][k]>=maxDist:
                                maxDist=unit.cube[i][j][k]
                                maxi=i
                                maxj=j
                                maxk=k
                disti=np.abs(unit.records[maxi][maxk]-float(features[unit.columns[maxk]]))
                distj=np.abs(unit.records[maxj][maxk]-float(features[unit.columns[maxk]]))
                if disti>distj:
                    typePool.remove(maxi)
                else:
                    typePool.remove(maxj)
            typeForUnit=self.types[typePool[0]]
            predictTypesDic[str(typeForUnit)]=predictTypesDic[str(typeForUnit)]+1
        finalSpeed=0
        max=0
        for tp in predictTypesDic:
            if predictTypesDic[tp]>max:
                max=predictTypesDic[tp]
                finalSpeed=int(tp)
                
        return finalSpeed

    def getCorrectsForOneRecord(self,features):
        isCorrect=np.zeros(len(self.decisionUnits))
        cur=0
        for unit in self.decisionUnits:
            typePool=[a for a in range(0,len(self.types))]
            while len(typePool)>1:
                ##find max 
                maxDist=0
                maxi=0
                maxj=0
                maxk=0;
                for i in typePool:
                    for j in typePool:
                        for k in range(0,len(unit.columns)):
                            if unit.cube[i][j][k]>=maxDist:
                                maxDist=unit.cube[i][j][k]
                                maxi=i
                                maxj=j
                                maxk=k
                disti=np.abs(unit.records[maxi][maxk]-float(features[unit.columns[maxk]]))
                distj=np.abs(unit.records[maxj][maxk]-float(features[unit.columns[maxk]]))
                if disti>distj:
                    typePool.remove(maxi)
                else:
                    typePool.remove(maxj)
            typeForUnit=self.types[typePool[0]]
            if str(typeForUnit)== str(features[-1]):
                isCorrect[cur]=isCorrect[cur]+1
            cur=cur+1
            
        return isCorrect

    def optimizeDecisionUnits(self, featuresFrame):
        featuresTable=list(featuresFrame.values)
        correctRate=np.zeros(len(self.decisionUnits))
        for features in featuresTable:
            isCorrect=self.getCorrectsForOneRecord(features)
            for i in range(0,len(isCorrect)):
                correctRate[i]=correctRate[i]+isCorrect[i]
        for i in range(0,len(correctRate)):
            correctRate[i]=correctRate[i]/len(featuresTable)
#             print(str(i)+":"+str(correctRate[i])+" ")
        x=range(0,len(correctRate))
        pl.plot(x,correctRate,'*')
#         pl.show()
        return correctRate
    
    #we keep the total number of decision units to be 500, if not error will alert
    def deleteUnusefulDecisionUnits(self, correctRate):
        print("len(self.decisionUnits): "+ str(len(self.decisionUnits)))
        for i in range(len(correctRate)-1,-1,-1):
            if(correctRate[i]<0.3):
                self.decisionUnits.remove(self.decisionUnits[i])
        print("len(self.decisionUnits): "+ str(len(self.decisionUnits)))



    def predictStrightWithFeatureCount(self,features):
        featureCount=np.zeros(28)
        predictTypesDic={}
        for tp in self.types:
            predictTypesDic[str(tp)]=0
        
        for unit in self.decisionUnits:
            typePool=[a for a in range(0,len(self.types))]
            while len(typePool)>1:
                ##find max 
                maxDist=0
                maxi=0
                maxj=0
                maxk=0;
                for i in typePool:
                    for j in typePool:
                        for k in range(0,len(unit.columns)):
                            if unit.cube[i][j][k]>=maxDist:
                                maxDist=unit.cube[i][j][k]
                                maxi=i
                                maxj=j
                                maxk=k
                disti=np.abs(unit.records[maxi][maxk]-float(features[unit.columns[maxk]]))
                distj=np.abs(unit.records[maxj][maxk]-float(features[unit.columns[maxk]]))
                if disti>distj:
                    typePool.remove(maxi)
                else:
                    typePool.remove(maxj)
                featureCount[unit.columns[maxk]]=featureCount[unit.columns[maxk]]+1
                
            typeForUnit=self.types[typePool[0]]
            predictTypesDic[str(typeForUnit)]=predictTypesDic[str(typeForUnit)]+1
        finalSpeed=0
        max=0
        for tp in predictTypesDic:
            if predictTypesDic[tp]>max:
                max=predictTypesDic[tp]
                finalSpeed=int(tp)
        
        print(featureCount)
        for i in range(0,len(featureCount)):
            self.featureCount[i]=self.featureCount[i]+featureCount[i]
        return finalSpeed


    def predictWithTowHighestProb(self,features):
        predictTypesDic={}
        for tp in self.types:
            predictTypesDic[str(tp)]=0
        
        for unit in self.decisionUnits:
            typePool=[a for a in range(0,len(self.types))]
            while len(typePool)>1:
                ##find max 
                maxDist=0
                maxi=0
                maxj=0
                maxk=0;
                for i in typePool:
                    for j in typePool:
                        for k in range(0,len(unit.columns)):
                            if unit.cube[i][j][k]>=maxDist:
                                maxDist=unit.cube[i][j][k]
                                maxi=i
                                maxj=j
                                maxk=k
                disti=np.abs(unit.records[maxi][maxk]-float(features[unit.columns[maxk]]))
                distj=np.abs(unit.records[maxj][maxk]-float(features[unit.columns[maxk]]))
                if disti>distj:
                    typePool.remove(maxi)
                else:
                    typePool.remove(maxj)
            typeForUnit=self.types[typePool[0]]
            predictTypesDic[str(typeForUnit)]=predictTypesDic[str(typeForUnit)]+1
        finalSpeed=0
        max1=0
        maxSpeed1=0
        max2=0
        maxSpeed2=0
        
        for tp in predictTypesDic:
            if predictTypesDic[tp]>max1:
                max1=predictTypesDic[tp]
                maxSpeed1=int(tp)
        for tp in predictTypesDic:
            if int(tp)!=maxSpeed1 and predictTypesDic[tp]>max2:
                max2=predictTypesDic[tp]
                maxSpeed2=int(tp)
        
        finalSpeed=maxSpeed1*(max1*1.0/(max2+max1))+maxSpeed2*(max2*1.0/(max2+max1))
        return finalSpeed

            
    #featuresTable should be a dataframe of features
    def predictAverageSpeed(self,featuresFrame):
        featuresTable=list(featuresFrame.values)
        predDic={}
        for tp in self.types:
            predDic[str(tp)]=0
        for features in featuresTable:
            oncePredict=self.predict(features)
            predDic[str(oncePredict)]=predDic[str(oncePredict)]+1
        total=0
        for tp in predDic:
            total=total+int(tp)*predDic[tp]
        speed=total/len(featuresTable)
        return speed
    
    def testClassification(self, featuresFrame):
        featuresTable=list(featuresFrame.values)
        typesCount=len(self.types)
        #table used to sensus the predict-real table: columns are pred, rows are real
        testTable=np.zeros((typesCount,typesCount),dtype=int)
        dist1=0
        dist2=0
        for features in featuresTable:
            pred=self.predict(features)
            real=int(features[-1])
            dist1=dist1+(real/10.0-pred/10.0)
            dist2=dist2+np.power(real/10.0-pred/10.0,2)
            testTable[self.types.index(real)][self.types.index(pred)]=testTable[self.types.index(real)][self.types.index(pred)]+1
        distribution=pd.DataFrame(testTable,index=self.types,columns=self.types)
        print("real-predict table is :")
        print(distribution)
        
        totalAccurateCount=0
        for i in range(0,typesCount):
            totalAccurateCount=totalAccurateCount+testTable[i][i]
        totalAccuracy=totalAccurateCount/float(len(featuresTable))
        print("total prediction accuracy is: "+str(totalAccuracy))
        dist1=dist1/float(len(featuresTable))
        dist2=dist2/float(len(featuresTable))
        
        return distribution, totalAccuracy, dist1, dist2
        
    
    def testClassificationForDifferentPredict(self, featuresFrame):
        featuresTable=list(featuresFrame.values)
        dist1=0
        dist2=0
        for features in featuresTable:
            pred=self.predictStrightWith500Limit(features)
            real=int(features[-1])
            dist1=dist1+(real/10.0-pred/10.0)
            dist2=dist2+np.power(real/10.0-pred/10.0,2)
        dist1=dist1/float(len(featuresTable))
        dist2=dist2/float(len(featuresTable))
#         print("features use frequency:")
#         print(self.featureCount)
        
        return dist1, dist2
        
    def save(self):
        with open('speedEstimater.pkl', 'wb') as f:
            pickle.dump(self, f)
        print("status classification saved")

        
        
        
        
        
        
        
        
        
        
        
        
        
    