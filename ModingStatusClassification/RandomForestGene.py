'''
Created on 
@author: Patric
'''

from sklearn.ensemble import RandomForestClassifier

import numpy as np
# import pickle
# import json
class RandomForestGene(object):
    
    
    def dataspilit(self, pddata, raito):
        pddata['is_train'] = np.random.uniform(0, 1, len(pddata))<= raito
        train, test = pddata[pddata['is_train']==True], pddata[pddata['is_train']==False]
        return train,test
    

    def getfeatures(self, traindata):
        features=[]
        for name in traindata.columns:
            if name != "is_train" and name !="types" and name !="speed":
                features.append(name)
        return(traindata[features])
    
    
    def gettypes(self, traindata):
        return(traindata["types"])
    def getSpeed(self,traindata):
        return(traindata["speed"])
    
    def datatrainForType(self, traindata, numoftrees):
        trainfeatures=self.getfeatures(traindata)
        traintypes=self.gettypes(traindata)
        clf = RandomForestClassifier(n_estimators=numoftrees ,n_jobs=4)
        clf.fit(trainfeatures, traintypes)
        #a=clf.get_params(deep=True)
        #print(a)
        return clf
    
    def datatrainForSpeed(self, traindata, numoftrees):
        trainfeatures=self.getfeatures(traindata)
        traintypes=self.getSpeed(traindata)
        clf = RandomForestClassifier(n_estimators=numoftrees ,n_jobs=4)
        clf.fit(trainfeatures, traintypes)
        #a=clf.get_params(deep=True)
        #print(a)
        return clf
    
    
    
    



