'''
Created on 2

@author: Patric
'''
import pandas as pd
import numpy as np
import time as tm
import math
class RawDataProcessor(object):
    
    
    def parserawdata(self, source, status):
       
        datastore = pd.DataFrame({'x':[], 'y':[], 'z':[], 'time':[], 'types':[]})
        file = open(source)
        while 1:
            line = file.readline()
            # print(line)
            if not line:
                break
            n = 0
            count = 0
            while n < len(line):
                if line[n] == ' ':
                    count = count + 1
                    if count == 2:
                        xstart = n + 1
                    
                    if count == 3:
                        x = int(line[xstart:n])
                    if count == 5:
                        ystart = n + 1
                    if count == 6:
                        y = int(line[ystart:n])
                    if count == 8:
                        zstart = n + 1
                    if count == 9:
                        z = int(line[zstart:n])
                    if count == 10:
                        time = int(line[n + 1:len(line)])
                        
                n = n + 1
            # print(x, y, z, time)    
            oneline = pd.DataFrame({'x':[x], 'y':[y], 'z':[z], 'time':[time], 'types':[status]})
            datastore = datastore.append(oneline, ignore_index=True)
        return datastore
    
    def parserawdataNewAcc(self, source, status):   
        datastore = pd.DataFrame({'x':[], 'y':[], 'z':[], 'time':[], 'types':[]})
        f = open(source)
        totalLines=0
        while f.readline():
            totalLines=totalLines+1
        
        statusArray=(status+',')*totalLines
        statusArray=statusArray.split(',')
        statusArray.remove('')

        
        dataArray=np.ones((totalLines,4), dtype=float)    
        file = open(source)
        lineCount=0
        while 1:
            line = file.readline()
            # print(line)
            
            if not line:
                break
            line = line.split(',')
            dataArray[lineCount,0] = float(line[0])
            dataArray[lineCount,1] = float(line[1])
            dataArray[lineCount,2] = float(line[2])
            dataArray[lineCount,3] = 0.0
            lineCount=lineCount+1
        datastore=pd.DataFrame({'x':dataArray[:,0],
                   'y':dataArray[:,1],
                   'z':dataArray[:,2],
                   'time':dataArray[:,3],
                   'types':statusArray})
        return datastore
    
    def parseRawdataFromEvaBoard(self, source, status):   
        datastore = pd.DataFrame({'x':[], 'y':[], 'z':[], 'time':[], 'types':[]})
        throwStartLines=1000
        throwEndLines=1000
        
        f = open(source)
        totalLines=0
        while f.readline():
            totalLines=totalLines+1
        
        statusArray=(status+' ')*(totalLines-throwStartLines-throwEndLines)
        statusArray=statusArray.split(' ')
        #print(statusArray)
        statusArray.remove('')

        
        dataArray=np.ones((totalLines-throwStartLines-throwEndLines,4), dtype=float)    
        file = open(source)
        lineCount=0
        dataArrayCount=0
        while 1:
            line = file.readline()
            lineCount=lineCount+1
            # print(line)
            if lineCount<=throwStartLines:
                continue
            if lineCount>=(totalLines-throwEndLines):
                break
            
            line = line.split(' ')
            dataArray[dataArrayCount,0] = float(line[0])
            dataArray[dataArrayCount,1] = float(line[2])
            dataArray[dataArrayCount,2] = float(line[4])
            dataArray[dataArrayCount,3] = 0.0
            dataArrayCount=dataArrayCount+1
        print("dataArray:"+str(len(dataArray)))
        print("statusArray:"+str(len(statusArray)))
        datastore=pd.DataFrame({'x':dataArray[:,0],
                   'y':dataArray[:,1],
                   'z':dataArray[:,2],
                   'time':dataArray[:,3],
                   'types':statusArray})
        return datastore
    
        
    
    def parserawdataWithABS(self, source, status):
       
        datastore = pd.DataFrame({'x':[], 'y':[], 'z':[], 'time':[], 'types':[]})
        file = open(source)
        while 1:
            line = file.readline()
            # print(line)
            if not line:
                break
            n = 0
            count = 0
            while n < len(line):
                if line[n] == ' ':
                    count = count + 1
                    if count == 2:
                        xstart = n + 1
                    
                    if count == 3:
                        x = int(line[xstart:n])
                    if count == 5:
                        ystart = n + 1
                    if count == 6:
                        y = int(line[ystart:n])
                    if count == 8:
                        zstart = n + 1
                    if count == 9:
                        z = int(line[zstart:n])
                    if count == 10:
                        time = int(line[n + 1:len(line)])
                n = n + 1
            # print(x, y, z, time)    
            oneline = pd.DataFrame({'x':[np.abs(x)], 'y':[np.abs(y)], 'z':[np.abs(z)], 'time':[time], 'types':[status]})
            
            datastore = datastore.append(oneline, ignore_index=True)
        return datastore

    
    def genefeatures(self, pddata, kpoints):
        accthres = 50
        featureandtype = pd.DataFrame({'accavgx':[], 'accavgy':[], 'accavgz':[],
                                     'accmaxx':[], 'accmaxy':[], 'accmaxz':[],
                                     'accminx':[], 'accminy':[], 'accminz':[],
                                     'accabsx':[], 'accabsy':[], 'accabsz':[],
                                     'accavgres':[], 'Dx':[], 'Dy':[], 'Dz':[],
                                     'Da1':[], 'Da2':[], 'Da3':[], 'Db1':[], 'Db2':[], 'Db3':[],
                                     'D1':[], 'D2':[], 'D3':[], 'frex':[], 'frey':[], 'frez':[], 'types':[]})
        
        n = 0
        while n < (len(pddata) - kpoints):
            m = n
            accavgx = np.average(pddata['x'][m:(n + kpoints)])
            accavgy = np.average(pddata['y'][m:(n + kpoints)])
            accavgz = np.average(pddata['z'][m:(n + kpoints)])
            accmaxx = 0
            accmaxy = 0
            accmaxz = 0
            accminx = 0
            accminy = 0
            accminz = 0
            accabsx = 0
            accabsy = 0
            accabsz = 0 
            accres = 0
            frex = 0
            frey = 0
            frez = 0
            while m < (n + kpoints - 1):
                if pddata['x'][m] > accmaxx:
                    accmaxx = pddata['x'][m]
                if pddata['x'][m] < accminx:
                    accminx = pddata['x'][m]
                accabsx = accabsx + np.abs(pddata['x'][m] - accavgx)
                if (pddata['x'][m] - accthres > 0 and pddata['x'][m + 1] - accthres < 0) or (pddata['x'][m] - accthres < 0 and pddata['x'][m + 1] - accthres > 0):
                    frex = frex + 1
                if pddata['y'][m] > accmaxy:
                    accmaxy = pddata['y'][m]
                if pddata['y'][m] < accminy:
                    accminy = pddata['y'][m]
                accabsy = accabsy + np.abs(pddata['y'][m] - accavgy)
                if (pddata['y'][m] - accthres > 0 and pddata['y'][m + 1] - accthres < 0) or (pddata['y'][m] - accthres < 0 and pddata['y'][m + 1] - accthres > 0):
                    frey = frey + 1
                if pddata['z'][m] > accmaxz:
                    accmaxz = pddata['z'][m]
                if pddata['z'][m] < accminz:
                    accminz = pddata['z'][m]
                accabsz = accabsz + np.abs(pddata['z'][m] - accavgz)
                if (pddata['z'][m] - accthres > 0 and pddata['z'][m + 1] - accthres < 0) or (pddata['z'][m] - accthres < 0 and pddata['z'][m + 1] - accthres > 0):
                    frez = frez + 1
                
                accres = accres + np.abs(pddata['x'][m + 1] - pddata['x'][m]) + np.abs(pddata['y'][m + 1] - pddata['y'][m]) + np.abs(pddata['z'][m + 1] - pddata['z'][m])
                
                m = m + 1
            if m == n + kpoints - 1:
                if pddata['x'][m] > accmaxx:
                    accmaxx = pddata['x'][m]
                if pddata['x'][m] < accminx:
                    accminx = pddata['x'][m]
                accabsx = accabsx + np.abs(pddata['x'][m] - accavgx)
                if pddata['y'][m] > accmaxy:
                    accmaxy = pddata['y'][m]
                if pddata['y'][m] < accminy:
                    accminy = pddata['y'][m]
                accabsy = accabsy + np.abs(pddata['y'][m] - accavgy)
                if pddata['z'][m] > accmaxz:
                    accmaxz = pddata['z'][m]
                if pddata['z'][m] < accminz:
                    accminz = pddata['z'][m]
                accabsz = accabsz + np.abs(pddata['z'][m] - accavgz)
            accabsx = accabsx / kpoints
            accabsy = accabsy / kpoints
            accabsz = accabsz / kpoints
            accavgres = accres / (kpoints - 1)
            Dx = accmaxx - accminx
            Dy = accmaxy - accminy
            Dz = accmaxz - accminz
            Da1 = accavgx - accavgy - accavgz
            Da2 = accavgy - accavgz - accavgx
            Da3 = accavgz - accavgx - accavgy
            Db1 = accabsx - accabsy - accabsz
            Db2 = accabsy - accabsx - accabsz
            Db3 = accabsz - accabsx - accabsy
            D1 = Dx - Dy - Dy
            D2 = Dy - Dz - Dx
            D3 = Dz - Dx - Dy
            types = pddata['types'][n]
            
            oneline = pd.DataFrame({'accavgx':[accavgx], 'accavgy':[accavgy], 'accavgz':[accavgz],
                                  'accmaxx':[accmaxx], 'accmaxy':[accmaxy], 'accmaxz':[accmaxz],
                                  'accminx':[accminx], 'accminy':[accminy], 'accminz':[accminz],
                                  'accabsx':[accabsx], 'accabsy':[accabsy], 'accabsz':[accabsz],
                                  'accavgres':[accavgres], 'Dx':[Dx], 'Dy':[Dy], 'Dz':[Dz],
                                  'Da1':[Da1], 'Da2':[Da2], 'Da3':[Da3], 'Db1':[Db1], 'Db2':[Db2], 'Db3':[Db3],
                                  'D1':[D1], 'D2':[D2], 'D3':[D3], 'frex':[frex], 'frey':[frey], 'frez':[frez], 'types':[types]})
            
            
            featureandtype = featureandtype.append(oneline, ignore_index=True)
            
                    
            n = n + kpoints
       # print(featureandtype['accavgx'][0:20])    
        return featureandtype    



    def genefeaturesFast(self, pddata, kpoints):
        accthres = 20
        #=======================================================================
        # featureandtype = pd.DataFrame({'accavgx':[], 'accavgy':[], 'accavgz':[],
        #                              'accmaxx':[], 'accmaxy':[], 'accmaxz':[],
        #                              'accminx':[], 'accminy':[], 'accminz':[],
        #                              'accabsx':[], 'accabsy':[], 'accabsz':[],
        #                              'accavgres':[], 'Dx':[], 'Dy':[], 'Dz':[],
        #                              'Da1':[], 'Da2':[], 'Da3':[], 'Db1':[], 'Db2':[], 'Db3':[],
        #                              'D1':[], 'D2':[], 'D3':[], 'frex':[], 'frey':[], 'frez':[], 'types':[]})
        #=======================================================================
        dataArray=np.ones((int(pddata.shape[0]/kpoints),28), dtype=float)    
        
        statusArray=(pddata['types'][0]+' ')*(int((pddata.shape[0]/kpoints)))
        statusArray=statusArray.split(' ')
        #print(statusArray)
        statusArray.remove('')

        
        n = 0
        dataArrayCount=0
        while n < (len(pddata) - kpoints):
            m = n
            accavgx = np.average(pddata['x'][m:(n + kpoints)])
            accavgy = np.average(pddata['y'][m:(n + kpoints)])
            accavgz = np.average(pddata['z'][m:(n + kpoints)])
            accmaxx = 0
            accmaxy = 0
            accmaxz = 0
            accminx = 0
            accminy = 0
            accminz = 0
            accabsx = 0
            accabsy = 0
            accabsz = 0 
            accres = 0
            frex = 0
            frey = 0
            frez = 0
            while m < (n + kpoints - 1):
                if pddata['x'][m] > accmaxx:
                    accmaxx = pddata['x'][m]
                if pddata['x'][m] < accminx:
                    accminx = pddata['x'][m]
                accabsx = accabsx + np.abs(pddata['x'][m] - accavgx)
                if (pddata['x'][m] - accthres > 0 and pddata['x'][m + 1] - accthres < 0) or (pddata['x'][m] - accthres < 0 and pddata['x'][m + 1] - accthres > 0):
                    frex = frex + 1
                if pddata['y'][m] > accmaxy:
                    accmaxy = pddata['y'][m]
                if pddata['y'][m] < accminy:
                    accminy = pddata['y'][m]
                accabsy = accabsy + np.abs(pddata['y'][m] - accavgy)
                if (pddata['y'][m] - accthres > 0 and pddata['y'][m + 1] - accthres < 0) or (pddata['y'][m] - accthres < 0 and pddata['y'][m + 1] - accthres > 0):
                    frey = frey + 1
                if pddata['z'][m] > accmaxz:
                    accmaxz = pddata['z'][m]
                if pddata['z'][m] < accminz:
                    accminz = pddata['z'][m]
                accabsz = accabsz + np.abs(pddata['z'][m] - accavgz)
                if (pddata['z'][m] - accthres > 0 and pddata['z'][m + 1] - accthres < 0) or (pddata['z'][m] - accthres < 0 and pddata['z'][m + 1] - accthres > 0):
                    frez = frez + 1
                
                accres = accres + math.sqrt(math.pow(pddata['x'][m + 1] + pddata['x'][m],2) + math.pow(pddata['y'][m + 1] + pddata['y'][m],2) + math.pow(pddata['z'][m + 1] + pddata['z'][m],2))
                
                m = m + 1
            if m == n + kpoints - 1:
                if pddata['x'][m] > accmaxx:
                    accmaxx = pddata['x'][m]
                if pddata['x'][m] < accminx:
                    accminx = pddata['x'][m]
                accabsx = accabsx + np.abs(pddata['x'][m] - accavgx)
                if pddata['y'][m] > accmaxy:
                    accmaxy = pddata['y'][m]
                if pddata['y'][m] < accminy:
                    accminy = pddata['y'][m]
                accabsy = accabsy + np.abs(pddata['y'][m] - accavgy)
                if pddata['z'][m] > accmaxz:
                    accmaxz = pddata['z'][m]
                if pddata['z'][m] < accminz:
                    accminz = pddata['z'][m]
                accabsz = accabsz + np.abs(pddata['z'][m] - accavgz)
            accabsx = accabsx / kpoints
            accabsy = accabsy / kpoints
            accabsz = accabsz / kpoints
            accavgres = accres / (kpoints - 1)
            Dx = accmaxx - accminx
            Dy = accmaxy - accminy
            Dz = accmaxz - accminz
            Da1 = accavgx - accavgy - accavgz
            Da2 = accavgy - accavgz - accavgx
            Da3 = accavgz - accavgx - accavgy
            Db1 = accabsx - accabsy - accabsz
            Db2 = accabsy - accabsx - accabsz
            Db3 = accabsz - accabsx - accabsy
            D1 = Dx - Dy - Dz
            D2 = Dy - Dz - Dx
            D3 = Dz - Dx - Dy
            #types = pddata['types'][n]
            
            
            dataArray[dataArrayCount,0] = accavgx
            dataArray[dataArrayCount,1] = accavgy
            dataArray[dataArrayCount,2] = accavgz
            dataArray[dataArrayCount,3] = accmaxx
            dataArray[dataArrayCount,4] = accmaxy
            dataArray[dataArrayCount,5] = accmaxz
            dataArray[dataArrayCount,6] = accminx
            dataArray[dataArrayCount,7] = accminy
            dataArray[dataArrayCount,8] = accminz
            dataArray[dataArrayCount,9] = accabsx
            dataArray[dataArrayCount,10] = accabsy
            dataArray[dataArrayCount,11] = accabsz
            dataArray[dataArrayCount,12] = accavgres
            dataArray[dataArrayCount,13] = Dx
            dataArray[dataArrayCount,14] = Dy
            dataArray[dataArrayCount,15] = Dz
            dataArray[dataArrayCount,16] = Da1
            dataArray[dataArrayCount,17] = Da2
            dataArray[dataArrayCount,18] = Da3
            dataArray[dataArrayCount,19] = Db1
            dataArray[dataArrayCount,20] = Db2
            dataArray[dataArrayCount,21] = Db3
            dataArray[dataArrayCount,22] = D1
            dataArray[dataArrayCount,23] = D2
            dataArray[dataArrayCount,24] = D3
            dataArray[dataArrayCount,25] = frex
            dataArray[dataArrayCount,26] = frey
            dataArray[dataArrayCount,27] = frez
            
            
            #===================================================================
            # 
            # oneline = pd.DataFrame({'accavgx':[accavgx], 'accavgy':[accavgy], 'accavgz':[accavgz],
            #                       'accmaxx':[accmaxx], 'accmaxy':[accmaxy], 'accmaxz':[accmaxz],
            #                       'accminx':[accminx], 'accminy':[accminy], 'accminz':[accminz],
            #                       'accabsx':[accabsx], 'accabsy':[accabsy], 'accabsz':[accabsz],
            #                       'accavgres':[accavgres], 'Dx':[Dx], 'Dy':[Dy], 'Dz':[Dz],
            #                       'Da1':[Da1], 'Da2':[Da2], 'Da3':[Da3], 'Db1':[Db1], 'Db2':[Db2], 'Db3':[Db3],
            #                       'D1':[D1], 'D2':[D2], 'D3':[D3], 'frex':[frex], 'frey':[frey], 'frez':[frez], 'types':[types]})
            # 
            #===================================================================
            
                    
            n = n + kpoints
            dataArrayCount=dataArrayCount+1
       # print(featureandtype['accavgx'][0:20])    
        featureandtype=pd.DataFrame({'accavgx':dataArray[:,0], 'accavgy':dataArray[:,1], 'accavgz':dataArray[:,2],
                                  'accmaxx':dataArray[:,3], 'accmaxy':dataArray[:,4], 'accmaxz':dataArray[:,5],
                                  'accminx':dataArray[:,6], 'accminy':dataArray[:,7], 'accminz':dataArray[:,8],
                                  'accabsx':dataArray[:,9], 'accabsy':dataArray[:,10], 'accabsz':dataArray[:,11],
                                  'accavgres':dataArray[:,12], 'Dx':dataArray[:,13], 'Dy':dataArray[:,14], 'Dz':dataArray[:,15],
                                  'Da1':dataArray[:,16], 'Da2':dataArray[:,17], 'Da3':dataArray[:,18], 'Db1':dataArray[:,19], 'Db2':dataArray[:,20], 'Db3':dataArray[:,21],
                                  'D1':dataArray[:,22], 'D2':dataArray[:,23], 'D3':dataArray[:,24], 'frex':dataArray[:,25], 'frey':dataArray[:,26], 'frez':dataArray[:,27], 'types':statusArray})
        
        return featureandtype    

    def getClassificationFeaturePandasFromDataFile(self,source):
        f = open(source)
        totalLines=0
        while f.readline():
            totalLines=totalLines+1
        dataArray=np.ones((totalLines,28), dtype=float)    
        statusList=("Def"+' ')*totalLines
        statusList=statusList.split(' ')
        statusList.remove('')

        file=open(source)
        linecount=0
        while 1:
            line=file.readline()
            if not line:
                break
            line=line.split(',')
            for i in range(0,28):
                dataArray[linecount,i]=float(line[i])
            statusList[linecount]=line[29][:-1]
# we use line[29][-1] because the last char of line[29] is \n
#             print(line[29][:-1]+"++++"+line[29]+"++++")
            linecount=linecount+1
        featureandtype=pd.DataFrame({'accavgx':dataArray[:,0], 'accavgy':dataArray[:,1], 'accavgz':dataArray[:,2],
                                  'accmaxx':dataArray[:,3], 'accmaxy':dataArray[:,4], 'accmaxz':dataArray[:,5],
                                  'accminx':dataArray[:,6], 'accminy':dataArray[:,7], 'accminz':dataArray[:,8],
                                  'accabsx':dataArray[:,9], 'accabsy':dataArray[:,10], 'accabsz':dataArray[:,11],
                                  'accavgres':dataArray[:,12], 'Dx':dataArray[:,13], 'Dy':dataArray[:,14], 'Dz':dataArray[:,15],
                                  'Da1':dataArray[:,16], 'Da2':dataArray[:,17], 'Da3':dataArray[:,18], 'Db1':dataArray[:,19], 'Db2':dataArray[:,20], 'Db3':dataArray[:,21],
                                  'D1':dataArray[:,22], 'D2':dataArray[:,23], 'D3':dataArray[:,24], 'frex':dataArray[:,25], 'frey':dataArray[:,26], 'frez':dataArray[:,27], 'types':statusList})
        return featureandtype
        
    def getRegressionFeaturePandasFromDataFile(self,source):
        f = open(source)
        totalLines=0
        while f.readline():
            totalLines=totalLines+1
        dataArray=np.ones((totalLines,28), dtype=float)    
        statusList=("Def"+' ')*totalLines
        statusList=statusList.split(' ')
        statusList.remove('')

        file=open(source)
        linecount=0
        while 1:
            line=file.readline()
            if not line:
                break
            line=line.split(',')
            for i in range(0,28):
                dataArray[linecount,i]=float(line[i])
            statusList[linecount]=line[28]
#             print(line[28])
            linecount=linecount+1
        featureandtype=pd.DataFrame({'accavgx':dataArray[:,0], 'accavgy':dataArray[:,1], 'accavgz':dataArray[:,2],
                                  'accmaxx':dataArray[:,3], 'accmaxy':dataArray[:,4], 'accmaxz':dataArray[:,5],
                                  'accminx':dataArray[:,6], 'accminy':dataArray[:,7], 'accminz':dataArray[:,8],
                                  'accabsx':dataArray[:,9], 'accabsy':dataArray[:,10], 'accabsz':dataArray[:,11],
                                  'accavgres':dataArray[:,12], 'Dx':dataArray[:,13], 'Dy':dataArray[:,14], 'Dz':dataArray[:,15],
                                  'Da1':dataArray[:,16], 'Da2':dataArray[:,17], 'Da3':dataArray[:,18], 'Db1':dataArray[:,19], 'Db2':dataArray[:,20], 'Db3':dataArray[:,21],
                                  'D1':dataArray[:,22], 'D2':dataArray[:,23], 'D3':dataArray[:,24], 'frex':dataArray[:,25], 'frey':dataArray[:,26], 'frez':dataArray[:,27], 'types':statusList})
        return featureandtype
        
        
        #statusArray=(status+' ')*(totalLines-throwStartLines-throwEndLines)
        #statusArray=statusArray.split(' ')
        #print(statusArray)
        #statusArray.remove('')

        
        
        
        



    
    def geneemptyparseframe(self):
        return(pd.DataFrame({'x':[], 'y':[], 'z':[], 'time':[], 'types':[]}))

    
    def genefeaturesframe(self):
        return(pd.DataFrame({'accavgx':[], 'accavgy':[], 'accavgz':[],
                             'accmaxx':[], 'accmaxy':[], 'accmaxz':[],
                             'accminx':[], 'accminy':[], 'accminz':[],
                             'accabsx':[], 'accabsy':[], 'accabsz':[],
                             'accavgres':[], 'Dx':[], 'Dy':[], 'Dz':[],
                             'Da1':[], 'Da2':[], 'Da3':[], 'Db1':[], 'Db2':[], 'Db3':[],
                             'D1':[], 'D2':[], 'D3':[], 'frex':[], 'frey':[], 'frez':[], 'types':[]}))
    
    
    
    
            
        
    
    
    
    



