import sys
import os
from time import time
import numpy as np
import torch
import torch_geometric as tg
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import pickle

sys.path.append('./util')
from gcnSurrogateUtil import *

class PointRegressor():
    
###############################################################################
    def __init__(self, algorithm):
        super(PointRegressor, self).__init__()
        self.algorithm = algorithm
    
###############################################################################
    def logTrans(self, x):
        return np.sign(x)*np.log(10.0*np.abs(x)+1.0)
    
###############################################################################
    def invLogTrans(self, y):
        return np.sign(y)*(np.exp(np.abs(y))-1.0)/10.0
    
###############################################################################
    def fitSS(self, graphList):
        self.ss = StandardScaler()
        if self.flatten:
            allResponses = np.empty((0,1))
            for graph in graphList:
                allResponses = np.vstack([allResponses, graph.y.reshape(-1,1)])
        else:
            allResponses = np.empty((0,graphList[0].y.numpy().size))
            for graph in graphList:
                allResponses = np.vstack([allResponses, graph.y.reshape(1,-1)])
        self.ss.fit(allResponses)
        return
    
###############################################################################
    def applySS(self, graphList):
        transformedGraphList = [g.clone() for g in graphList] # deep copy
        for graph in transformedGraphList:
            if self.ssTrans:
                if self.flatten:
                    graph.y = torch.as_tensor(self.ss.transform(graph.y.reshape(-1,1).cpu()).reshape(-1,2), dtype=torch.float)
                else:
                    graph.y = torch.as_tensor(self.ss.transform(graph.y.reshape(1,-1).cpu()).reshape(-1,2), dtype=torch.float)
            if self.logTrans: 
                graph.y = self.logTrans(graph.y)
        return transformedGraphList
    
###############################################################################
    def applyInvSS(self, out):
        if self.logTrans: 
            out = self.invLogTrans(out)
        if self.ssTrans:
            if self.flatten:
                out = self.ss.inverse_transform(out.reshape(-1,1)).reshape(-1,2)
            else:
                out = self.ss.inverse_transform(out.reshape(1,-1)).reshape(-1,2)
        return out
    
###############################################################################
    def trainModel(self, trainGraphs, valGraphs, saveDir=None, flatten=False, logTrans=True, ssTrans=True):
        t = time()
        # data transformation
        self.flatten = flatten
        self.logTrans = logTrans
        self.ssTrans = ssTrans
        self.fitSS(trainGraphs)
        trainGraphsScaled = self.applySS(trainGraphs)
        valGraphsScaled = self.applySS(valGraphs)

        # put data in tabular form
        Xtrain = np.vstack([graph.pos.numpy().flatten() for graph in trainGraphsScaled])
        Ytrain = np.vstack([graph.y.numpy().flatten() for graph in trainGraphsScaled])
    
        # build regressor for each output
        self.allModels = []
        for i in range(Xtrain.shape[1]):
            rf = RandomForestRegressor()
            rf.fit(Xtrain, Ytrain[:,i])
            self.allModels.append(rf)

        # save model
        if saveDir:
            if not os.path.exists(saveDir): os.makedirs(saveDir, exist_ok=True)
            self.checkptFile = os.path.join(saveDir, 'trainedRfModel')
            with open(self.checkptFile, 'wb') as fb:
                pickle.dump(self.__dict__, fb)
            
        print(f'trained {Xtrain.shape[1]} random forest models in {time()-t:.2f} seconds')
            
###############################################################################
    def predict(self, inputs):
        NUM_OUTPUTS = len(self.allModels)
        NUM_SAMPLES = len(inputs)
        
        # prep data
        inputsScaled = self.applySS(inputs)
        Xinput = np.vstack([graph.pos.numpy().flatten() for graph in inputsScaled])
        predArray = np.zeros((NUM_SAMPLES, NUM_OUTPUTS))
        
        # run inference for each model
        for i in range(NUM_OUTPUTS):
            predArray[:,i] = self.allModels[i].predict(Xinput) 
        
        # reshape and inverse transpose
        preds = []
        for j in range(NUM_SAMPLES):
            yPred = predArray[j,:].reshape(-1,2)
            yPred = self.applyInvSS(yPred)
            preds.append(yPred)
        
        return preds

###############################################################################
    def testModel(self, inputs, baselineRef=None, level='set'):
        preds = self.predict(inputs)
        if baselineRef: baselineRef = [b.y.cpu().numpy() for b in baselineRef]
        return computeFieldLossMetrics([g.y.cpu().numpy() for g in inputs], 
                                          preds, 
                                          baselineRef=baselineRef, level=level)
    
###############################################################################
    def loadModel(self, modelFile):
        self.__dict__.clear()
        with open(modelFile, 'rb') as fb:
            newDict = pickle.load(fb)
            self.__dict__.update(newDict) 
            
