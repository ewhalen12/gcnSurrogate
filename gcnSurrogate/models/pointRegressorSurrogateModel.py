import sys
import os
from time import time
import numpy as np
import torch
import torch_geometric as tg
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import pickle

from ..util.gcnSurrogateUtil import *

class PointRegressor():
    
###############################################################################
    def __init__(self, algorithm):
        super(PointRegressor, self).__init__()
        self.algorithm = algorithm
    
###############################################################################
    def logTransFunc(self, x):
        return np.sign(x)*np.log(10.0*np.abs(x)+1.0)
    
###############################################################################
    def invLogTransFunc(self, y):
        return np.sign(y)*(np.exp(np.abs(y))-1.0)/10.0
    
###############################################################################
    def makeLoadsBinary(self, graph):
        graph.x = (graph.x!=0).double()
        return graph
    
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
    def applyAllTransforms(self, graphList):
        transformedGraphList = [g.clone() for g in graphList] # deep copy
        for graph in transformedGraphList:
            if self.binaryLoads:
                graph = self.makeLoadsBinary(graph)
            if self.ssTrans:
                if self.flatten:
                    graph.y = torch.as_tensor(self.ss.transform(graph.y.reshape(-1,1).cpu()).reshape(-1,2), dtype=torch.float)
                else:
                    graph.y = torch.as_tensor(self.ss.transform(graph.y.reshape(1,-1).cpu()).reshape(-1,2), dtype=torch.float)
            if self.logTrans: 
                graph.y = self.logTransFunc(graph.y)
        return transformedGraphList
    
###############################################################################
    def applyAllInvTransforms(self, out):
        if self.logTrans: 
            out = self.invLogTransFunc(out)
        if self.ssTrans:
            if self.flatten:
                out = self.ss.inverse_transform(out.reshape(-1,1)).reshape(-1,2)
            else:
                out = self.ss.inverse_transform(out.reshape(1,-1)).reshape(-1,2)
        return out
    
###############################################################################
    def trainModel(self, trainGraphs, valGraphs, saveDir=None, flatten=False, logTrans=False, ssTrans=True, binaryLoads=True,
                  useXFeatures=True):
        t = time()
        self.flatten = flatten
        self.logTrans = logTrans
        self.ssTrans = ssTrans
        self.binaryLoads = binaryLoads
        self.useXFeatures = useXFeatures
    
        # data transformation
        self.fitSS(trainGraphs)
        trainGraphsScaled = self.applyAllTransforms(trainGraphs)
        valGraphsScaled = self.applyAllTransforms(valGraphs)

        # put data in tabular form
        Xtrain = np.vstack([graph.pos.numpy().flatten() for graph in trainGraphsScaled])
        if self.useXFeatures:
            XtrainFeatures = np.vstack([graph.x.numpy().flatten() for graph in trainGraphsScaled])
            Xtrain = np.concatenate([Xtrain, XtrainFeatures], axis=1)        

        Ytrain = np.vstack([graph.y.numpy().flatten() for graph in trainGraphsScaled])
    
        # build regressor for each output
        self.allModels = []
        for i in range(Ytrain.shape[1]):
            rf = RandomForestRegressor()
            rf.fit(Xtrain, Ytrain[:,i])
            self.allModels.append(rf)

        # save model
        if saveDir:
            if not os.path.exists(saveDir): os.makedirs(saveDir, exist_ok=True)
            self.checkptFile = os.path.join(saveDir, 'trainedRfModel')
            with open(self.checkptFile, 'wb') as fb:
                pickle.dump(self.__dict__, fb)
            
        print(f'trained {len(self.allModels)} random forest models in {time()-t:.2f} seconds')
            
###############################################################################
    def predict(self, inputs):
        NUM_OUTPUTS = len(self.allModels)
        NUM_SAMPLES = len(inputs)
        
        # prep data
        inputsScaled = self.applyAllTransforms(inputs)
        Xinput = np.vstack([graph.pos.numpy().flatten() for graph in inputsScaled])
        
        if self.useXFeatures:
            XinputFeatures = np.vstack([graph.x.numpy().flatten() for graph in inputsScaled])
            Xinput = np.concatenate([Xinput, XinputFeatures], axis=1)  
        
        predArray = np.zeros((NUM_SAMPLES, NUM_OUTPUTS))
        
        # run inference for each model
        for i in range(NUM_OUTPUTS):
            predArray[:,i] = self.allModels[i].predict(Xinput) 
        
        # reshape and inverse transpose
        preds = []
        for j in range(NUM_SAMPLES):
            yPred = predArray[j,:].reshape(-1,2)
            yPred = self.applyAllInvTransforms(yPred)
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
            
