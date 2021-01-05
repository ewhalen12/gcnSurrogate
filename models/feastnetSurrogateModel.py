import sys
import os
from time import time
import numpy as np
import torch
import torch_geometric as tg
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler

sys.path.append('./util')
from gcnSurrogateUtil import *

class FeaStNet(torch.nn.Module):
    
###############################################################################
    def __init__(self, device=torch.device('cuda')):
        super(FeaStNet, self).__init__()
        self.device = device
        self.checkptFile = None
        
        self.norm0 = tg.nn.BatchNorm(2, momentum=0.3, affine=True, track_running_stats=True)
        self.lin0 = torch.nn.Linear(4, 16)
        self.conv0 = tg.nn.FeaStConv(16, 32, heads=8)
        self.normc0 = tg.nn.BatchNorm(32, momentum=0.3, affine=True, track_running_stats=True)
        self.conv1 = tg.nn.FeaStConv(32, 64, heads=8)
        self.normc1 = tg.nn.BatchNorm(64, momentum=0.3, affine=True, track_running_stats=True)
        self.conv2 = tg.nn.FeaStConv(64, 128, heads=8)
        self.normc2 = tg.nn.BatchNorm(128, momentum=0.3, affine=True, track_running_stats=True)
        self.conv3 = tg.nn.FeaStConv(128, 256, heads=8)
        self.normc3 = tg.nn.BatchNorm(256, momentum=0.3, affine=True, track_running_stats=True)
        self.conv4 = tg.nn.FeaStConv(256, 128, heads=8)
        self.normc4 = tg.nn.BatchNorm(128, momentum=0.3, affine=True, track_running_stats=True)
        self.conv5 = tg.nn.FeaStConv(128, 128, heads=8)
        self.normc5 = tg.nn.BatchNorm(128, momentum=0.3, affine=True, track_running_stats=True)
        self.lin1 = torch.nn.Linear(128, 64)
        self.lin2 = torch.nn.Linear(64, 2)
        
###############################################################################
    def forward(self, data):
        x = self.norm0(data.pos)
        x = torch.cat([x, data.x.float()], 1)
        x = self.lin0(x)
        x = F.relu(x)
        x = self.conv0(x, data.edge_index)
        x = self.normc0(x)
        x = F.relu(x)
        x = self.conv1(x, data.edge_index)
        x = self.normc1(x)
        x = F.relu(x)        
        x = self.conv2(x, data.edge_index)
        x = self.normc2(x)
        x = F.relu(x)
        x = self.conv3(x, data.edge_index)
        x = self.normc3(x)
        x = F.relu(x)
        x = self.conv4(x, data.edge_index)
        x = self.normc4(x)
        x = F.relu(x)
        x = self.conv5(x, data.edge_index)
        x = self.normc5(x)
        x = F.relu(x)
        x = self.lin1(x)
        x = F.relu(x)
        x = self.lin2(x)
        return x
    
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
    def trainModel(self, trainGraphs, valGraphs, epochs=10, saveDir=None, batch_size=256, flatten=False, logTrans=True, ssTrans=True,
                   restartFile=None):
        if restartFile:
            print('loading restart file')
            self.loadModel(restartFile)
            self.checkptFile = None
        else: 
            # data transformation settings
            self.flatten = flatten
            self.logTrans = logTrans
            self.ssTrans = ssTrans
            self.fitSS(trainGraphs)
            
        trainGraphsScaled = self.applySS(trainGraphs)
        loader = tg.data.DataLoader(trainGraphsScaled, batch_size=batch_size, shuffle=True)

        # prep validation data
        valGraphsScaled = self.applySS(valGraphs)
        valLoader = tg.data.DataLoader(valGraphsScaled, batch_size=1, shuffle=False)

        # prep model
        optimizer = torch.optim.Adam(self.parameters(), lr=0.0001, weight_decay=10e-4)
        trainHist, valHist = [], []
        bestEpoch = 0
        self.to(self.device)
        if saveDir and not os.path.exists(saveDir): os.makedirs(saveDir, exist_ok=True)

        for epoch in range(epochs):

            ### train ###
            self.train()
            t = time()
            batchHist = []
            for batch in loader:
                batch.to(self.device)
                optimizer.zero_grad()
                out = self(batch)
                loss = F.mse_loss(out, batch.y)
                loss.backward()
                optimizer.step()
                batchHist.append(loss.item())

            trainHist.append(np.mean(batchHist))

            ### validate ###
            batchHist = []
            self.eval()
            with torch.no_grad():
                for batch in valLoader:
                    batch.to(self.device)
                    out = self(batch)
                    loss = F.mse_loss(out, batch.y)
                    batchHist.append(loss.item())
            valHist.append(np.mean(batchHist))

            print(f'epoch: {epoch}   trainLoss: {trainHist[-1]:.4e}   valLoss:{valHist[-1]:.4e}  time: {(time()-t):.2e}')

            if saveDir:
                with open(os.path.join(saveDir, 'trainlog.csv'), 'a') as fp: 
                    fp.write(f'{epoch},{trainHist[-1]},{valHist[-1]},{(time()-t)}\n')

                # if new best model
                if (np.argmin(valHist) == len(valHist)-1):
                    lastCheckpt = self.checkptFile
                    self.checkptFile = os.path.join(saveDir, f'checkpoint_{epoch}')
                    bestEpoch = epoch
                    torch.save(self, self.checkptFile) # save best model
                    if lastCheckpt: os.remove(lastCheckpt) # cleanup
                    
        # load best model
        print(f'loading checkpoint {bestEpoch}')
        self.loadModel(self.checkptFile)
        
        return {'train': trainHist, 'val': valHist}
    
###############################################################################
    def predict(self, inputs):
        # prep data
        inputsScaled = self.applySS(inputs)
        testLoader = tg.data.DataLoader(inputsScaled, batch_size=1, shuffle=False)

        preds = []
        self.eval()
        with torch.no_grad():
            for batch in testLoader:
                batch.to(self.device)
                out = self(batch)
                p = self.applyInvSS(out.cpu().numpy())
                preds.append(p)
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
        saved = torch.load(modelFile)
        for key, val in saved.__dict__.items():
            setattr(self, key, val)

    
    