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
# old default: 'L16/C32/C64/C128/C256/C128/C128/L64/L2'
###############################################################################
    def __init__(self, device=torch.device('cuda'), heads=8, numInputCords=2,
                 architecture='L16/C32/C64/C128/C256/C512/C256/C128/L64/L2', numOutputs=2, 
                 useConstraints=True):
        super(FeaStNet, self).__init__()
        
        self.device = device
        self.checkptFile = None
        self.architecture = architecture
        self.useConstraints = useConstraints
        self.layers = torch.nn.ModuleList()
        
        # create each layer based on user specs
        # add a relu layer after every linear or conv layer except the last one
        # add batch norm at the beginning and after every conv layer
        archParsed = self.architecture.split('/')
        lastSize = numInputCords
        self.layers.append(tg.nn.BatchNorm(lastSize, momentum=0.3, affine=True, track_running_stats=True))
        for i in range(len(archParsed)-1):
            layerStr = archParsed[i]
            lCode = layerStr[0]
            lSize = int(layerStr[1:])
            assert lCode in ['L', 'C'], 'invalid layer code'
            assert isinstance(lSize, int), 'layer size must be an integer'
            
            if i == 0 and useConstraints: lastSize *= 2
            
            # linear layer
            if lCode == 'L':
                self.layers.append(torch.nn.Linear(lastSize, lSize))
                self.layers.append(torch.nn.ReLU())
                lastSize = lSize
            
            # conv layer
            else:
                self.layers.append(tg.nn.FeaStConv(lastSize, lSize, heads=heads))
                self.layers.append(tg.nn.BatchNorm(lSize, momentum=0.3, affine=True, track_running_stats=True))
                self.layers.append(torch.nn.ReLU())
                lastSize = lSize
        
        # no activation or normalization on output layer
        layerStr = archParsed[-1]
        lCode = layerStr[0]
        assert lCode in ['L', 'C'], 'invalid layer code'
        assert isinstance(lSize, int), 'layer size must be an integer'
            
        if lCode == 'L':
            self.layers.append(torch.nn.Linear(lastSize, numOutputs))
        else:
            self.layers.append(tg.nn.FeaStConv(lastSize, numOutputs, heads=heads))
    
        
###############################################################################
    def forward(self, data):
        x = data.pos
        for i in range(len(self.layers)):
            layer = self.layers[i]
            
            if i == 1 and self.useConstraints:
                x = torch.cat([x, data.x.float()], 1)
            
            # conv layer
            if isinstance(layer, tg.nn.conv.feast_conv.FeaStConv):
                x = layer.forward(x, data.edge_index)
                
            # linear layer
            else:
                x = layer.forward(x)

        return x
    
###############################################################################
    def logTransFunc(self, x):
        return np.sign(x)*np.log(10.0*np.abs(x)+1.0)
    
###############################################################################
    def invLogTransFunc(self, y):
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
                graph.y = self.logTransFunc(graph.y)
        return transformedGraphList
    
###############################################################################
    def applyInvSS(self, out):
        if self.logTrans: 
            out = self.invLogTransFunc(out)
        if self.ssTrans:
            if self.flatten:
                out = self.ss.inverse_transform(out.reshape(-1,1)).reshape(-1,2)
            else:
                out = self.ss.inverse_transform(out.reshape(1,-1)).reshape(-1,2)
        return out
    
###############################################################################
    def trainModel(self, trainGraphs, valGraphs, epochs=10, saveDir=None, batchSize=256, flatten=False, logTrans=False, ssTrans=True,
                   restartFile=None, lr=1e-3, weightDecay=1e-3):
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
        loader = tg.data.DataLoader(trainGraphsScaled, batch_size=batchSize, shuffle=True)

        # prep validation data
        valGraphsScaled = self.applySS(valGraphs)
        valLoader = tg.data.DataLoader(valGraphsScaled, batch_size=1, shuffle=False)

        # prep model
        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=weightDecay)
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
            
###############################################################################
    def getNumParams(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
            