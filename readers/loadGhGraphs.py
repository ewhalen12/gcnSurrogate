import sys
import numpy as np
import pandas as pd
import torch
import torch_geometric as tg

def loadGhGraphs(doeFile, NUM_DV=5):
    # read doe file
    df = pd.read_csv(doeFile, header=None)
    NUM_NODES = df.iloc[0,NUM_DV]
    NUM_EDGES = df.iloc[0,NUM_DV+1]
    
    # assign meaningful column names
    varNames = [f'dv{i}' for i in range(NUM_DV)]
    metaColNames = ['NUM_NODES', 'NUM_EDGES', 'parent']
    xCoordNames = [f'n{i}x' for i in range(NUM_NODES)]
    yCoordNames = [f'n{i}y' for i in range(NUM_NODES)]
    xDispNames = [f'n{i}dx' for i in range(NUM_NODES)]
    yDispNames = [f'n{i}dy' for i in range(NUM_NODES)]
    constrNames = ['nPinn', 'nRoller']
    edgeStartNames = [f'e{i}s' for i in range(NUM_EDGES)]
    edgeEndNames = [f'e{i}e' for i in range(NUM_EDGES)]
    df.columns = varNames+metaColNames+xCoordNames+yCoordNames+xDispNames+yDispNames+constrNames+edgeStartNames+edgeEndNames
    
    # loop over designs and convert to graphs
    graphList = []
    for index, row in df.iterrows():
        # get node coordinates and displacements
        points = row[xCoordNames+yCoordNames].values.reshape(-1,2, order='F')
        disp = row[xDispNames+yDispNames].values.reshape(-1,2, order='F')

        # get supports
        supports = np.zeros_like(disp)
        supports[int(row['nPinn']),:] = 1
        supports[int(row['nRoller']),1] = 1

        # get connectivity
        conectivity = row[edgeStartNames+edgeEndNames].values.reshape(2,-1)

        # build graph
        graph = tg.data.Data(pos = torch.from_numpy(points).float(),
                             edge_index = torch.tensor(conectivity, dtype=torch.long),
                             x = torch.tensor(supports),
                             y = torch.from_numpy(disp).float())
        graph.edge_index = tg.utils.to_undirected(graph.edge_index)
        graphList.append(graph)
    return graphList