import os
import glob
import json
import numpy as np
import torch
import torch_geometric as tg

###############################################################################
def loadConmechGraphs(modelDir, nodeDims=[0,2], supportDims=[0,2], 
                      loadDims=[2]):
    graphList = []
    for modelFile in glob.glob(os.path.join(modelDir, '*.json')):
        g = _loadConmechGraph(modelFile, nodeDims=nodeDims, 
                              supportDims=supportDims, loadDims=loadDims)
        graphList.append(g)

    return graphList


###############################################################################    
def _loadConmechGraph(modelFile, nodeDims=[0,2], supportDims=[0,2], 
                      loadDims=[2]):

    # read raw json
    with open(modelFile) as f:
        resultData = json.load(f)

    # parse
    pos = np.array([d['point'] for d in resultData['nodes']])
    edges = np.array([d['end_node_inds'] for d in resultData['elements']])
    disp = np.array(resultData['results'])
    supports = np.array([d['condition'] for d in resultData['supports']]).astype(float)
    # resultData['designParams']

    loads = np.zeros_like(supports)
    for load in resultData['loadcases']['0']['ploads']:
        i = load['node_ind']
        loads[i,:3] = load['force']
        loads[i,3:] = load['moment']

    # filter dimensions
    pos = pos[:,nodeDims]
    disp = disp[:,nodeDims]
    supports = supports[:,supportDims]
    loads = loads[:,loadDims]
    supportsAndLoads = np.hstack([supports, loads])

    # build graph
    graph = tg.data.Data(pos = torch.from_numpy(pos).float(),
                         edge_index = torch.tensor(edges.T, dtype=torch.long),
                         x = torch.tensor(supportsAndLoads),
                         y = torch.from_numpy(disp).float())
    graph.edge_index = tg.utils.to_undirected(graph.edge_index)

    return graph