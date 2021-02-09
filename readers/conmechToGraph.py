import json
import numpy as np
import pandas as pd
import torch
import torch_geometric as tg

def loadConmechGraph(resultFile):
    with open(resultFile) as f:
        resultData = json.load(f)
    
    # get node coordinates and displacements
    nodeDf = pd.DataFrame(resultData['node_displacement'].values())
    nodeDf = nodeDf.set_index('node_id')
    nodeDf = nodeDf.sort_values('node_id')

    points = np.array(list(nodeDf['node_pose']))
    points = points[:,[0,2]]                      # keep x, z coords only
    disp = np.array(list(nodeDf['displacement']))
    disp = disp[:,[0,2]] 
    
    # get supports (hardcoded for now)
    supports = np.zeros((19,2))
    supports[11,:] = 1
    supports[17,1] = 1

    # get connectivity
    elemDf = pd.DataFrame(resultData['element_reaction'].values())
    elemDf = elemDf.set_index('element_id')
    elemDf = elemDf.sort_values('element_id')
    conectivity = np.array(list(elemDf['node_ids']), dtype='int64')
    
    # build graph
    graph = tg.data.Data(pos = torch.from_numpy(points).float(),
                         edge_index = torch.tensor(conectivity).t().contiguous(),
                         x = torch.tensor(supports),
                         y = torch.from_numpy(disp).float())
    graph.edge_index = tg.utils.to_undirected(graph.edge_index)

    return graph