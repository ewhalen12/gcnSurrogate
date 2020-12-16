import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# toss out the wost 10% of designs
def filterbyDisp(graphList, pctCutoff):
    maxes = [max(np.abs(graph.y.numpy().flatten())) for graph in graphList]
    df = pd.DataFrame(maxes, columns=['maxes'])
    mask = df['maxes'].rank(pct=True) <= pctCutoff
    return [g for g,b in zip(graphList, mask) if b]

# partition into train, validate and test
def partitionGraphList(allGraphs):
    allIds = list(range(len(allGraphs)))
    other, testIds = train_test_split(allIds, test_size=0.2, shuffle=True, random_state=1234) # 20% test
    trainIds, valIds = train_test_split(other, test_size=0.15, shuffle=True, random_state=1234) # 15% val 15% test
    trainGraphs = [allGraphs[i] for i in trainIds]
    valGraphs = [allGraphs[i] for i in valIds]
    testGraphs = [allGraphs[i] for i in testIds]
    return trainGraphs, valGraphs, testGraphs