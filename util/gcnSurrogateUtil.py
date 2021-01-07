import numpy as np
import pandas as pd
import altair as alt
from sklearn.model_selection import train_test_split
import sklearn.metrics as skm

###############################################################################
# toss out the wost 10% of designs
def filterbyDisp(graphList, pctCutoff):
    maxes = [max(np.abs(graph.y.numpy().flatten())) for graph in graphList]
    df = pd.DataFrame(maxes, columns=['maxes'])
    mask = df['maxes'].rank(pct=True) <= pctCutoff
    return [g for g,b in zip(graphList, mask) if b]

###############################################################################
# partition into train, validate and test. Default 20% test 12% val 68% test
def partitionGraphList(allGraphs, testSize=0.2, valSize=0.15, seed=1234): 
    allIds = list(range(len(allGraphs)))
    if testSize > 0:
        other, testIds = train_test_split(allIds, test_size=testSize, shuffle=True, random_state=seed) 
    else:
        other, testIds = allIds, []
    if valSize > 0:
        trainIds, valIds = train_test_split(other, test_size=valSize, shuffle=True, random_state=seed) 
    else:
        trainIds, valIds = other, []
    trainGraphs = [allGraphs[i] for i in trainIds]
    valGraphs = [allGraphs[i] for i in valIds]
    testGraphs = [allGraphs[i] for i in testIds]
    return trainGraphs, valGraphs, testGraphs

###############################################################################
# plot training history
def plotHistory(history):
    histDf = pd.DataFrame(history)
    histDf['epoch'] = histDf.index
    return alt.Chart(histDf).transform_fold(
            ['train', 'val'],
            as_=['set', 'value']
            ).mark_line().encode(
                alt.X('epoch:Q'),
                alt.Y('value:Q', axis=alt.Axis(title='loss')),
                color=alt.Color('set:N'),
                tooltip=['epoch:Q', 'value:Q']
            ).properties(width=400, height=200)

###############################################################################
def pivotDict(dictOfLists):
    listOfDicts = [dict(zip(dictOfLists,t)) for t in zip(*dictOfLists.values())]
    return listOfDicts

###############################################################################
# Takes in two lists of numpy arrays (representing fields) and calculates
# several loss metrics on the user specified level.
def computeFieldLossMetrics(truths, preds, level='field', baselineRef=None):
    assert len(truths) > 0, 'truths must be a nonmepty list'
    assert len(preds) > 0, 'preds must be a nonmepty list'
    assert len(truths) == len(preds), 'truths and preds should be the same length'
    assert all([t.shape == p.shape for t, p in zip(truths, preds)]), 'the shape of all predicted fields must match the shape of their corresponding truth'
    assert level in ['point', 'point_agg', 'field', 'set'], 'level must be either \'point\', \'point_agg\', \'field\' or \'set\''
    uniformShapes = all([t.shape == truths[0].shape for t in truths])
    
    metrics = {}
    
    # ---point-level metrics---
    errorList = [p-t for t, p in zip(truths, preds)]
    relErrList = [np.abs(p-t)/(np.maximum(np.abs(t), np.abs(p))+0.01) for t, p in zip(truths, preds)]
    
    if level == 'point':
        metrics['errors'] = errorList
        # metrics['relErrs'] = [e/t for e, t in zip(errorList, truths)]
        metrics['relErrs'] = relErrList
        return metrics

    # ---point-aggregate-level metrics---
    if level == 'point_agg':
        assert uniformShapes, 'point-aggregate metrics only apply when all fields have the same dimensions'
        stackedErrors = np.stack(errorList)
        metrics['mse'] = np.mean(stackedErrors**2, axis=0)
        metrics['mae'] = np.mean(np.abs(stackedErrors), axis=0)
        stackedRelErrs = np.stack(relErrList)
        metrics['mre'] = np.mean(stackedRelErrs, axis=0)
        metrics['r2'] = _pointAggR2(truths, preds)
        return metrics
        
    
    # ---field-level metrics--- 
    truePeakList = [np.max(np.abs(t)) for t in truths]
    predPeakList = [np.max(np.abs(p)) for p in preds]
    
    if level == 'field':
        metrics['mse'] = [np.mean(e**2) for e in errorList]
        metrics['mae'] = [np.mean(np.abs(e)) for e in errorList]
        metrics['mre'] = [np.mean(re) for re in relErrList]
        metrics['maxAE'] = [np.max(np.abs(e)) for e in errorList]
        metrics['mae/peak'] = [mae/peak for mae, peak in zip(metrics['mae'], truePeakList)]
        metrics['maxAE/peak'] = [maxE/peak for maxE, peak in zip(metrics['maxAE'], truePeakList)]
        metrics['relEAtPeak'] = [np.abs(tp-pp)/tp for tp, pp in zip(truePeakList, predPeakList)]
        return metrics
    
    # ---set-level metrics---
    if level == 'set':
        concatErrors = np.concatenate(errorList)
        metrics['mse'] = np.mean(concatErrors**2)
        metrics['mae'] = np.mean(np.abs(concatErrors))

        concatRelErrors = np.concatenate(relErrList)
        metrics['mre'] = np.mean(concatRelErrors)

        if baselineRef:
            baselineMetrics = baseline(baselineRef, truths, level='set')
            metrics['mse/baseline'] = metrics['mse'] / baselineMetrics['mse']
            metrics['mae/baseline'] = metrics['mae'] / baselineMetrics['mae']

        metrics['peakR2'] = skm.r2_score(truePeakList, predPeakList)
        if uniformShapes:
            metrics['maxAggR2'] = np.max(_pointAggR2(truths, preds))
            metrics['meanAggR2'] = np.mean(_pointAggR2(truths, preds))
            metrics['minAggR2'] = np.min(_pointAggR2(truths, preds))

        return metrics

###############################################################################
# Calculates loss metrics for a model that always predicts the mean.
# Can be used as a baseline for predictive models. truths is a list of numpy 
# arrays. level is the result level from computeFieldLossMetrics
def baseline(train, test, level='field', avgLevel=None):
    assert len(train) > 0, 'train must be a nonmepty list'
    assert len(test) > 0, 'test must be a nonmepty list'
    assert level in ['point', 'point_agg', 'field', 'set'], 'level must be either \'point\', \'point_agg\', \'field\' or \'set\''
    assert avgLevel in ['point', 'set', None], 'if specified, avgLevel must be either \'point\' or \'set\''

    uniformFields = all([train[0].shape == t.shape for t in train])

    if (avgLevel == 'point') or (not avgLevel and uniformFields):
        assert uniformFields, 'point-based averaging requires that all fields are the same shape'
        avgField = np.mean(train, axis=0)
        preds = [avgField]*len(test)
    else:
        avgVal = np.mean([np.mean(t) for t in train])
        preds = [np.full(t.shape, avgVal) for t in test]
    
    return computeFieldLossMetrics(test, preds, level=level)


###############################################################################
# helper functions
def _pointAggR2(truths, preds):
    stackedPreds = np.stack(preds)
    stackedTruths = np.stack(truths)
    pointAggR2 = np.zeros(preds[0].shape)
    for ij in np.ndindex(preds[0].shape):
        pointAggR2[ij] = skm.r2_score(stackedTruths[(slice(None),)+ij], 
                                      stackedPreds[(slice(None),)+ij])
    return pointAggR2