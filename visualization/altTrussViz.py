import sys
import numpy as np
import pandas as pd
import altair as alt

sys.path.append('./util')
from gcnSurrogateUtil import *

###############################################################################
def plotTruss(graph, showDeformed=False, defScale=10, showUndeformed=True, prediction=None, 
               baseColor='#4C78A8', fadedColor='#C7D5E5', brightColor='#0AD6FF', width=600, 
               height=300, z=10, lineWidth=2.0, lineOpacity=1.0, showPoints=False, pointSize=1, 
               withoutConfigure=False, background='white'):
    
    dfPoints = pd.DataFrame(graph.pos.numpy(), columns=['x', 'y'])
    cg = dfPoints.mean().values
    domX = [cg[0]-0.5*width/z,cg[0]+0.5*width/z]
    domY = [cg[1]-0.5*height/z,cg[1]+0.5*height/z]
    chartList = []

    if showUndeformed and not showDeformed:
        chartList.extend(plotGraph(graph.pos.numpy(), graph.edge_index, domX, domY, color=baseColor, 
                                   lineWidth=lineWidth, lineOpacity=lineOpacity, showPoints=showPoints, pointSize=pointSize))
        
    if showDeformed and not showUndeformed:
        chartList.extend(plotGraph((graph.pos+defScale*graph.y).numpy(), graph.edge_index, domX, domY, color=baseColor, 
                                   lineWidth=lineWidth, lineOpacity=lineOpacity, showPoints=showPoints, pointSize=pointSize))
        
    if showUndeformed and showDeformed:
        chartList.extend(plotGraph(graph.pos.numpy(), graph.edge_index, domX, domY, color=fadedColor, 
                                   lineWidth=lineWidth, lineOpacity=lineOpacity, showPoints=showPoints, pointSize=pointSize))
        chartList.extend(plotGraph((graph.pos+defScale*graph.y).numpy(), graph.edge_index, domX, domY, color=baseColor, 
                                   lineWidth=lineWidth, lineOpacity=lineOpacity, showPoints=showPoints, pointSize=pointSize))
        
    if prediction is not None:
        chartList.extend(plotGraph((graph.pos+defScale*prediction).numpy(), graph.edge_index, domX, domY, color=brightColor, 
                                   lineWidth=lineWidth, lineOpacity=lineOpacity, showPoints=showPoints, pointSize=pointSize))
    if withoutConfigure:
        return alt.layer(*chartList).properties(width=width, height=height)
    else:
        return alt.layer(*chartList, background=background).configure_view(strokeOpacity=0).properties(width=width, height=height)


###############################################################################
def plotGraph(pos, edge_index, domX, domY, color='#4C78A8', showPoints=False, lineWidth=2.0, lineOpacity=1, pointSize=1.0):
    chartList = []
    pointOpacity = 1 if showPoints else 0
    df = pd.DataFrame(pos, columns=['x', 'y'])
    base = alt.Chart(df).mark_circle().encode(
        alt.X('x:Q', scale=alt.Scale(domain=domX), axis=None), 
        alt.Y('y:Q', scale=alt.Scale(domain=domY), axis=None),
        opacity=alt.value(pointOpacity),
        color = alt.value(color),
        size = alt.value(pointSize)
    )
    chartList.append(base)
    
    start, end = edge_index
    for i,j in zip(start, end):
        if i < j:
            line = alt.Chart(df.iloc[[i,j]]).mark_line().encode(
                        alt.X('x:Q'), 
                        alt.Y('y:Q'),
                        color=alt.value(color),
                        opacity=alt.value(lineOpacity),
                        size=alt.value(lineWidth)
                    )
            chartList.append(line)
    return chartList

###############################################################################
def interactiveErrorPlot(graphList, predictions):
    # get field-level metrics
    resultsDictWide = computeFieldLossMetrics([g.y.cpu().numpy() for g in graphList], 
                                  predictions, 
                                  baselineRef=None, level='field')
    resultsDict = [dict(zip(resultsDictWide,t)) for t in zip(*resultsDictWide.values())]
    dfRes = pd.DataFrame(resultsDict)
    
    # build a dataframe of individual bars
    df = buildBarDf(graphList, predictionList=predictions)
    df = df.merge(dfRes['mse'].reset_index(), left_on='design', right_on='index')
    df['Percentile'] = df['mse'].rank(pct=True)
    
    # create a selector
    selector = alt.selection_single(empty='all', fields=['design'])

    # create a base plot
    base = alt.Chart(df).properties(
    ).add_selection(selector)

    # create the error scatter plot
    points = base.mark_circle(size=60).encode(
        y=alt.Y('Percentile:Q'),
        x=alt.X('mse:Q', scale=alt.Scale(type='log'), axis=alt.Axis(tickCount=5, format=".0e"), title='MSE'),
        opacity = alt.OpacityValue(0.02),
        color=alt.condition(selector, alt.value('#4C78A8'), alt.value('lightgray'), legend=None),
        tooltip=['design:O', 'mse:Q']
    ).properties(width=300, height=300)

    # create a visualizatino of all graphs
    width=450
    height=300
    z=10
    cg = df.mean().values
    domX = [cg[0]-0.5*width/z,cg[0]+0.5*width/z]
    domY = [cg[1]-0.5*height/z,cg[1]+0.5*height/z]
    order = ['undeformed', 'deformed', 'prediction']
    colors = ['#C7D5E5', '#4C78A8', '#0AD6FF']
    trussViz = base.mark_line().encode(
        alt.X('x:Q', scale=alt.Scale(domain=domX), axis=None), 
        alt.Y('y:Q', scale=alt.Scale(domain=domY), axis=None),
        color=alt.Color('type:N', sort=order, scale=alt.Scale(domain=order, range=colors), legend=alt.Legend(title='')),
        detail='bar'
    ).transform_filter(
        selector
    ).properties(width=width, height=height)

    return alt.hconcat(points, trussViz).configure_view(strokeOpacity=0)

###############################################################################
def buildBarDf(graphList, predictionList=None):
    defScale = 80
    allPos = np.empty((0,2))
    allIds = np.empty(0)
    allBars = np.empty(0)
    allTypes = np.empty(0)
    bar = 0
    for l, graph in enumerate(graphList): # for each grpah
        pos = graph.pos.numpy()
        res = (graph.pos+defScale*graph.y).numpy()
        if predictionList is not None: 
            pred = (graph.pos+defScale*predictionList[l]).numpy()
        start, end = graph.edge_index
        for i,j in zip(start, end): # for each beam
            if i < j:
                # undeformed bars
                allPos = np.vstack((allPos, pos[[i,j],:])) # add two rows, one for each endpoint
                allIds = np.append(allIds, [l]*2);
                allBars = np.append(allBars, [bar]*2);
                allTypes = np.append(allTypes, ['undeformed']*2);
                bar += 1

                # deformed bars
                allPos = np.vstack((allPos, res[[i,j],:]))
                allIds = np.append(allIds, [l]*2);
                allBars = np.append(allBars, [bar]*2);
                allTypes = np.append(allTypes, ['deformed']*2);
                bar += 1
                
                # predictions
                if predictionList is not None:
                    allPos = np.vstack((allPos, pred[[i,j],:]))
                    allIds = np.append(allIds, [l]*2);
                    allBars = np.append(allBars, [bar]*2);
                    allTypes = np.append(allTypes, ['prediction']*2);
                    bar += 1
                    
    return pd.DataFrame({'x':allPos[:,0], 'y':allPos[:,1], 'design':allIds, 'bar':allBars, 'type':allTypes})


