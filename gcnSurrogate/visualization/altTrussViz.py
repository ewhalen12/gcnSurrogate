import sys
import numpy as np
import pandas as pd
import altair as alt
import geojson

from ..util.gcnSurrogateUtil import *

###############################################################################
def plotTruss(graph, showDeformed=False, defScale=10, showUndeformed=True, prediction=None, 
              baseColor='#000000', fadedColor='#C7D5E5', brightColor='#0096C7', loadColor='#0096C7', width=600, 
              domX='auto', domY='auto', lineWidth=2.0, lineOpacity=1.0, 
              showPoints=False, pointSize=1, withoutConfigure=False, background='white', showAxis=False,
              showSupports=False, showLoads=False, loadTheta=-90, loadSize=3000, supportSize=3000, autosize=True):
    
    dfPoints = pd.DataFrame(graph.pos.numpy(), columns=['x', 'y'])
    domX = [dfPoints['x'].min(), dfPoints['x'].max()] if domX=='auto' else domX
    domY = [dfPoints['y'].min(), dfPoints['y'].max()] if domY=='auto' else domY
    rangeX = domX[1]-domX[0]
    rangeY = domY[1]-domY[0]
    height = width*rangeY/rangeX  # should guarentee equal aspect ratio
    chartList = []

    if showUndeformed and not showDeformed:
        chartList.extend(plotGraph(graph.pos.numpy(), graph.edge_index, domX, domY, color=baseColor, 
                                   lineWidth=lineWidth, lineOpacity=lineOpacity, showPoints=showPoints, pointSize=pointSize, showAxis=showAxis))
        
    if showDeformed and not showUndeformed:
        chartList.extend(plotGraph((graph.pos+defScale*graph.y).numpy(), graph.edge_index, domX, domY, color=baseColor, 
                                   lineWidth=lineWidth, lineOpacity=lineOpacity, showPoints=showPoints, pointSize=pointSize, showAxis=showAxis))
        
    if showUndeformed and showDeformed:
        chartList.extend(plotGraph(graph.pos.numpy(), graph.edge_index, domX, domY, color=fadedColor, 
                                   lineWidth=lineWidth, lineOpacity=lineOpacity, showPoints=showPoints, pointSize=pointSize, showAxis=showAxis))
        chartList.extend(plotGraph((graph.pos+defScale*graph.y).numpy(), graph.edge_index, domX, domY, color=baseColor, 
                                   lineWidth=lineWidth, lineOpacity=lineOpacity, showPoints=showPoints, pointSize=pointSize, showAxis=showAxis))
        
    if prediction is not None:
        chartList.extend(plotGraph((graph.pos+defScale*prediction).numpy(), graph.edge_index, domX, domY, color=brightColor, 
                                   lineWidth=lineWidth, lineOpacity=lineOpacity, showPoints=showPoints, pointSize=pointSize, showAxis=showAxis))
        
    if showSupports:
        chartList.extend(plotSupports(graph, color=loadColor, size=supportSize))
        
    if showLoads:
        chartList.extend(plotLoads(graph, color=loadColor, size=loadSize, theta=loadTheta))
        
    if withoutConfigure:
        fig = alt.layer(*chartList).properties(width=width, height=height)
        return fig
    else:
        fig = alt.layer(*chartList, background=background).configure_view(strokeOpacity=0).properties(width=width, height=height)
        if autosize: fig.autosize=alt.AutoSizeParams(contains='padding', type='fit')
        return fig


###############################################################################
def plotGraph(pos, edge_index, domX, domY, color='#4C78A8', showPoints=False, lineWidth=2.0, lineOpacity=1, pointSize=1.0, showAxis=False):
    chartList = []
    pointOpacity = 1 if showPoints else 0
    df = pd.DataFrame(pos, columns=['x', 'y'])
    axis = alt.Axis() if showAxis else None
    base = alt.Chart(df).mark_circle().encode(
        alt.X('x:Q', scale=alt.Scale(domain=domX, nice=False), axis=axis), 
        alt.Y('y:Q', scale=alt.Scale(domain=domY, nice=False), axis=axis),
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

# ###############################################################################
# def plotSupports(graph, width, height, domX, domY, color='#4C78A8', size=35):
#     rangeX = domX[1]-domX[0]
#     rangeY = domY[1]-domY[0]
    
#     numPts = graph.pos.shape[0]
#     chartList = []
#     for i in range(numPts):
#         if graph.x[i,0]==1 or graph.x[i,1]==1:
#             pos = graph.pos.numpy()[i,:]
#             pixelPos = [width*(pos[0]-domX[0])/rangeX, -height*(pos[1]-domY[1])/rangeY]
#             if graph.x[i,0]==1 and graph.x[i,1]==1:
#                 chartList.append(plotPinSupport(pixelPos, size=size, color=color))
#             elif graph.x[i,1]==1:
#                 chartList.append(plotPinSupport(pixelPos, size=size, color=color))
#                 chartList.append(plotRollerSupport(pixelPos, size=size, color=color))
#             else:
#                 print('WARNING: VERTICAL ROLLER NOT IMPLEMENTED')
                
#     return chartList

###############################################################################
def plotSupports(graph, color='#4C78A8', size=100):
    posxdf = pd.DataFrame(np.hstack([graph.pos,graph.x]), columns=['x','y','sx','sy', 'load'])
    # pins
    wt = 0.5
    ht = 0.866
    triangleSvg = f"M 0 0 L {wt} {ht} L {-wt} {ht} Z "
    pinChart = alt.Chart(posxdf.loc[(posxdf.sx==1) & (posxdf.sy==1)]).mark_point(
        shape=triangleSvg, 
        fill=color, 
        stroke=color, 
        opacity=1.0, 
        size=size).encode(x='x', y='y')

    # x-rollers
    r = 0.15
    w = 0.25
    y = ht+r
    c = 0.05
    rollerSvg = f"""M 0 0 L {wt} {ht} L {-wt} {ht} Z 
                        M{w-r},{y} A{c},{c},0,0,1,{w+r},{y}A{c},{c},0,1,1,{w-r},{y} 
                        M{-w-r},{y} A{c},{c},0,0,1,{-w+r},{y} A{c},{c},0,1,1,{-w-r},{y}"""
    rollerXChart = alt.Chart(posxdf.loc[(posxdf.sx==0) & (posxdf.sy==1)]).mark_point(
        shape=rollerSvg, 
        fill=color, 
        stroke=color, 
        opacity=1.0, 
        size=size).encode(x='x', y='y')
                
    return [pinChart, rollerXChart]

# ###############################################################################
# def plotLoads(graph, width, height, domX, domY, color='#4C78A8', size=10, theta=-90):
#     rangeX = domX[1]-domX[0]
#     rangeY = domY[1]-domY[0]
    
#     numPts = graph.pos.shape[0]
#     chartList = []
#     for i in range(numPts):
#         if graph.x[i,2]!=0:
#             pos = graph.pos.numpy()[i,:]
#             pixelPos = [width*(pos[0]-domX[0])/rangeX, -height*(pos[1]-domY[1])/rangeY]
#             magnitude = np.abs(graph.x[i,2])
#             chartList.append(plotForce(pixelPos, size=size, color=color, theta=theta))
# #             chartList.append(plotForceLabel(magnitude, pos.tolist()))
#     return chartList

###############################################################################
def plotLoads(graph, color='#4C78A8', size=100, theta=-90):
    posxdf = pd.DataFrame(np.hstack([graph.pos,graph.x]), columns=['x','y','sx','sy', 'load'])
    
    L=3
    w=0.1
    arPts = np.array([[0.0, 0.0], 
                       [-4*w, -np.sqrt(3.0)/2.0], 
                       [-w, -np.sqrt(3.0)/2.0], 
                       [-w, -L], 
                       [w, -L], 
                       [w, -np.sqrt(3.0)/2.0], 
                       [4*w, -np.sqrt(3.0)/2.0]])

    # rotate
    theta = np.radians(theta)+np.pi/2
    c, s = np.cos(theta), np.sin(theta)
    R = np.array(((c, -s), (s, c)))
    arPts = np.matmul(arPts,R) 

    forceChart = alt.Chart(posxdf.loc[posxdf.load!=0]).mark_point(
        shape=polylineToSvg(arPts), 
        fill=color, 
        stroke=color, 
        opacity=1.0, 
        size=size).encode(x='x', y='y')

    return [forceChart]

###############################################################################
def polylineToSvg(coords):
    return 'M'+' '.join([f'L {i} {j}' for i,j in coords])[1:]

###############################################################################
# def plotPinSupport(pixelPos, size=35, color='#4C78A8'):
#     triPts = np.array([[0, 0], 
#                        [-1/2.0, np.sqrt(3.0)/2.0], 
#                        [1/2.0, np.sqrt(3.0)/2.0], 
#                        [0, 0]])
#     triPts *= size
#     triPts += pixelPos
#     triangle = geojson.Polygon([triPts.tolist()])
#     tFig = alt.Chart(triangle).mark_geoshape(color=color).encode().properties(
#         projection={'type': 'identity', 'scale': 1, 'translate':[0,0]})
#     return tFig

###############################################################################
# def plotRollerSupport(pixelPos, size=35, color='#4C78A8'):
#     theta = np.linspace(0,2*np.pi,num=32,endpoint=True).reshape(-1,1)
#     cirPts = np.concatenate([np.cos(theta), np.sin(theta)], axis=1)
    
#     cirPts *= size/7
#     cirPts += pixelPos
    
#     cirPts1 = cirPts + [size/3,size]
#     cirPts2 = cirPts + [-size/3,size]
    
#     cir1 = geojson.Polygon([cirPts1.tolist()])
#     cir2 = geojson.Polygon([cirPts2.tolist()])
#     featCol = geojson.FeatureCollection([cir1, cir2])
#     cirData = alt.InlineData(values=featCol, format=alt.DataFormat(property='features',type='json')) 
#     cFig = alt.Chart(cirData).mark_geoshape(color=color).encode().properties(
#         projection={'type': 'identity', 'scale': 1, 'translate':[0,0]})
#     return cFig

###############################################################################
# def plotForce(pixelPos, size=10, color='#4C78A8', L=3, w=0.1, theta=-90):
#     arPts = np.array([[0.0, 0.0], 
#                        [-4*w, -np.sqrt(3.0)/2.0], 
#                        [-w, -np.sqrt(3.0)/2.0], 
#                        [-w, -L], 
#                        [w, -L], 
#                        [w, -np.sqrt(3.0)/2.0], 
#                        [4*w, -np.sqrt(3.0)/2.0], 
#                        [0.0, 0.0]])
    
#     # rotate
#     theta = np.radians(theta)+np.pi/2
#     c, s = np.cos(theta), np.sin(theta)
#     R = np.array(((c, -s), (s, c)))
#     arPts = np.matmul(arPts,R) 
    
#     arPts *= size
#     arPts += pixelPos
#     arrow = geojson.Polygon([arPts.tolist()])
#     aFig = alt.Chart(arrow).mark_geoshape(color=color).encode().properties(
#         projection={'type': 'identity', 'scale': 1, 'translate':[0,0]})
#     aFig
#     return aFig

###############################################################################
# def plotForceLabel(val, pos, size=25, shift=[1,5], unitStr='Kips'):
#         return alt.Chart(
#             {'values': [{"text": f'{val}\n{unitStr}', 
#                          'x':pos[0]+shift[0], 
#                          'y':pos[1]+shift[1]}]}
#         ).mark_text(size=size, align='left', font='Arial', lineBreak='\n').encode(
#             x='x:Q',
#             y='y:Q',
#             text="text:N"
#         ).properties()

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


