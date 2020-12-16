import numpy as np
import pandas as pd
import altair as alt

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