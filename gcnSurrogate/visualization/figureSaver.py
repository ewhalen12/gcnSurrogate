import subprocess
import os
from PIL import Image
import shutil
from svgutils.compose import Unit
import svgutils.transform as sg
from collections.abc import Iterable 

# The width/height properties of an altair-viz chart apply to the plotting area
# only (exclude legend, title, etc.) unless 
# .autosize = alt.AutoSizeParams(contains='padding', type='fit') is set, in which
# case they apply to the entire chart. Charts with autosize can be layered but 
# cannot be concatenated. Autosize also cannot be applied to concatenated charts.
# For this reason there is no way to precisely control the size oc concatenated
# charts in altair-viz.

# As a workaround, this function takes a list of altair charts and their respective 
# coordinates on a canvas. It saves each chart as an svg, reloads the data with svgutil,
# and places them in the desired location. Both png and eps files can be exported.

###############################################################################        
def chartsToFigure(chartData, figName, baseDir, canvasSize=[600, 400], savePng=True, saveEps=False):
    # create a fresh directory
    figDir = os.path.join(baseDir, figName)
    figSvg = os.path.join(figDir, figName+'.svg')
    if os.path.exists(figDir) and os.path.isdir(figDir):
        shutil.rmtree(figDir)
    os.mkdir(figDir)
    
    # single-chart figure: charts is just a single altair chart
    if not isinstance(chartData, Iterable):
        chartData.save(figSvg, scale_factor=1.0)
        
    # multi-chart figure: chart data is a collection of charts and positions
    else:
        fig = sg.SVGFigure(Unit(canvasSize[0]), Unit(canvasSize[1]))
        
        for i,(chart,pos) in enumerate(chartData):
            chartSvg = os.path.join(figDir, f'{i}.svg')
            chart.save(chartSvg, scale_factor=1.0)
            p = sg.fromfile(chartSvg).getroot()
            p.moveto(pos[0],pos[1])
            fig.append(p)
    
        fig.save(figSvg)
    
    # export in desired file formats
    if savePng:
        figPng = os.path.join(baseDir, figName+'.png')
        svgToPng(figSvg, figPng)
        
    if saveEps:
        figEps = os.path.join(baseDir, figName+'.eps')
        svgToEps(figSvg, figEps)
    
    # return image 
    img = Image.open(figPng)
    print(img.size)
    img.thumbnail([i/4 for i in img.size])
    return img

###############################################################################    
def svgToPng(svgFile, pngFile):
    subprocess.call(['inkscape', '--export-png='+pngFile, svgFile])

###############################################################################    
def svgToEps(svgFile, epsFile):
    subprocess.call(['inkscape', '--export-eps='+epsFile, svgFile]) 
    
    