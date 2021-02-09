import numpy as np
import torch
import torch_geometric as tg
import torch.nn.functional as F
import pyvista as pv

def plotTruss(graphs, screenshot=False, showPointLabels=True, showDeformed=False, defScale=10, showUndeformed=True, 
              prediction=None, baseColor='03045E', fadedColor='EAEBEB', brightColor = '0AD6FF', background=[255,255,255], 
              line_width=1.5):
    # build plotter
    plotter = pv.Plotter(off_screen=True, notebook= not screenshot)
    plotter.set_background(background)
    
    if not isinstance(graphs, list): graphs = [graphs]
    for graph in graphs:
    
        # get node coordinates
        pos = graph.pos.numpy()
        if graph.pos.numpy().shape[-1] == 2:
            pos = np.hstack([pos[:,:1], np.zeros((pos.shape[0], 1)), pos[:,1:]]) # restore y-coords

        # covnert graph edges to lines
        lines = []
        edges = graph.edge_index.numpy()
        for i in range(edges.shape[-1]):
            if edges[0,i] > edges[1,i]:
                lines = lines + [2, edges[0,i], edges[1,i]]

        # get supports
        supported  = np.where(graph.x.numpy()[:,1] == 1)[0]
        free  = np.where(graph.x.numpy()[:,1] == 0)[0]

        # build a mesh and plot
        if showUndeformed:
            mesh = pv.PolyData(pos)
            mesh.lines = lines
            plotter.add_mesh(mesh, 
                             color=fadedColor if showDeformed else baseColor, 
                             line_width=line_width, 
                             point_size=0)
            if showPointLabels:
                plotter.add_point_labels(mesh.points[supported], 
                                         [i for i in supported], 
                                         font_size=10,
                                         text_color='black')
                plotter.add_point_labels(mesh.points[free], 
                                         [i for i in free], 
                                         font_size=10,
                                         text_color='white')
        # build a deformed mesh
        if showDeformed:
            disp = graph.y.numpy()
            if graph.y.numpy().shape[-1] == 2:
                disp = np.hstack([disp[:,:1], np.zeros((disp.shape[0], 1)), disp[:,1:]]) # restore y-coords
            defMesh = pv.PolyData(pos+disp*defScale)
            defMesh.lines = lines
            plotter.add_mesh(defMesh, 
                             color=baseColor, 
                             line_width=line_width, 
                             point_size=0)

        # build a prediction mesh
        if prediction is not None:
            if prediction.shape[-1] == 2:
                prediction = np.hstack([prediction[:,:1], np.zeros((prediction.shape[0], 1)), prediction[:,1:]]) # restore y-coords
            predMesh = pv.PolyData(pos+prediction*defScale)
            predMesh.lines = lines
            plotter.add_mesh(predMesh, 
                             color=brightColor, 
                             line_width=line_width, 
                             point_size=0)
        
    zoom = -55.0
    focusX, focusY = 25.0, -11.0
    cpos = [(focusX, zoom, focusY), (focusX, 0.0, focusY), (0.0, 0.0, 1.0)]
    
    out = plotter.show(window_size=(500,250), cpos=cpos, screenshot=screenshot)
    if screenshot:
        img = out[1]
        mask = np.all(img==background, axis=2)
        img = np.append(img, np.full((img.shape[0],img.shape[1],1), fill_value=255, dtype=np.uint8), axis=2)
        img[mask,3] = 0
        return img