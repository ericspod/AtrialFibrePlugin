from eidolon import *
import os
from AtrialFibrePlugin import (
        writeMeshFile,problemFile, calculateGradientDirs, generateSimplexEdgeMap, TriMeshGraph, generateNodeElemMap,
        getElemDirectionAdj, followElemDirAdj, calculateMeshGradient, estimateThickness
        )

from sfepy.base.conf import ProblemConf
from sfepy.applications import solve_pde

import itertools

def selectTetFace(tet,acceptInds):
    faces=[]
    for face in itertools.combinations(tet,3):
        if all(f in acceptInds for f in face):
            faces.append(face)
            
    return faces

def convertNodeToElemField(nodefield,elems):
    result=RealMatrix(nodefield.getName(),elems.n(),nodefield.m())
    result.meta(StdProps._elemdata,'True')
    
    for i,elem in enumerate(elems):
        elemvals=[nodefield[e] for e in elem]
        if nodefield.m()==1:
            result[i]=avg(elemvals)
        else:
            result[i]=tuple(map(avg,zip(*elemvals)))
            
    return result
            

def convertElemToNodeField(elemfield,numnodes,elems):
    result=RealMatrix(elemfield.getName(),numnodes,elemfield.m())
    nodemap=generateNodeElemMap(numnodes,elems)
    
    for n,nodes in enumerate(nodemap):
        nodevals=[elemfield[nn] for nn in nodes]
        if elemfield.m()==1:
            result[n]=avg(nodevals)
        else:
            result[n]=tuple(map(avg,zip(*nodevals)))
            
    return result
    

w,h,d=4,4,16
nodes,inds=generateHexBox(w-2,h-2,d-2)

# twist cube around origin in XZ plane
nodes=[vec3(0,(1-n.z())*halfpi,n.x()+1).fromPolar()+vec3(0,n.y(),0) for n in nodes]

bottomnodes=list(range(0,w*h)) # indices of nodes at the bottom of the cube
topnodes=list(range(len(nodes)-w*h,len(nodes))) # indices of nodes at the top of the cube

tinds=[]

# convert hex to tets, warning: index() relies on float precision
hx=ElemType.Hex1NL.xis
tx=divideHextoTet(1)
for hexind in inds:
   for tet in tx:
       tinds.append([hexind[hx.index(t)] for t in tet])
       
# boundary conditions field for laplace solve
boundaryfield=[0]*len(nodes)
boundaryfield[:w*h]=[1]*(w*h) # bottom boundary condition nodes
boundaryfield[-w*h:]=[2]*(w*h) # top boundary condition nodes
#boundaryfield[-int(w*h*4.6)]=2 # test single embedded boundary node

#directionalField=[(0,0,0)]*len(nodes)
#directionalField[:w*h]=[(1,0,0)]*(w*h) # bottom fibre directional nodes
#directionalField[-w*h:]=[(0,1,1)]*(w*h) # top fibre directional nodes


#surfacefaces=listSum(selectTetFace(t,bottomnodes+topnodes) for t in tinds) # list of surface face indices
#surfacegraph=TriMeshGraph(nodes,surfacefaces)

fields=[('boundaryfield',boundaryfield,'inds')] #,('directionalField',directionalField,'inds')]
ds=PyDataSet('boxDS',nodes,[('inds',ElemType._Tet1NL,tinds)],fields)
obj=MeshSceneObject('box',ds)    
mgr.addSceneObject(obj)

nodes=ds.getNodes()
inds=first(ds.enumIndexSets())

eidolon.calculateElemExtAdj(ds)
adj=ds.getIndexSet('inds'+eidolon.MatrixType.adj[1])

#writeMeshFile('test.mesh',nodes,tinds,boundaryfield,None,3)
#
#with open(problemFile) as p:
#    with open('prob.py','w') as o:
#        o.write(p.read()%{'inputfile':'test.mesh','outdir':'.'})
#        
#
#outobj=VTK.loadObject('test.vtk')
#grad=outobj.datasets[0].getDataField('t')
#grad.meta(StdProps._spatial,'inds')
#grad.meta(StdProps._topology,'inds')
#ds.setDataField(grad)



grad=calculateMeshGradient(os.path.join(scriptdir,'grad'),nodes,inds,boundaryfield,VTK)
ds.setDataField(grad)    

dirs=calculateGradientDirs(nodes,generateSimplexEdgeMap(nodes.n(),inds),grad)
#dirs=convertNodeToElemField(dirs,inds)
#dirs=convertElemToNodeField(dirs,ds.getNodes().n(),ds.getIndexSet('inds'))
#ds.setDataField(dirs)
 

elemdiradj=getElemDirectionAdj(nodes,inds,adj,dirs)

thickness=estimateThickness(nodes,inds,elemdiradj)
ds.setDataField(thickness)

#elemFollow=followElemDirAdj(elemdiradj)
#
#elemDirField=interpolateElemDirections(ds.getIndexSet('inds'),elemFollow,grad,directionalField)
##elemDirField=convertElemToNodeField(elemDirField,ds.getNodes().n(),ds.getIndexSet('inds'))
#elemDirField.meta(StdProps._spatial,'inds')
#elemDirField.meta(StdProps._topology,'inds')
#ds.setDataField(elemDirField)


#rep=obj.createRepr(ReprType._line,0)
#mgr.addSceneObjectRepr(rep)
#
#mid=ElemType.Tet1NL.basis(0.25,0.25,0.25)
#
#elemnodes=[ElemType.Tet1NL.applyCoeffs([nodes[e] for e in elem],mid) for elem in tinds]
#elemobj=MeshSceneObject('elemobj',PyDataSet('elemobjds',elemnodes,[],[thickness]))
#mgr.addSceneObject(elemobj)
#
#rep=elemobj.createRepr(ReprType._glyph,0,externalOnly=False,drawInternal=True,glyphname='arrow',glyphscale=(0.01,0.01,0.03),
#                   dfield=elemDirField.getName(),vecfunc=VecFunc._Linear,matname='Rainbow',field=grad.getName())
#
#mgr.addSceneObjectRepr(rep)

#rep=obj.createRepr(ReprType._volume,matname='Rainbow',field=elemDirField.getName())
#mgr.addSceneObjectRepr(rep)

mgr.setAxesType(AxesType._originarrows) # top right corner axes
mgr.controller.setRotation(0.8,0.8)
mgr.setCameraSeeAll()


