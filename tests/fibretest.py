from eidolon import *
from AtrialFibrePlugin import (
        writeMeshFile,problemFile, calculateGradientDirs, generateSimplexEdgeMap, TriMeshGraph, generateNodeElemMap,
        getElemDirectionAdj, followElemDirAdj, interpolateElemDirections
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

directionalField=[(0,0,0)]*len(nodes)
directionalField[:w*h]=[(1,0,0)]*(w*h) # bottom fibre directional nodes
directionalField[-w*h:]=[(0,1,1)]*(w*h) # top fibre directional nodes


surfacefaces=listSum(selectTetFace(t,bottomnodes+topnodes) for t in tinds) # list of surface face indices
surfacegraph=TriMeshGraph(nodes,surfacefaces)

fields=[('boundaryfield',boundaryfield,'inds'),('directionalField',directionalField,'inds')]
ds=PyDataSet('boxDS',nodes,[('inds',ElemType._Tet1NL,tinds)],fields)
obj=MeshSceneObject('box',ds)    
mgr.addSceneObject(obj)

writeMeshFile('test.mesh',nodes,tinds,boundaryfield,None,3)

with open(problemFile) as p:
    with open('prob.py','w') as o:
        o.write(p.read()%{'inputfile':'test.mesh','outdir':'.'})
        
p=ProblemConf.from_file('prob.py')
solve_pde(p)

outobj=VTK.loadObject('test.vtk')
grad=outobj.datasets[0].getDataField('t')
grad.meta(StdProps._spatial,'inds')
grad.meta(StdProps._topology,'inds')
ds.setDataField(grad)
        

dirs=calculateGradientDirs(ds.getNodes(),generateSimplexEdgeMap(ds.getNodes().n(),tinds),grad)
dirs=convertNodeToElemField(dirs,ds.getIndexSet('inds'))
#dirs=convertElemToNodeField(dirs,ds.getNodes().n(),ds.getIndexSet('inds'))
ds.setDataField(dirs)


#def getElemDirectionAdj(nodes,elems,dirField,adj=None):
#    if not adj:
#        ds=PyDataSet('tmp',nodes,[elems])
#        calculateElemExtAdj(ds)
#        adj=ds.getIndexSet(elems.getName()+MatrixType.adj[1])
#        
#    et=ElemType[elems.getType()]
#    result=IndexMatrix('diradj',elems.n(),4)
#    result.meta(StdProps._elemdata,'True')
#    result.fill(elems.n())
#    
#    def getFaceInDirection(start,direction,enodes):
#        dray=Ray(start,direction)
#        
#        for f,face in enumerate(et.faces):
#            fnodes=[enodes[i] for i in face[:3]]
#            if dray.intersectsTri(*fnodes):
#                return f
#            
#        return None
#    
#    for e,elem in enumerate(elems):
#        edir=vec3(*dirField[e])
#        enodes=[nodes[n] for n in elem] # elem nodes
#        center=et.applyBasis(enodes,0.25,0.25,0.25) # elem center
#            
#        forward=getFaceInDirection(center,edir,enodes)
#        result[e,0]=forward
#        result[e,1]=adj[e,forward]
#        
#        backward=getFaceInDirection(center,-edir,enodes)
#        result[e,2]=backward
#        result[e,3]=adj[e,backward]
#            
#        assert result[e,0]<elems.n()
#        
#    return result
#        
#
#def followElemDirAdj(elemdiradj):
#    result=IndexMatrix('diradj',elemdiradj.n(),4)
#    result.fill(elemdiradj.n())
#    result.meta(StdProps._elemdata,'True')
#    
#    def followElem(start,isForward):
#        curelem=start
#        index=1 if isForward else 3
#        
#        while curelem>=start and elemdiradj[curelem,index]<elemdiradj.n():
#            curelem=elemdiradj[curelem,index]
#            
#        if curelem<start: # previously assigned value, use this since the path from here on is the same
#            return result[curelem,index-1]
#        else:
#            return curelem
#    
#    for e in range(elemdiradj.n()):
#        forward=followElem(e,True)
#        result[e,0]=forward
#        result[e,1]=elemdiradj[forward,0]
#        
#        backward=followElem(e,False)
#        result[e,2]=backward
#        result[e,3]=elemdiradj[backward,2]
#        
#    return result
#
#
#def interpolateElemDirections(elems,elemFollow,gradField,directionalField):
#    et=ElemType[elems.getType()]
#    elemdirField=RealMatrix('elemdirField',elems.n(),3)
#    elemdirField.meta(StdProps._elemdata,'True')
#    elemdirField.fill(0)
#    
#    def getDirectionalFaceValue(elem,face):
#        endinds=elems[elem]
#        faceinds=[endinds[f] for f in et.faces[face]]
#        return avg(vec3(*directionalField[f]) for f in faceinds).norm()
#    
#    for e in range(elems.n()):
#        elem1,face1,elem2,face2=elemFollow[e]
#        
#        dir1=getDirectionalFaceValue(elem1,face1)
#        dir2=getDirectionalFaceValue(elem2,face2)
#        grad=avg(gradField[i] for i in elems[e])
#        
#        elemdirField[e]=tuple(dir1*grad+dir2*(1-grad))
#        
#        
#    return elemdirField
        

elemdiradj=getElemDirectionAdj(ds.getNodes(),ds.getIndexSet('inds'),dirs)
elemFollow=followElemDirAdj(elemdiradj)

elemDirField=interpolateElemDirections(ds.getIndexSet('inds'),elemFollow,grad,directionalField)
#elemDirField=convertElemToNodeField(elemDirField,ds.getNodes().n(),ds.getIndexSet('inds'))
elemDirField.meta(StdProps._spatial,'inds')
elemDirField.meta(StdProps._topology,'inds')
ds.setDataField(elemDirField)


rep=obj.createRepr(ReprType._line,0)
mgr.addSceneObjectRepr(rep)

mid=ElemType.Tet1NL.basis(0.25,0.25,0.25)

elemnodes=[ElemType.Tet1NL.applyCoeffs([nodes[e] for e in elem],mid) for elem in tinds]
elemobj=MeshSceneObject('elemobj',PyDataSet('elemobjds',elemnodes,[],[elemDirField,dirs,convertNodeToElemField(grad,ds.getIndexSet('inds'))]))
mgr.addSceneObject(elemobj)

rep=elemobj.createRepr(ReprType._glyph,0,externalOnly=False,drawInternal=True,glyphname='arrow',glyphscale=(0.01,0.01,0.03),
                   dfield=elemDirField.getName(),vecfunc=VecFunc._Linear,matname='Rainbow',field=grad.getName())

mgr.addSceneObjectRepr(rep)

#rep=obj.createRepr(ReprType._volume,matname='Rainbow',field=elemDirField.getName())
#mgr.addSceneObjectRepr(rep)

mgr.setAxesType(AxesType._originarrows) # top right corner axes
mgr.controller.setRotation(0.8,0.8)
mgr.setCameraSeeAll()

