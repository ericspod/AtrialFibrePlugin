from eidolon import *
from AtrialFibrePlugin import writeMeshFile,problemFile, calculateGradientDirs, generateSimplexEdgeMap, TriMeshGraph

from sfepy.base.conf import ProblemConf
from sfepy.applications import solve_pde

import itertools

def selectTetFace(tet,acceptInds):
    faces=[]
    for face in itertools.combinations(tet,3):
        if all(f in acceptInds for f in face):
            faces.append(face)
            
    return faces
            

w,h,d=16,16,32
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

directionalfield=[(0,0,0)]*len(nodes)
directionalfield[:w*h]=[(1,0,0)]*(w*h) # bottom fibre directional nodes
directionalfield[-w*h:]=[(0,0,1)]*(w*h) # top fibre directional nodes



surfacefaces=listSum(selectTetFace(t,bottomnodes+topnodes) for t in tinds) # list of surface face indices
surfacegraph=TriMeshGraph(nodes,surfacefaces)

#ds=PyDataSet('boxDS',nodes,[('inds',ElemType._Tet1NL,tinds)],[('boundaryfield',boundaryfield,'inds')])
#obj=MeshSceneObject('box',ds)    
#mgr.addSceneObject(obj)

writeMeshFile('test.mesh',nodes,tinds,boundaryfield,None,3)

with open(problemFile) as p:
    with open('prob.py','w') as o:
        o.write(p.read()%{'inputfile':'test.mesh','outdir':'.'})
        
p=ProblemConf.from_file('prob.py')
solve_pde(p)

obj=VTK.loadObject('test.vtk')
mgr.addSceneObject(obj)

ds=obj.datasets[0]
dirs=calculateGradientDirs(ds.getNodes(),generateSimplexEdgeMap(ds.getNodes().n(),tinds),ds.getDataField('t'))
ds.setDataField(dirs)

rep=obj.createRepr(ReprType._line,0)
mgr.addSceneObjectRepr(rep)

#rep=obj.createRepr(ReprType._volume,0)
#mgr.addSceneObjectRepr(rep)
#
#rep.applyMaterial('Rainbow',field='t')

rep=obj.createRepr(ReprType._glyph,0,externalOnly=False,drawInternal=True,glyphname='arrow',
                   dfield='dirs',vecfunc=VecFunc._Linear,glyphscale=(0.01,0.01,0.03))

mgr.addSceneObjectRepr(rep)
rep.applyMaterial('Rainbow',field='t')

mgr.setAxesType(AxesType._originarrows) # top right corner axes
mgr.controller.setRotation(0.8,0.8)
mgr.setCameraSeeAll()

