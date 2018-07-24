from eidolon import *
from AtrialFibrePlugin import writeMeshFile,problemFile, calculateGradientDirs, generateSimplexEdgeMap

from sfepy.base.conf import ProblemConf
from sfepy.applications import solve_pde

w,h,d=16,16,32
nodes,inds=generateHexBox(w-2,h-2,d-2)

# twist cube around origin in XZ plane
nodes=[vec3(0,(1-n.z())*halfpi,n.x()+1).fromPolar()+vec3(0,n.y(),0) for n in nodes]

tinds=[]

# convert hex to tets, warning: index() relies on float precision
hx=ElemType.Hex1NL.xis
tx=divideHextoTet(1)
for hexind in inds:
   for tet in tx:
       tinds.append([hexind[hx.index(t)] for t in tet])
       
field=[0]*len(nodes)
field[:w*h]=[1]*(w*h) # bottom boundary condition nodes
field[-w*h:]=[2]*(w*h) # top boundary condition nodes
#field[-int(w*h*4.6)]=2 # test single embedded boundary node

#ds=PyDataSet('boxDS',nodes,[('inds',ElemType._Tet1NL,tinds)],[('field',field,'inds')])
#obj=MeshSceneObject('box',ds)    
#mgr.addSceneObject(obj)

writeMeshFile('test.mesh',nodes,tinds,field,None,3)

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

