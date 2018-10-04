
import os
import eidolon
from eidolon import vec3, RealMatrix, StdProps, avg, halfpi,  ElemType, ReprType
import AtrialFibrePlugin


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
 

w,h,d=4,4,16
nodes,inds=eidolon.generateHexBox(w-2,h-2,d-2)

# twist cube around origin in XZ plane
nodes=[vec3(0,(1-n.z())*halfpi,n.x()+1).fromPolar()+vec3(0,n.y(),0) for n in nodes]

bottomnodes=list(range(0,w*h)) # indices of nodes at the bottom of the cube
topnodes=list(range(len(nodes)-w*h,len(nodes))) # indices of nodes at the top of the cube

tinds=[]

# convert hex to tets, warning: index() relies on float precision
hx=ElemType.Hex1NL.xis
tx=eidolon.divideHextoTet(1)
for hexind in inds:
   for tet in tx:
       tinds.append([hexind[hx.index(t)] for t in tet])
       
# boundary conditions field for laplace solve
boundaryField=[0]*len(nodes)
boundaryField[:w*h]=[1]*(w*h) # bottom boundary condition nodes
boundaryField[-w*h:]=[2]*(w*h) # top boundary condition nodes

# initial per-node fibre field with directions defined on the top and bottom surfaces
fibreField=[(0,0,0)]*len(nodes)
fibreField[:w*h]=[(1,0,0)]*(w*h) # bottom fibre directional nodes
fibreField[-w*h:]=[(0,1,1)]*(w*h) # top fibre directional nodes

# field defining gradient from top to bottom surface
gradfield=AtrialFibrePlugin.calculateMeshGradient(os.path.join(scriptdir,'tetmesh'),nodes,tinds,boundaryField,VTK)
gradfield.meta(StdProps._spatial,'inds')
gradfield.meta(StdProps._topology,'inds')

# calculate a directional field storing the gradient direction in mesh coordinates for each node
edgemap=AtrialFibrePlugin.generateSimplexEdgeMap(len(nodes),tinds)
directionalField=AtrialFibrePlugin.calculateGradientDirs(nodes,edgemap,gradfield)
    
fields=[
        ('boundaryField',boundaryField,'inds'),
        ('fibreField',fibreField,'inds'),
        ('directionalField',list(map(tuple,directionalField)),'inds'),
]
ds=eidolon.PyDataSet('boxDS',nodes,[('inds',ElemType._Tet1NL,tinds)],fields)


# extract the element adjacency matrix
eidolon.calculateElemExtAdj(ds)
adj=ds.getIndexSet('inds'+eidolon.MatrixType.adj[1])

# calculate the per-element directional adjacency matrix, ie. for each element state which neighbour is in the direction the field points
elemdiradj=AtrialFibrePlugin.getElemDirectionAdj(ds.getNodes(),ds.getIndexSet('inds'),adj,directionalField)
# follow the gradient field lines and for each element state which (element #, face #) is at each end of the line
elemFollow=AtrialFibrePlugin.followElemDirAdj(elemdiradj)

# estimate the thickness of measuring how far each element is from each surface by following gradient field
elemThickness=AtrialFibrePlugin.estimateThickness(ds.getNodes(),ds.getIndexSet('inds'),elemdiradj)

# convert the per-node initial fibre field to a per-element one
elemdirfield=convertNodeToElemField(ds.getDataField('fibreField'),ds.getIndexSet('inds'))
elemdirfield.setName('fibres')

interpFunc=lambda dir1,dir2,grad:tuple(dir1*grad+dir2*(1-grad))

# for each element, interpolate between the surface fibre directions from the elements at either end of the gradient field
for e in range(len(tinds)):
    elem1,face1,elem2,face2=elemFollow[e]
    
    dir1=elemdirfield[elem1]
    dir2=elemdirfield[elem2]
    grad=avg(gradfield[i] for i in tinds[e])
    
    elemdirfield[e]=interpFunc(vec3(*dir1),vec3(*dir2),grad)

ds.getDataField(gradfield)
ds.setDataField(elemThickness)
ds.setDataField(elemdirfield)

obj=eidolon.MeshSceneObject('box',ds)    
mgr.addSceneObject(obj)   

    
rep=obj.createRepr(ReprType._glyph,0,drawInternal=True,externalOnly=False,perelem=True, 
                   glyphname= 'arrow', dfield='fibres',glyphscale= (0.015, 0.015, 0.06))
mgr.addSceneObjectRepr(rep)

rep.applyMaterial('Rainbow',field=elemThickness.getName())

rep01=obj.createRepr(ReprType._line,0,drawInternal=True,externalOnly=False)
mgr.addSceneObjectRepr(rep01)

mgr.setAxesType(eidolon.AxesType._originarrows) # top right corner axes
mgr.controller.setRotation(0.8,0.8)
mgr.setCameraSeeAll()

