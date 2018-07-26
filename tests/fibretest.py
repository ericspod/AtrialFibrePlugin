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
            

w,h,d=8,8,16
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
ds.setDataField(dirs)


def findSurfaceTris(nodes,inds,vecLength,surfaceGraph,nodeDirections,gradField,outField):
    numnodes=len(nodes)
    oc=Octree.fromMesh(2,nodes,inds)
    
    def getDirection(pos):
        leaf=oc.getLeaf(pos)
        
        if leaf is None:
            return None
        
        for i in leaf.leafdata:
            elem=nodes.mapIndexRow(inds,i)
            xi=pointSearchLinTet(pos,*elem)
            if xi.isInUnitCube() and sum(xi)<=1.0:
                elemdirs=[nodeDirections[d] for d in inds[i]]
                return ElemType.Tet1NL.applyBasis(elemdirs,*xi)
            
        return None
    
    def followFieldLine(start,startdir,veclen):
        end=start+vec3(*startdir)*veclen
        tri=surfaceGraph.getIntersectedTri(start,end)
            
        while tri is None:
            nodedir=getDirection(end)
            if nodedir is None:
                break
            
            start=end
            end=start+vec3(*nodedir)*veclen
            tri=surfaceGraph.getIntersectedTri(start,end)
            
        return tri
        
    def getTriSurfaceDir(tri,u,v):
        v0,v1,v2=[outField[t] for t in surfaceGraph.tris[tri]]
        return vec3(*v0)*(1-u-v)+vec3(*v1)*u+vec3(*v2)*v
    
    for node in range(numnodes):
        outdir=outField[node]
        if vec3(*outdir).isZero():
            ptri=followFieldLine(nodes[node],nodeDirections[node],vecLength)
            ntri=followFieldLine(nodes[node],nodeDirections[node],-vecLength)
            
            if ptri is not None and ntri is not None:
                #outField[node]=(ptri,ntri,0)
                d1=getTriSurfaceDir(ptri[0],ptri[1][1],ptri[1][2])
                d2=getTriSurfaceDir(ntri[0],ntri[1][1],ntri[1][2])
                g=gradField[node]
                
                outField[node]=d1*g+d2*(1-g)
                
    

findSurfaceTris(ds.getNodes(),ds.getIndexSet('inds'),0.1,surfacegraph,dirs,grad,ds.getDataField('directionalField'))

rep=obj.createRepr(ReprType._line,0)
mgr.addSceneObjectRepr(rep)

rep=obj.createRepr(ReprType._glyph,0,externalOnly=False,drawInternal=True,glyphname='arrow',
                   dfield='directionalField',vecfunc=VecFunc._Linear,glyphscale=(0.01,0.01,0.03))

mgr.addSceneObjectRepr(rep)
rep.applyMaterial('Rainbow',field='t')

mgr.setAxesType(AxesType._originarrows) # top right corner axes
mgr.controller.setRotation(0.8,0.8)
mgr.setCameraSeeAll()

