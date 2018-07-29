from eidolon import *
from AtrialFibrePlugin import writeMeshFile,problemFile, calculateGradientDirs, generateSimplexEdgeMap, TriMeshGraph, generateNodeElemMap

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


def getElemDirectionAdj(nodes,elems,dirField,adj=None):
    if not adj:
        ds=PyDataSet('tmp',nodes,[elems])
        calculateElemExtAdj(ds)
        adj=ds.getIndexSet(elems.getName()+MatrixType.adj[1])
        
    et=ElemType[elems.getType()]
    result=IndexMatrix('diradj',elems.n(),2)
    result.meta(StdProps._elemdata,'True')
    result.fill(elems.n())
    
    for e,elem in enumerate(elems):
        edir=vec3(*dirField[e])
        enodes=[nodes[n] for n in elem] # elem nodes
        center=et.applyBasis(enodes,0.33333,0.33333,0.33333) # elem center
#        dist=max(center.distTo(n) for n in enodes)
#        edir=edir.norm()*dist # elem direction, definitely intersects face
        
        dray=Ray(center,edir)
        
        for f,face in enumerate(et.faces):
            fnodes=[enodes[i] for i in face[:3]]
            if dray.intersectsTri(*fnodes):
                result[e,0]=f
                result[e,1]=adj[e,f]
                break
            
        assert result[e,0]<elems.n()
        
#        norms=[enodes[f[0]].planeNorm(enodes[f[1]],enodes[f[2]]) for f in et.faces]
#        minangleface=min(range(len(norms)),key=lambda i:norms[i].angleTo(edir))
#        result[e,0]=minangleface
#        result[e,1]=adj[e,minangleface]
        
    return result
        

def followElemDirAdj(nodes,elems,elemdiradj):
#    et=ElemType[elems.getType()]
    result=IndexMatrix('diradj',elems.n(),2)
    result.meta(StdProps._elemdata,'True')
    
    for e in range(elems.n()):
        curelem=e
        
        while elemdiradj[curelem,1]<elems.n():
            curelem=elemdiradj[curelem,1]
            
        result[e]=(curelem,elemdiradj[curelem,0])
        
    return result


def interpolateElemDirections(elems,elemFollow,gradField,dirField,directionalField):
    et=ElemType[elems.getType()]
    elemdirField=RealMatrix('elemdirField',elems.n(),3)
    elemdirField.meta(StdProps._elemdata,'True')
    elemdirField.fill(0)
    
    for e,elem in enumerate(elems):
        endelem,endface=elemFollow[e]
        #faceinds=et.faces[endface]
        
        #assert all(elem[f] in topnodes for f in faceinds)
        
#        enddir=avg(vec3(*directionalField[elem[f]]) for f in faceinds).norm()
        enddir=dirField[endelem]
        elemdirField[e]=[endface]*3 #tuple(enddir)
        
    return elemdirField
        

elemdiradj=getElemDirectionAdj(ds.getNodes(),ds.getIndexSet('inds'),dirs)
elemFollow=elemdiradj#followElemDirAdj(ds.getNodes(),ds.getIndexSet('inds'),elemdiradj)

elemDirField=interpolateElemDirections(ds.getIndexSet('inds'),elemFollow,grad,dirs,directionalField)
#elemDirField=convertElemToNodeField(elemDirField,ds.getNodes().n(),ds.getIndexSet('inds'))
elemDirField.meta(StdProps._spatial,'inds')
elemDirField.meta(StdProps._topology,'inds')
ds.setDataField(elemDirField)

#def findSurfaceTris(nodes,inds,vecLength,surfaceGraph,fieldLines):
#    '''
#    Given a mesh defined by nodes `nodes' and tet indices `inds', with `surfaceGraph' representing the surfaces of the
#    mesh with defined directions (ie. inner and outer surfaces), return a map relating nodes to indices stating which
#    triangles the node's field lines pass through and where. The field lines are defined by the `fieldLines' field which
#    stores a vector for each node of `node's indicating the field direction at that point. 
#    '''
#    numnodes=len(nodes)
#    oc=Octree.fromMesh(2,nodes,inds)
#    
#    def getDirection(pos):
#        '''Get the field line direction at the position `pos' somewhere in the mesh. Returns None if not in the mesh.'''
#        leaf=oc.getLeaf(pos)
#        
#        if leaf is None:
#            return None
#        
#        for i in leaf.leafdata:
#            elem=nodes.mapIndexRow(inds,i)
#            xi=pointSearchLinTet(pos,*elem)
#            
#            # pos is in element i
#            if xi.isInUnitCube() and sum(xi)<=1.0:
#                elemdirs=[fieldLines[d] for d in inds[i]]
#                return ElemType.Tet1NL.applyBasis(elemdirs,*xi)
#            
#        return None
#    
#    def followFieldLine(start,startdir,veclen):
#        '''
#        Follow the field lines starting at `start' with direction `startdir', advancing by a vector of length `veclen'.
#        Returns a (triangle,(t,u,v)) pair once a surface triangle has been intersected, None otherwise.
#        '''
#        pos=start
#        prev=start
#        nodedir=getDirection(pos)
#        
#        while nodedir is not None:
#            prev,pos=pos,pos+vec3(*nodedir)*veclen
#            nodedir=getDirection(pos)
#            
#        if pos!=prev:
#            return surfaceGraph.getIntersectedTri(pos,prev)
#        
#        return None
#                
##        end=start+vec3(*startdir)*veclen
##        tri=surfaceGraph.getIntersectedTri(start,end)
##            
##        while tri is None:
##            nodedir=getDirection(end)
##            if nodedir is None:
##                break
##            
##            start=end
##            end=start+vec3(*nodedir)*veclen
##            tri=surfaceGraph.getIntersectedTri(start,end)
##            
##        return tri
#        
#    nodemap={}
#    
#    for node in range(numnodes):
#        printFlush(node)
#        ptri=followFieldLine(nodes[node],fieldLines[node],vecLength)
#        ntri=followFieldLine(nodes[node],fieldLines[node],-vecLength)
#        
#        if ptri is not None and ntri is not None:
#            nodemap[node]=(ptri,ntri)
#                
#    return nodemap
#    
#
#def interpolateNodeDirections(nodemap,surfaceGraph,gradField,directionField):
#    def getTriSurfaceDir(tri,u,v):
#        v0,v1,v2=[directionField[t] for t in surfaceGraph.tris[tri]]
#        return vec3(*v0)*(1-u-v)+vec3(*v1)*u+vec3(*v2)*v
#    
#    for node,(ptri,ntri) in nodemap.items():
#        if vec3(*directionalField[node]).isZero(): # skip nodes with assigned directions (ie. on the surfaces)
#            d1=getTriSurfaceDir(ptri[0],ptri[1][1],ptri[1][2])
#            d2=getTriSurfaceDir(ntri[0],ntri[1][1],ntri[1][2])
#            g=gradField[node]
#            
#            directionField[node]=d1*g+d2*(1-g)
#
#
#nodemap=findSurfaceTris(ds.getNodes(),ds.getIndexSet('inds'),1.0,surfacegraph,dirs)
#
#interpolateNodeDirections(nodemap,surfacegraph,grad,ds.getDataField('directionalField'))


rep=obj.createRepr(ReprType._line,0)
mgr.addSceneObjectRepr(rep)

mid=ElemType.Tet1NL.basis(0.25,0.25,0.25)

elemnodes=[ElemType.Tet1NL.applyCoeffs([nodes[e] for e in elem],mid) for elem in tinds]
elemobj=MeshSceneObject('elemobj',PyDataSet('elemobjds',elemnodes,[],[elemDirField,dirs,convertNodeToElemField(grad,ds.getIndexSet('inds'))]))
mgr.addSceneObject(elemobj)

rep=elemobj.createRepr(ReprType._glyph,0,externalOnly=False,drawInternal=True,glyphname='arrow',
                   dfield=elemDirField.getName(),vecfunc=VecFunc._Linear,glyphscale=(0.01,0.01,0.03),matname='Rainbow',field=grad.getName())

mgr.addSceneObjectRepr(rep)
#rep.applyMaterial('Rainbow',field=grad.getName())

rep=obj.createRepr(ReprType._volume,matname='Rainbow',field=elemDirField.getName())
mgr.addSceneObjectRepr(rep)

mgr.setAxesType(AxesType._originarrows) # top right corner axes
mgr.controller.setRotation(0.8,0.8)
mgr.setCameraSeeAll()

