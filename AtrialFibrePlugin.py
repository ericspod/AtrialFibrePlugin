# Eidolon Biomedical Framework
# Copyright (C) 2016-8 Eric Kerfoot, King's College London, all rights reserved
# 
# This file is part of Eidolon.
#
# Eidolon is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# Eidolon is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License along
# with this program (LICENSE.txt).  If not, see <http://www.gnu.org/licenses/>
'''
Atrial fibre generation plugin.
'''

import os
import stat
import ast
import shutil
import datetime
import zipfile
import warnings
from itertools import starmap
from collections import defaultdict

try:
    import configparser
except ImportError:
    import ConfigParser as configparser
    
try:
    from sfepy.base.conf import ProblemConf
    from sfepy.applications import solve_pde
    from sfepy.base.base import output
except ImportError:
    warnings.warn('SfePy needs to be installed or in PYTHONPATH to generate fiber directions.')

from eidolon import ScenePlugin, Project, avg, vec3, successive, first, RealMatrix, IndexMatrix, StdProps, timing
import eidolon, ui

import numpy as np

scriptdir= os.path.dirname(os.path.abspath(__file__)) # this file's directory

# directory/file names
uifile=os.path.join(scriptdir,'AtrialFibrePlugin.ui') 
deformDir=os.path.join(scriptdir,'deformetricaC')
deformExe=os.path.join(deformDir,'deformetrica')
architecture=os.path.join(scriptdir,'architecture.ini') 
problemFile=os.path.join(scriptdir,'problemfile.py') 

# registration file names
decimatedFile='subject.vtk'
targetFile='target.vtk'
datasetFile='data_set.xml'
modelFile='model.xml'
optimFile='optimization_parameters.xml'
registeredFile='Registration_subject_to_subject_0__t_9.vtk'
decimate='decimate-surface'

# deformetrica parameters
kernelWidthSub=5000
kernelWidthDef=8000
kernelType='cudaexact'
dataSigma=0.1
stepSize=0.000001

# field names
regionField='regions'
landmarkField='landmarks'
directionField='directions'

objNames=eidolon.enum(
    'atlasmesh',
    'epimesh','epinodes',
    'endomesh','endonodes',
    'architecture'
)

regTypes=eidolon.enum('endo','epi')


# load the UI file into the ui namespace, this is subtyped below
ui.loadUI(open(uifile).read())


def showLines(nodes,lines,name='Lines',matname='Default'):
    mgr=eidolon.getSceneMgr()
    lineds=eidolon.LineDataSet(name+'DS',nodes,lines)
    obj=eidolon.MeshSceneObject(name,lineds)
    mgr.addSceneObject(obj)
    
    rep=obj.createRepr(eidolon.ReprType._line,matname=matname)
    mgr.addSceneObjectRepr(rep)
    
    return obj,rep


class initdict(defaultdict):
    def __init__(self,initfunc,*args,**kwargs):
        defaultdict.__init__(self,None,*args,**kwargs)
        self.initfunc=initfunc
        
    def __missing__(self,key):
        value=self.initfunc(key)
        self.__setitem__(key,value)
        return value
    

class plane(object):
    def __init__(self,center,norm):
        self.center=center
        self.norm=norm.norm()
        
    def dist(self,pt):
        return pt.planeDist(self.center,self.norm)
        
    def moveUp(self,dist):
        self.center+=self.norm*dist
        
    def numPointsAbove(self,nodes):
        return sum(1 for n in nodes if self.dist(n)>=0)
        
    def between(self,nodes,otherplane):
        numnodes=len(nodes)
        return self.numPointsAbove(nodes)==numnodes and otherplane.numPointsAbove(nodes)==numnodes
       
    def findIntersects(self,nodes,inds):
        numnodes=inds.m()
        result=[]        
        for n in range(inds.n()):
            if 0<self.numPointsAbove(nodes.mapIndexRow(inds,n))<numnodes:
                result.append(n)
                
        return result
        
        
class TriMeshGraph(object):
    def __init__(self,nodes,tris,ocdepth=3):
        self.nodes=nodes if isinstance(nodes,eidolon.Vec3Matrix) else eidolon.listToMatrix(nodes,'nodes')
        self.tris=tris if isinstance(tris,eidolon.IndexMatrix) else eidolon.listToMatrix(tris,'tris')
        
        self.tricenters=[avg(self.getTriNodes(r),vec3()) for r in range(self.tris.n())]
        self.adj=generateTriAdj(self.tris) # elem -> elems
        self.nodeelem=generateNodeElemMap(self.nodes.n(),self.tris) # node -> elems
        self.edges=generateSimplexEdgeMap(self.nodes.n(),self.tris) # node -> nodes
        self.boundbox=eidolon.BoundBox(nodes)
        self.octree=eidolon.Octree(ocdepth,self.boundbox.getDimensions(),self.boundbox.center)
        
        self.octree.addMesh(self.nodes,self.tris)
        
        def computeDist(key):
            i,j=key
            return self.tricenters[i].distTo(self.tricenters[j])
            
        self.tridists=initdict(computeDist)
        
    def getIntersectedTri(self,start,end):
        startoc=self.octree.getLeaf(start)
        endoc=self.octree.getLeaf(end)
        inds=(startoc.leafdata if startoc is not None else []) + (endoc.leafdata if endoc is not None else []) 
        
        r=eidolon.Ray(start,end-start)
        
        for tri in inds:
            d=r.intersectsTri(*self.getTriNodes(tri))
            if d:
                return tri
            
        return None
        
    def getPathSubgraph(self,starttri,endtri):
        return getAdjTo(self.adj,starttri,endtri)
        
    def getTriNodes(self,triindex):
        return self.nodes.mapIndexRow(self.tris,triindex)
    
    def getTriNorm(self,triindex):
        a,b,c=self.getTriNodes(triindex)
        return a.planeNorm(b,c)
    
    def getSharedNodeTris(self,triindex):
        tris=set()
        for n in self.tris[triindex]:
            tris.update(self.nodeelem[n])
            
        tris.remove(triindex)
        return list(sorted(tris))
        
    def getPath(self,starttri,endtri,acceptTri=None):
        return dijkstra(self.adj,starttri,endtri,lambda i,j:self.tridists[(i,j)],acceptTri)
    
    
def loadArchitecture(path,section):
    '''
    Load the architecture from the given file `path' and return values from the given section (endo or epi). The
    return value is a tuple containing:
        landmarks: 0-based indices of landmark nodes in the atlas
        lmlines  : 0-based index pairs defining lines between indices in landmarks
        lmregions: a list of maps each defining a retion, which are mappings from a 0-based index into lmlines to the 
                   line's landmark index pair
        lmstim   : a per-region specifier list stating which lines (L# for index #) or atlas node (N#) defines stimulation
        lground  : a per-region specifier list stating which lines (L# for index #) or atlas node (N#) defines ground
    '''
    c=configparser.SafeConfigParser()
    assert len(c.read(path))>0
    
    landmarks=ast.literal_eval(c.get(section,'landmarks')) # 0-based node indices
    lines=ast.literal_eval(c.get(section,'lines')) # 1-based landmark indices
    regions=ast.literal_eval(c.get(section,'regions')) # 1-based landmark indices
    stimulus=ast.literal_eval(c.get(section,'stimulus')) # per region
    ground=ast.literal_eval(c.get(section,'ground')) # per region
#    types=ast.literal_eval(c.get('endo','type')) # per region    
    
    lmlines=[subone(l) for l in lines if max(l)<=len(landmarks)] # filter for lines with existing node indices
    lmregions=[subone(r) for r in regions]
#    lmregions=[subone(r) for r in regions if all(i<=len(landmarks) for i in r)]
    
    lmstim=stimulus[:len(lmregions)]
    lmground=ground[:len(lmregions)]
    
    allregions=[]
    for r in lmregions:
        lr={i:(a,b) for i,(a,b) in enumerate(lmlines) if a in r and b in r}
        if len(lr)>2:
            allregions.append(lr)

    return landmarks,lmlines,allregions,lmstim,lmground
    

def writeMeshFile(filename,nodes,inds,nodegroup,indgroup,dim):
    with open(filename,'w') as o:
        print('MeshVersionFormatted 1',file=o)
        print('Dimension %i'%dim,file=o)
        print('Vertices',file=o)
        print(len(nodes),file=o)
        
        for n in range(len(nodes)):
            for v in tuple(nodes[n])[:dim]:
                print('%20.10f'%v,end=' ',file=o)
                
            group=0 if nodegroup is None else nodegroup[n]
            
            print(group,file=o)
            
        print('Triangles' if len(inds[0])==3 else 'Tetrahedra',file=o)
        print(len(inds),file=o)
        
        for n in range(len(inds)):
            print(*['%10i'%(t+1) for t in inds[n]],file=o,end=' ')
            group=0 if indgroup is None else indgroup[n]
            
            print(group,file=o)


def registerSubjectToTarget(subjectObj,targetObj,outdir,decimpath,VTK):
    '''
    Register the `subjectObj' mesh object to `targetObj' mesh object putting data into directory `outdir'. The subject 
    will be decimated to have roughly the same number of nodes as the target and then stored as subject.vtk in `outdir'. 
    Registration is done with Deformetrica and result stored as 'Registration_subject_to_subject_0__t_9.vtk' in `outdir'.
    '''
    dpath=os.path.join(outdir,decimatedFile)
    tmpfile=os.path.join(outdir,'tmp.vtk')
    
    shutil.copy(os.path.join(deformDir,datasetFile),os.path.join(outdir,datasetFile))
 
    model=open(os.path.join(deformDir,modelFile)).read()
    model=model.replace('%1',str(dataSigma))
    model=model.replace('%2',str(kernelWidthSub))
    model=model.replace('%3',str(kernelType))
    model=model.replace('%4',str(kernelWidthDef))

    with open(os.path.join(outdir,modelFile),'w') as o:
        o.write(model)
        
    optim=open(os.path.join(deformDir,optimFile)).read()
    optim=optim.replace('%1',str(stepSize))
    
    with open(os.path.join(outdir,optimFile),'w') as o:
        o.write(optim)

    VTK.saveLegacyFile(tmpfile,subjectObj,datasettype='POLYDATA')
    VTK.saveLegacyFile(os.path.join(outdir,targetFile),targetObj,datasettype='POLYDATA')
        
    snodes=subjectObj.datasets[0].getNodes()
    tnodes=targetObj.datasets[0].getNodes()
    
    sizeratio=float(tnodes.n())/snodes.n()
    sizepercent=str(100*(1-sizeratio))[:6] # percent to decimate by
    
    # decimate the mesh most of the way towards having the same number of nodes as the target
    ret,output=eidolon.execBatchProgram(decimpath,tmpfile,dpath,'-reduceby',sizepercent,'-ascii',logcmd=True)
    assert ret==0,output
    
    ret,output=eidolon.execBatchProgram(deformExe,"registration", "3D", modelFile, datasetFile, optimFile, "--output-dir=.",cwd=outdir,logcmd=True)
    assert ret==0,output
    
    return output


def transferLandmarks(archFilename,fieldname,sourceObj,subjectObj,outdir,VTK):
    '''
    Register the landmarks defined as node indices on `sourceObj' to equivalent node indices on `subjectObj' via the
    decimated and registered intermediary stored in `outdir'. The result is a list of index pairs associating a node
    index in `subjectObj' for every landmark index in `sourceObj'.
    '''
    decimated=os.path.join(outdir,decimatedFile)
    registered=os.path.join(outdir,registeredFile)
    
    lmarks=loadArchitecture(archFilename,fieldname)[0]
    
    reg=VTK.loadObject(registered) # mesh registered to target
    dec=VTK.loadObject(decimated) # decimated unregistered mesh
    
    tnodes=sourceObj.datasets[0].getNodes() # target points
    rnodes=reg.datasets[0].getNodes() # registered decimated points
    dnodes=dec.datasets[0].getNodes() # unregistered decimated points
    snodes=subjectObj.datasets[0].getNodes() # original subject points
    
    lmpoints=[(tnodes[m],m) for m in lmarks]
    
    # TODO: use scipy.spatial.cKDTree?
    def getNearestPointIndex(pt,nodes):
        '''Find the index in `nodes' whose vector is closest to `pt'.'''
        return min(range(len(nodes)),key=lambda i:pt.distToSq(nodes[i]))
    
    rpoints=[(getNearestPointIndex(pt,rnodes),m) for pt,m in lmpoints]
    
    spoints=[(getNearestPointIndex(dnodes[i],snodes),m) for i,m in rpoints]
        
    assert len(spoints)==len(lmpoints)
    assert all(p[0] is not None for p in spoints)
    
    
    return spoints


def generateTriAdj(tris):
    '''
    Generates a table (n,3) giving the indices of adjacent triangles for each triangle, with n indicating a free edge.
    The indices in each row are in order rather than per triangle edge. The result is the dual of the triangle mesh.
    '''
    edgemap = {} # maps edges to the first triangle having that edge
    result=IndexMatrix(tris.getName()+'Adj',tris.n(),3)
    result.fill(tris.n())
    
    # Find adjacent triangles by constructing a map from edges defined by points (a,b) to the triangle having that edge,
    # when that edge is encountered twice then the current triangle is adjacent to the one that originally added the edge.
    for t1,tri in enumerate(tris): # iterate over each triangle t1
        for a,b in successive(tri,2,True): # iterate over each edge (a,b) of t1
            k=(min(a,b),max(a,b)) # key has uniform edge order
            t2=edgemap.pop(k,None) # attempt to find edge k in the map, None indicates edge not found
            
            if t2 is not None: # an edge is shared if already encountered, thus t1 is adjacent to t2
                result[t1]=sorted(set(result[t1]+(t2,)))
                result[t2]=sorted(set(result[t2]+(t1,)))
            else:
                edgemap[k]=t1 # first time edge is encountered, associate this triangle with it

    return result


@timing
def getAdjTo(adj,start,end):
    '''Returns a subgraph of `adj',represented as a node->[neighbours] dict, which includes nodes `start' and `end'.'''
    visiting=set([start])
    found={}
    numnodes=adj.n()
    
    while end not in found:
        visit=visiting.pop()
        neighbours=[n for n in adj.getRow(visit) if n<numnodes]
        found[visit]=neighbours
        visiting.update(n for n in neighbours if n not in found)
                
    return found
        
   
def generateNodeElemMap(numnodes,tris):
    '''Returns a list relating each node index to the set of element indices using that node.'''
    nodemap=[set() for _ in range(numnodes)]

    for i,tri in enumerate(tris):
        for n in tri:
            nodemap[n].add(i)
            
    return nodemap
    

def generateSimplexEdgeMap(numnodes,simplices):
    '''
    Returns a list relating each node index to the set of node indices joined to it by graph edges. This assumes the mesh
    has `numnodes' number of nodes and simplex topology `simplices'.
    '''
    nodemap=[set() for _ in range(numnodes)]

    for simplex in simplices:
        simplex=set(simplex)
        for s in simplex:
            nodemap[s].update(simplex.difference((s,)))
            
    return nodemap
    

@timing
def dijkstra(adj, start, end,distFunc,acceptTri=None):
    #http://benalexkeen.com/implementing-djikstras-shortest-path-algorithm-with-python/
    # shortest paths is a dict of nodes to previous node and distance
    paths = {start: (None,0)}
    curnode = start
    visited = set()
    # consider only subgraph containing start and end, this expands geometrically so should contain the minimal path
    adj=getAdjTo(adj,start,end) 
    
    if acceptTri is not None:
        accept=lambda a: (a in adj and acceptTri(a))
    else:
        accept=lambda a: a in adj
    
    while curnode != end:
        visited.add(curnode)
        destinations = list(filter(accept,adj[curnode]))
        curweight = paths[curnode][1]

        for dest in destinations:
            weight = curweight+distFunc(curnode,dest)
            
            if dest not in paths or weight < paths[dest][1]:
                paths[dest] = (curnode, weight)
        
        nextnodes = {node: paths[node] for node in paths if node not in visited}
        
        if not nextnodes:
            raise ValueError("Route %i -> %i not possible"%(start,end))
            
        # next node is the destination with the lowest weight
        curnode = min(nextnodes, key=lambda k:nextnodes[k][1])
    
    # collect path from end node back to the start
    path = []
    while curnode is not None:
        path.insert(0,curnode)
        curnode = paths[curnode][0]
        
    return path
    
    
def subone(v):
    return tuple(i-1 for i in v)
    

def findNearestIndex(node,nodelist):
    return min(range(len(nodelist)),key=lambda i:node.distToSq(nodelist[i]))
        

def getContiguousTris(graph,starttri,acceptTri):
    accepted=[starttri]
    
    adjacent=first(i for i in graph.getSharedNodeTris(starttri) if i not in accepted and acceptTri(i))
    
    while adjacent is not None:
        accepted.append(adjacent)
        
        for a in accepted[::-1]:
            allneighbours=graph.getSharedNodeTris(a)
            adjacent=first(i for i in allneighbours if i not in accepted and acceptTri(i))
            if adjacent:
                break
        
    return accepted


@timing
def findTrisBetweenNodes(start,end,landmarks,graph):
    start=landmarks[start]
    end=landmarks[end]

    starttri=first(graph.nodeelem[start])
    endtri=first(graph.nodeelem[end])
    
    nodes=graph.nodes
    startnode=nodes[start]
    endnode=nodes[end]
    
    easypath= graph.getPath(starttri,endtri)
    
    midnode=graph.tricenters[easypath[len(easypath)//2]]
    
    # define planes to bound the areas to search for triangles to within the space of the line
    splane=plane(startnode,midnode-startnode)
    eplane=plane(endnode,midnode-endnode)


    # adjust the plane's positions to account for numeric error
    adjustdist=1e1
    splane.moveUp(-adjustdist)
    eplane.moveUp(-adjustdist)
    
    
    assert starttri is not None
    assert endtri is not None
    
    # TODO: plane normal determination still needs work
    #linenorm=midnode.planeNorm(startnode,endnode)
    #linenorm=graph.getTriNorm(easypath[len(easypath)//2]).cross(midnode-startnode)
    linenorm=eidolon.avg(graph.getTriNorm(e) for e in easypath).cross(midnode-startnode)
    
    lineplane=plane(splane.center,linenorm)
    
    indices=set([starttri,endtri]) # list of element indices on lineplane between splane and eplane
    
    for i in range(graph.tris.n()):
        trinodes=graph.getTriNodes(i)
        
        numabove=lineplane.numPointsAbove(trinodes)
        if numabove in (1,2) and splane.between(trinodes,eplane):    
            indices.add(i)
    
    accepted=getContiguousTris(graph,starttri,lambda i:i in indices)
    
    if endtri not in accepted:
        eidolon.printFlush('---Resorting to easypath')
        accepted=easypath 
        
    return accepted
    
    
@timing
def assignRegion(region,index,assignmat,landmarks,linemap,graph):
    
    def getEnclosedGraph(adj,excludes,start):
        visiting=set([start])
        found=set()
        numnodes=adj.n()
        
        assert start is not None
        
        while visiting:
            visit=visiting.pop()
            neighbours=[n for n in adj.getRow(visit) if n<numnodes and n not in excludes]
            found.add(visit)
            visiting.update(n for n in neighbours if n not in found)
                    
        return found
    
    # collect all tri indices on the border of this region
    bordertris=set()
    for lineindex,(a,b) in region.items():
        if (a,b) in linemap:
            line=linemap[(a,b)]
        else:
            line=findTrisBetweenNodes(a,b,landmarks,graph)
            linemap[(a,b)]=line
            linemap[(b,a)]=line
            
            # assign line ID to triangles on the line
            for tri in line:
                assignmat[tri,2]=lineindex
            
        bordertris.update(line)
        
    bordertri=graph.tricenters[first(bordertris)]
    
    farthest=max(range(len(graph.tris)),key=lambda i:graph.tricenters[i].distToSq(bordertri))
    
    maxgraph=getEnclosedGraph(graph.adj,bordertris,farthest)
    
    for tri in range(len(graph.tris)):
        if tri not in maxgraph or tri in bordertris:
            if assignmat[tri,0]<0:
                assignmat[tri,0]=index
            elif assignmat[tri,1]<0:
                assignmat[tri,1]=index
    
    
@timing
def generateRegionField(obj,landmarkObj,regions,task=None):
    ds=obj.datasets[0]
    nodes=ds.getNodes()
    tris=first(ind for ind in ds.enumIndexSets() if ind.m()==3 and bool(ind.meta(StdProps._isspatial)))
    lmnodes=landmarkObj.datasets[0].getNodes()
    linemap={}
    
    landmarks=[nodes.indexOf(lm)[0] for lm in lmnodes]
    
    graph=TriMeshGraph(nodes,tris)
    
    filledregions=RealMatrix(regionField,tris.n(),3)
    filledregions.meta(StdProps._elemdata,'True')
    filledregions.fill(-10)
    
    if task:
        task.setMaxProgress(len(regions))

    for rindex,region in enumerate(regions):
        eidolon.printFlush('Region',rindex,len(regions),region)
        assignRegion(region,rindex,filledregions,landmarks,linemap,graph)   
        if task:
            task.setProgress(rindex+1)
        
    return filledregions,linemap


def extractTriRegion(nodes,tris,acceptFunc):
    '''
    Extract the region from the mesh (nodes,tris) as defined by the triangle acceptance function `acceptFunc'. The return
    value is a tuple containing the list of new nodes, a list of new tris, a map from old node indices in `nodes' to new 
    indices in the returned node list, and a map from triangle indices in `tris' to new ones in the returned triangle list.
    '''
    #old -> new
    newnodes=[] # new node set
    newtris=[] # new triangle set
    nodemap={} # maps old node indices to new
    trimap={} # maps old triangle indices to new
    
    for tri in range(len(tris)):
        if acceptFunc(tri):
            newtri=list(tris[tri])
            
            for i,n in enumerate(newtri):
                if n not in nodemap:
                    nodemap[n]=len(newnodes)
                    newnodes.append(nodes[n])
                    
                newtri[i]=nodemap[n]
                
            trimap[tri]=len(newtris)
            newtris.append(newtri)
            
    return newnodes,newtris,nodemap,trimap


def calculateGradientDirs(nodes,edges,gradientField):
    '''
    Returns a RealMatrix object containing the vector field for each node of `graph' pointing in the gradient direction
    for the given field RealMatrix object `gradientField'.
    '''
    #https://math.stackexchange.com/questions/2627946/how-to-approximate-numerically-the-gradient-of-the-function-on-a-triangular-mesh/2632616#2632616
    
    numnodes=len(nodes)
    nodedirs=eidolon.RealMatrix('dirs',numnodes,3)
    
    for n in range(numnodes):
        ngrad=gradientField[n]
        nnode=nodes[n]
        edgegrads=[gradientField[i]-ngrad for i in edges[n]] # field gradient in edge directions
        edgedirs=[list(nodes[i]-nnode) for i in edges[n]] # edge directional vectors
        
        # node direction is solution for x in Ax=b where A is edge directions and b edge gradients
        nodedir=np.linalg.lstsq(np.asarray(edgedirs),np.asarray(edgegrads),rcond=None)
        nodedirs[n]=vec3(*nodedir[0]).norm()
    
    return nodedirs
    
    

@timing
def calculateDirectionField(obj,landmarkObj,regions,regtype,tempdir,VTK):
    _,lmlines,allregions,lmstim,lmground=loadArchitecture(architecture,regtype)
    
    regions=regions or list(range(len(allregions)))
    
    ds=obj.datasets[0]
    nodes=ds.getNodes()
    tris=first(ind for ind in ds.enumIndexSets() if ind.m()==3 and bool(ind.meta(StdProps._isspatial)))
    regionfield=ds.getDataField(regionField)
    regionfield=np.asarray(regionfield,np.int32)
    
    lmnodes=landmarkObj.datasets[0].getNodes()
    landmarks=[nodes.indexOf(lm)[0] for lm in lmnodes]
    
    directionfield=RealMatrix(directionField,nodes.n(),3)
    directionfield.fill(-10)
    
    gradientfield=RealMatrix('gradient',nodes.n(),1)
    gradientfield.fill(-1)
    
    obj.datasets[0].setDataField(gradientfield)
    
    def collectNodes(nodemap,trimap,components):
        nodeinds=set()
        for comp in components:
            if comp[0]=='L':
                lind=int(comp[1:])-1
                
                for tri in trimap:
                    if int(regionfield[tri,2])==lind:
                        nodeinds.update(nodemap[t] for t in tris[tri])
            else:
                nodeinds.add(nodemap[landmarks[int(comp[1:])-1]])
                
        return nodeinds
    
    for r in regions:
        eidolon.printFlush('Region',r,lmstim[r],lmground[r])
        
        try:
            rfile=os.path.join(tempdir,'region%.2i.mesh'%r)
            lfile=os.path.join(tempdir,'region%.2i.log'%r)
            ofile=os.path.join(tempdir,'region%.2i.vtk'%r)
            pfile=os.path.join(tempdir,'region%.2i.py'%r)
            newnodes,newtris,nodemap,trimap=extractTriRegion(nodes,tris,lambda i:r in regionfield[i,:2])
            
            assert len(newtris)>0, 'Empty region selected'
            
            stimnodes=collectNodes(nodemap,trimap,lmstim[r])
            groundnodes=collectNodes(nodemap,trimap,lmground[r])
            
            if len(stimnodes)==0:
                raise ValueError('Region %i has no stim nodes'%r)
            elif not all(0<=s<len(newnodes) for s in stimnodes):
                raise ValueError('Region %i has invalid stim nodes: %r'%(r,stimnodes))
            
            if len(groundnodes)==0:
                raise ValueError('Region %i has no ground nodes'%r)    
            elif not all(0<=s<len(newnodes) for s in groundnodes):
                raise ValueError('Region %i has invalid ground nodes: %r'%(r,groundnodes))
            
            # convert triangles to tets
            for t in range(len(newtris)):
                a,b,c=[newnodes[i] for i in newtris[t]]
                norm=a.planeNorm(b,c)
                newtris[t].append(len(newnodes))
                newnodes.append(avg((a,b,c))+norm)
            
            nodegroup=[1 if n in stimnodes else (2 if n in groundnodes else 0) for n in range(len(newnodes))]
            
            assert 1 in nodegroup, 'Region %i does not assign stim nodes (%r)'%(r,stimnodes)
            assert 2 in nodegroup, 'Region %i does not assign ground nodes (%r)'%(r,groundnodes)
            
            writeMeshFile(rfile,newnodes,newtris,nodegroup,None,3)
            
            with open(problemFile) as p:
                with open(pfile,'w') as o:
                    o.write(p.read()%{'inputfile':rfile,'outdir':tempdir})
                
            p=ProblemConf.from_file(pfile)
            output.set_output(lfile,True,True)
            solve_pde(p)
            
            robj=VTK.loadObject(ofile)
            gfield=robj.datasets[0].getDataField('t')
            
            for oldn,newn in nodemap.items():
                gradientfield[oldn,0]=gfield[newn]
                
        except Exception as e:
            eidolon.printFlush(e)
            
### Project objects

class AtrialFibrePropWidget(ui.QtWidgets.QWidget,ui.Ui_AtrialFibre):
    def __init__(self,parent=None):
        super(AtrialFibrePropWidget,self).__init__(parent)
        self.setupUi(self)
        

class AtrialFibreProject(Project):
    def __init__(self,name,parentdir,mgr):
        Project.__init__(self,name,parentdir,mgr)
        self.header='AtrialFibre.createProject(%r,scriptdir+"/..")\n' %(self.name)
        
        self.AtrialFibre=mgr.getPlugin('AtrialFibre')
        self.VTK=self.mgr.getPlugin('VTK')
        
        self.AtrialFibre.project=self # associate project with plugin
        
        self.backDir=self.logDir=self.getProjectFile('logs')
        
        self.addHandlers()
        
    def create(self):
        Project.create(self)
        if not os.path.isdir(self.logDir):
            os.mkdir(self.logDir)
        
    def getPropBox(self):
        prop=Project.getPropBox(self)

        # remove the UI for changing the project location
        eidolon.cppdel(prop.chooseLocLayout)
        eidolon.cppdel(prop.dirButton)
        eidolon.cppdel(prop.chooseLocLabel)

        self.afprop=AtrialFibrePropWidget()
        prop.verticalLayout.insertWidget(prop.verticalLayout.count()-1,self.afprop)

        def setConfigMap(combo,name):
            @combo.currentIndexChanged.connect
            def _set(i):
                self.configMap[name]=str(combo.itemText(i))
                
        setConfigMap(self.afprop.atlasBox,objNames._atlasmesh)
        setConfigMap(self.afprop.endoBox,objNames._endomesh)
        setConfigMap(self.afprop.epiBox,objNames._epimesh)
        
        self.afprop.endoReg.clicked.connect(lambda:self._registerLandmarks(objNames._endomesh,regTypes._endo))
        self.afprop.endoDiv.clicked.connect(lambda:self._divideRegions(objNames._endomesh,regTypes._endo))
        self.afprop.endoEdit.clicked.connect(lambda:self._editLandmarks(objNames._endomesh,regTypes._endo))

        self.afprop.epiReg.clicked.connect(lambda:self._registerLandmarks(objNames._epimesh,regTypes._epi))
        self.afprop.epiDiv.clicked.connect(lambda:self._divideRegions(objNames._epimesh,regTypes._epi))
        self.afprop.epiEdit.clicked.connect(lambda:self._editLandmarks(objNames._epimesh,regTypes._epi))
        
        self.afprop.genButton.clicked.connect(self._generate)
        
        return prop
        
    def updatePropBox(self,proj,prop):
        Project.updatePropBox(self,proj,prop)

        scenemeshes=[o for o in self.memberObjs if isinstance(o,eidolon.MeshSceneObject)]

        names=sorted(o.getName() for o in scenemeshes)
        eidolon.fillList(self.afprop.atlasBox,names,self.configMap.get(objNames._atlasmesh,-1))
        eidolon.fillList(self.afprop.endoBox,names,self.configMap.get(objNames._endomesh,-1))
        eidolon.fillList(self.afprop.epiBox,names,self.configMap.get(objNames._epimesh,-1))
        
    @eidolon.taskmethod('Adding Object to Project')
    def checkIncludeObject(self,obj,task):
        '''Check whether the given object should be added to the project or not.'''

        if not isinstance(obj,eidolon.MeshSceneObject) or obj in self.memberObjs or obj.getObjFiles() is None:
            return

        @timing
        def _copy():
            self.mgr.removeSceneObject(obj)
            self.addMesh(obj)

        pdir=self.getProjectDir()
        files=list(map(os.path.abspath,obj.getObjFiles() or []))

        if not files or any(not f.startswith(pdir) for f in files):
            msg="Do you want to add %r to the project? This requires saving/copying the object's file data into the project directory."%(obj.getName())
            self.mgr.win.chooseYesNoDialog(msg,'Adding Object',_copy)
            
    def addMesh(self,obj):
        filename=self.getProjectFile(obj.getName())
        self.VTK.saveObject(obj,filename,setFilenames=True)
        self.addObject(obj)
        self.mgr.addSceneObject(obj)
        self.save()
            
    def createTempDir(self,prefix='tmp'):
        path=self.getProjectFile(prefix+datetime.datetime.now().strftime('%Y%m%d%H%M%S'))
        os.mkdir(path)
        return path
            
    def _registerLandmarks(self,meshname,regtype):
        atlas=self.getProjectObj(self.configMap.get(objNames._atlasmesh,''))
        subj=self.getProjectObj(self.configMap.get(meshname,''))
        
        assert atlas is not None
        assert subj is not None
        
        endo=self.getProjectObj(regtype)
        
        if endo is not None:
            self.mgr.removeSceneObject(endo)
            
#        tempdir=self.createTempDir('reg') 
        tempdir=self.getProjectFile('reg20180530193140') # TODO: for testing
            
        result=self.AtrialFibre.registerLandmarks(subj,atlas,regtype,tempdir)
        
        self.mgr.checkFutureResult(result)
        
        @eidolon.taskroutine('Add points')
        def _add(task):
            obj=eidolon.Future.get(result)
            obj.setName(regtype+'nodes')
            self.addMesh(obj)
            
        self.mgr.runTasks(_add())
        
    def _editLandmarks(self,meshname,regtype):
        pass
    
    def _divideRegions(self,meshname,regtype):
        mesh=self.getProjectObj(self.configMap.get(meshname,''))
        points=self.getProjectObj(regtype+'nodes')
        
        assert mesh is not None
        assert points is not None
        
        result=self.AtrialFibre.divideRegions(mesh,points,regtype)
        
        self.mgr.checkFutureResult(result)
        
        @eidolon.taskroutine('Save mesh')
        def _save(task):
            self.VTK.saveObject(mesh,mesh.getObjFiles()[0])
            
            rep=mesh.createRepr(eidolon.ReprType._volume,0)
            self.mgr.addSceneObjectRepr(rep)
            rep.applyMaterial('Rainbow',field=regionField,valfunc='Column 1')
            self.mgr.setCameraSeeAll()
            
#            lobj,lrep=showLines(points.datasets[0].getNodes(),lmlines,'AllLines','Red')
            
        self.mgr.runTasks(_save())
            
    def _generate(self):
        endomesh=self.getProjectObj(self.configMap.get(objNames._endomesh,''))
        epimesh=self.getProjectObj(self.configMap.get(objNames._epimesh,''))
        
        if endomesh is None:
            self.mgr.showMsg('Cannot find endo mesh %r'%self.configMap.get(objNames._endomesh,''))
        elif epimesh is None:
            self.mgr.showMsg('Cannot find epi mesh %r'%self.configMap.get(objNames._epimesh,''))
        elif endomesh.datasets[0].getDataField('regions') is None:
            self.mgr.showMsg('Endo mesh does not have region field assigned!')
        elif epimesh.datasets[0].getDataField('regions') is None:
            self.mgr.showMsg('Epi mesh does not have region field assigned!')
        else:
            tempdir=self.createTempDir('dirs') 
            endopoints=self.getProjectObj('endonodes')
            epipoints=self.getProjectObj('epinodes')
            regions=[]
            
            result=self.AtrialFibre.generateMesh(endomesh,epimesh,endopoints,epipoints,tempdir,regions)
            self.mgr.checkFutureResult(result)
#            self.mgr.addSceneObjectTask(result)


class AtrialFibrePlugin(ScenePlugin):
    def __init__(self):
        ScenePlugin.__init__(self,'AtrialFibre')
        self.project=None

    def init(self,plugid,win,mgr):
        ScenePlugin.init(self,plugid,win,mgr)
        self.VTK=self.mgr.getPlugin('VTK')
        
        assert self.VTK is not None, 'Cannot find VTK plugin!'
        
        if self.win!=None:
            self.win.addMenuItem('Project','AtrialFibreProj'+str(plugid),'&Atrial Fibre Project',self._newProjDialog)
            
        # extract the deformetrica zip file if not present
        if not os.path.isdir(deformDir):
            z=zipfile.ZipFile(deformDir+'.zip')
            z.extractall(scriptdir)
            os.chmod(deformExe,stat.S_IRUSR|stat.S_IXUSR|stat.S_IWUSR)
            
        self.mirtkdir=os.path.join(eidolon.getAppDir(),eidolon.LIBSDIR,'MIRTK','Linux')
        eidolon.addPathVariable('LD_LIBRARY_PATH',self.mirtkdir)
        self.decimate=os.path.join(self.mirtkdir,decimate)
        
    def _newProjDialog(self):
        def chooseProjDir(name):
            newdir=self.win.chooseDirDialog('Choose Project Root Directory')
            if len(newdir)>0:
                self.createProject(name,newdir)

        self.win.chooseStrDialog('Choose Project Name','Project',chooseProjDir)

    def createProject(self,name,parentdir):
        if self.project==None:
            self.mgr.createProjectObj(name,parentdir,AtrialFibreProject)

    @eidolon.taskmethod('Registering landmarks')
    def registerLandmarks(self,meshObj,atlasObj,regtype,outdir,task=None):
        output=registerSubjectToTarget(meshObj,atlasObj,outdir,self.decimate,self.VTK)
        eidolon.printFlush(output)
        
        points=transferLandmarks(architecture,regtype,atlasObj,meshObj,outdir,self.VTK)
        
        subjnodes=meshObj.datasets[0].getNodes()
        ptds=eidolon.PyDataSet('pts',[subjnodes[n[0]] for n in points],[('landmarkField','',points)])
        
        return eidolon.MeshSceneObject('LM',ptds)
    
    @eidolon.taskmethod('Dividing mesh into regions')
    def divideRegions(self,mesh,points,regtype,task=None):
        lmlines,lmregions=loadArchitecture(architecture,regtype)[1:3]
        
        filledregions,linemap=generateRegionField(mesh,points,lmregions,task)
        mesh.datasets[0].setDataField(filledregions)
        
    @eidolon.taskmethod('Generating mesh')  
    def generateMesh(self,endomesh,epimesh,endopoints,epipoints,outdir,regions=[],task=None):
        calculateDirectionField(endomesh,endopoints,regions,regTypes._endo,outdir,self.VTK)

### Add the project

eidolon.addPlugin(AtrialFibrePlugin()) # note this occurs after other projects are loaded and is not in the subprocesses namespaces
