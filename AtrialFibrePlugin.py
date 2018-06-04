
import os
import stat
import ast
import shutil
import datetime
import zipfile
from collections import defaultdict

try:
    import configparser
except ImportError:
    import ConfigParser as configparser

from eidolon import ScenePlugin, Project, avg, vec3, listSum, successive, first, RealMatrix, IndexMatrix, StdProps
import eidolon, ui

scriptdir= os.path.dirname(os.path.abspath(__file__)) # this file's directory

# directory/file names
uifile=os.path.join(scriptdir,'AtrialFibrePlugin.ui') 
deformdir=os.path.join(scriptdir,'deformetricaC')
deformExe=os.path.join(deformdir,'deformetrica')
architecture=os.path.join(scriptdir,'architecture.ini') 

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

objNames=eidolon.enum(
    'atlasmesh',
    'epimesh','epinodes',
    'endomesh','endonodes',
    'architecture'
)

regTypes=eidolon.enum('endo','epi')


# load the UI file into the ui namespace, this is subtyped below
ui.loadUI(open(uifile).read())


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
    def __init__(self,nodes,tris):
        self.nodes=nodes
        self.tris=tris
        self.tricenters=[avg(nodes.mapIndexRow(tris,r),vec3()) for r in range(tris.n())]
        self.adj=generateTriAdj(tris)
        self.nodelem=generateNodeElemMap(nodes.n(),tris) # node -> elems
        
        def computeDist(key):
            i,j=key
            return self.tricenters[i].distToSq(self.tricenters[j])
            
        self.tridists=initdict(computeDist)
        
    def getPathSubgraph(self,starttri,endtri):
        return getAdjTo(self.adj,starttri,endtri)
        
    def getTriNodes(self,triindex):
        return self.nodes.mapIndexRow(self.tris,triindex)
    
    def getSharedNodeTris(self,triindex):
        return listSum([self.nodelem[n] for n in self.tris[triindex]])
        
    def getPath(self,starttri,endtri,acceptTri=None):
        return dijkstra(self.adj,self.tridists,starttri,endtri,acceptTri)
    
    
def loadArchitecture(path,section):
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
    lmstim=stimulus[:len(lmregions)]
    lmground=ground[:len(lmregions)]
    
    

    return landmarks,lmlines,lmregions,lmstim,lmground
    

def registerSubjectToTarget(subjectObj,targetObj,outdir,decimpath,VTK):
    '''
    Register the `subjectObj' mesh object to `targetObj' mesh object putting data into directory `outdir'. The subject 
    will be decimated to have roughly the same number of nodes as the target and then stored as subject.vtk in `outdir'. 
    Registration is done with Deformetrica and result stored as 'Registration_subject_to_subject_0__t_9.vtk' in `outdir'.
    '''
    dpath=os.path.join(outdir,decimatedFile)
    tmpfile=os.path.join(outdir,'tmp.vtk')
    
    shutil.copy(os.path.join(deformdir,datasetFile),os.path.join(outdir,datasetFile))
 
    model=open(os.path.join(deformdir,modelFile)).read()
    model=model.replace('%1',str(dataSigma))
    model=model.replace('%2',str(kernelWidthSub))
    model=model.replace('%3',str(kernelType))
    model=model.replace('%4',str(kernelWidthDef))

    with open(os.path.join(outdir,modelFile),'w') as o:
        o.write(model)
        
    optim=open(os.path.join(deformdir,optimFile)).read()
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


def transferLandmarks(landmarks,sourceObj,subjectObj,outdir,VTK):
    '''
    Register the landmarks defined as node indices on `sourceObj' to equivalent node indices on `subjectObj' via the
    decimated and registered intermediary stored in `outdir'. The result is a list of index pairs associating a node
    index in `subjectObj' for every landmark index in `sourceObj'.
    '''
#    target=os.path.join(outdir,targetFile)
    decimated=os.path.join(outdir,decimatedFile)
    registered=os.path.join(outdir,registeredFile)
    
#    targ=VTK.loadObject(target) # target
    reg=VTK.loadObject(registered) # mesh registered to target
    dec=VTK.loadObject(decimated) # decimated unregistered mesh
    #subj=VTK.loadObject(subject) # original mesh which was decimated then registered
    
    tnodes=sourceObj.datasets[0].getNodes() # target points
    rnodes=reg.datasets[0].getNodes() # registered decimated points
    dnodes=dec.datasets[0].getNodes() # unregistered decimated points
    snodes=subjectObj.datasets[0].getNodes() # original subject points
    
    lmpoints=[(tnodes[m],m) for m in landmarks]
    
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
    '''Returns a map relating each node index to the set of element indices using that node.'''
    nodemap=[set() for _ in range(numnodes)]

    for i,tri in enumerate(tris):
        for n in tri:
            nodemap[n].add(i)
            
    return nodemap
    

def dijkstra(adj,tridists, start, end,acceptTri=None):
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
            weight = curweight+tridists[(curnode,dest)] 
            
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
        

def findTrisBetweenNodes(start,end,landmarks,graph):
    start=landmarks[start]
    end=landmarks[end]
    
    nodes=graph.nodes
    
    # define planes to bound the areas to search for triangles to within the space of the line
    splane=plane(nodes[start],nodes[end]-nodes[start])
    eplane=plane(nodes[end],nodes[start]-nodes[end])

    # adjust the plane's positions to account for numeric error
    adjustdist=1e-1
    splane.moveUp(-adjustdist)
    eplane.moveUp(-adjustdist)
    
#    starttri=first(n for n in graph.nodelem[start] if splane.between(graph.getTriNodes(n),eplane))
#    endtri=first(n for n in graph.nodelem[end] if splane.between(graph.getTriNodes(n),eplane))
    starttri=first(graph.nodelem[start])
    endtri=first(graph.nodelem[end])
    
    assert starttri is not None
    assert endtri is not None
    
    # find the triangle center nearest to the midpoint of the line
    midplane=plane((splane.center+eplane.center)*0.5,splane.norm)
    midinds=midplane.findIntersects(nodes,graph.tris)
    midind=findNearestIndex(midplane.center,[graph.tricenters[m] for m in midinds])
    midtri=graph.tricenters[midinds[midind]]
    
    # use the midpoint triangle to calculate the normal for the plane on the line between start and end nodes
    lineplane=plane(splane.center,midtri.planeNorm(splane.center,eplane.center))
    
    indices=set([starttri,endtri]) # list of element indices on lineplane between splane and eplane
    
    for i in range(graph.tris.n()):
        trinodes=graph.getTriNodes(i)
        
        numabove=lineplane.numPointsAbove(trinodes)
        if numabove in (1,2) and splane.between(trinodes,eplane):    
            indices.add(i)
    
    accepted=[starttri]
    
    # find the first triangle adjacent to the starting triangle which is in the indices of triangles on the plane
    adjacent=first(i for i in graph.adj[accepted[-1]] if i in indices and i not in accepted)
    
    # if no adjacent triangle found then the first triangle may not be adjacent but does use the `start' node so select that one
    if adjacent is None:
        adjacent=first(i for i in graph.nodelem[start] if i in indices)
        accepted=[]
    
    # add each triangle to the accepted list which is contiguous with the starting triangle/node
    while adjacent is not None:
        accepted.append(adjacent)
        
        allneighbours=set()
        for a in accepted:
            allneighbours.update(graph.adj[a])
            
        adjacent=first(i for i in allneighbours if i in indices and i not in accepted)
    
    # failed to find a straight line path in the selected indices, default to dijsktra shortest path
    if endtri not in accepted:
        accepted=graph.getPath(starttri,endtri)
    else:
        try:
            accept=lambda a:(a in accepted or any(b in accepted for b in graph.getSharedNodeTris(a)))
        
            accepted=min([accepted,graph.getPath(starttri,endtri,accept)],key=len)
        except:
            pass
        
    return accepted
    
    
def assignRegion(region,index,assignmat,landmarks,graph):

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
    for i,lml in enumerate(region):
        line=findTrisBetweenNodes(lml[0],lml[1],landmarks,graph)
        bordertris.update(line)

    # find the two subgraphs formed by dividing the graph along the borders, the smaller of the two is the enclosed set of tris
        
    outside1=first(i for i in range(graph.adj.n()) if i not in bordertris) # first tri not on the border
    subgraph1=getEnclosedGraph(graph.adj,bordertris,outside1) # 
    
    subgraph1tris=set(bordertris)
    subgraph1tris.update(subgraph1)
    
    outside2=first(i for i in range(graph.adj.n()) if i not in subgraph1tris) # first tri not on border or in subgraph1
    subgraph2=getEnclosedGraph(graph.adj,subgraph1tris,outside2)
        
    mingraph=min([subgraph1,subgraph2],key=len) # the smaller graph is the one within the region

    for tri in bordertris.union(mingraph):
        assignmat[tri,1]=assignmat[tri,0]
        assignmat[tri,0]=index
    
    
def generateRegionField(obj,landmarkObjs,regions,task=None):
    ds=obj.datasets[0]
    nodes=ds.getNodes()
    lmnodes=landmarkObjs.datasets[0].getNodes()
    
#    tris=ds.getIndexSet('tris')
    tris=first(ind for ind in ds.enumIndexSets() if ind.m()==3 and bool(ind.meta(StdProps._isspatial)))
    
    landmarks=[nodes.indexOf(lm)[0] for lm in lmnodes]
    
    graph=TriMeshGraph(nodes,tris)
    
    filledregions=RealMatrix('regions',tris.n(),2)
    filledregions.fill(-10)
    ds.setDataField(filledregions)

    if task:
        task.setMaxProgress(len(regions))

    for rindex in range(0,len(regions)):
        region=regions[rindex]
        assignRegion(region,rindex,filledregions,landmarks,graph)    
        
        if task:
            task.setProgress(rindex)
        
    return filledregions


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

        @eidolon.timing
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
            
        tempdir=self.createTempDir('reg') 
#        tempdir=self.getProjectFile('reg20180530193140') 
            
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
            
        self.mgr.runTasks(_save())
            
    def _generate(self):
        endomesh=self.getProjectObj(self.configMap.get(regTypes._endo,''))
        epimesh=self.getProjectObj(self.configMap.get(regTypes._epi,''))
        
        if endomesh.datasets[0].getDataField('regions')==None:
            self.mgr.showMsg('Endo mesh does not have region field assigned!')
        elif epimesh.datasets[0].getDataField('regions')==None:
            self.mgr.showMsg('Epi mesh does not have region field assigned!')
        else:
            result=self.AtrialFibre.generateMesh(endomesh,epimesh)
            self.mgr.addSceneObjectTask(result)

class AtrialFibrePlugin(ScenePlugin):
    def __init__(self):
        ScenePlugin.__init__(self,'AtrialFibre')
        self.project=None

    def init(self,plugid,win,mgr):
        ScenePlugin.init(self,plugid,win,mgr)
        
        if self.win!=None:
            self.win.addMenuItem('Project','AtrialFibreProj'+str(plugid),'&Atrial Fibre Project',self._newProjDialog)
            
        # extract the deformetrica zip file if not present
        if not os.path.isdir(deformdir):
            z=zipfile.ZipFile(deformdir+'.zip')
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

    def getCWD(self):
        return self.project.getProjectDir()
        
    @eidolon.taskmethod('Registering landmarks')
    def registerLandmarks(self,meshObj,atlasObj,regtype,outdir,task=None):
        VTK=self.mgr.getPlugin('VTK')
        assert VTK is not None
        
        output=registerSubjectToTarget(meshObj,atlasObj,outdir,self.decimate,VTK)
        eidolon.printFlush(output)
        
        lmarks,lines=loadArchitecture(architecture,regtype)[0:2]
        
        points=transferLandmarks(lmarks,atlasObj,meshObj,outdir,VTK)
        
        subjnodes=meshObj.datasets[0].getNodes()
        indices=[('lines',eidolon.ElemType._Line1NL,lines),('landmarkField','',points)]
        ptds=eidolon.PyDataSet('pts',[subjnodes[n[0]] for n in points],indices)
        
        return eidolon.MeshSceneObject('LM',ptds)
    
    @eidolon.taskmethod('Dividing mesh into regions')
    def divideRegions(self,mesh,points,regtype,task=None):
        lmlines,lmregions=loadArchitecture(architecture,regtype)[1:3]
        
        allregions=[]
        for r in lmregions:
            lr=[(a,b) for a,b in lmlines if a in r and b in r]
            
            if len(lr)>2:
                allregions.append(lr)
        
        generateRegionField(mesh,points,allregions,task)
    
    @eidolon.taskmethod('Generating mesh')  
    def generateMesh(self,endomesh,epimesh,task=None):
        pass


eidolon.addPlugin(AtrialFibrePlugin())
