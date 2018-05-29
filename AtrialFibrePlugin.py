
import os
import ast
import shutil
import datetime

try:
    import configparser
except ImportError:
    import ConfigParser as configparser

from eidolon import ScenePlugin, Project
import eidolon, ui

scriptdir= os.path.dirname(os.path.abspath(__file__)) # this file's directory

uifile=os.path.join(scriptdir,'AtrialFibrePlugin.ui') 
deformdir=os.path.join(scriptdir,'deformetricaC')
deformExe=os.path.join(deformdir,'deformetrica')
architecture=os.path.join(scriptdir,'architecture.ini') 

decimatedFile='subject.vtk'
datasetFile='data_set.xml'
modelFile='model.xml'
optimFile='optimization_parameters.xml'
registeredFile='Registration_subject_to_subject_0__t_9.vtk'

# deformetrica parameters
kernelWidthSub=5000
kernelWidthDef=8000
kernelType='cudaexact'
dataSigma=0.1
stepSize=0.000001

#decimate=os.path.join(mirtk,'decimate-surface') # TODO: fix path


objNames=eidolon.enum(
    'atlasmesh',
    'epimesh','epinodes',
    'endomesh','endonodes',
    'architecture'
)

regTypes=eidolon.enum('endo','epi')


# load the UI file into the ui namespace, this is subtyped below
ui.loadUI(open(uifile).read())


def subone(v):
    return tuple(i-1 for i in v)
    
    
def loadArchitecture(path,section):
    c=configparser.SafeConfigParser()
    assert len(c.read(path))>0
    
    landmarks=ast.literal_eval(c.get(section,'landmarks')) # 0-based node indices
    lines=ast.literal_eval(c.get(section,'lines')) # 1-based landmark indices
    regions=ast.literal_eval(c.get(section,'regions')) # 1-based landmark indices
    stimulus=ast.literal_eval(c.get(section,'stimulus')) # per region
    ground=ast.literal_eval(c.get(section,'ground')) # per region
#    types=ast.literal_eval(c.get('endo','type')) # per region    
    
    lmlines=[subone(l) for l in lines if max(l)<=len(landmarks)] # filter out lines with existing node indices
    lmregions=[subone(r) for r in regions if all(i<=len(landmarks) for i in r)]
    lmstim=stimulus[:len(lmregions)]
    lmground=ground[:len(lmregions)]

    return landmarks,lmlines,lmregions,lmstim,lmground
    

def registerSubjectToTarget(subjectObj,targetObj,outdir):
    '''
    Register the `subjectObj' mesh to the `targetObj' VTK mesh object putting data into directory `outdir'. The subject 
    will be decimated to have roughly the same number of nodes as the target mesh and then stored as subject.vtk in 
    `outdir'. Registration is done with Deformetrica and result stored as 'Registration_subject_to_subject_0__t_9.vtk' 
    in `outdir'.
    '''
    dpath=os.path.join(outdir,decimatedFile)
    
    shutil.copy(os.path.join(deformdir,datasetFile),os.path.join(outdir,datasetFile))
    #shutil.copy(os.path.join(deformdir,target),os.path.join(outdir,targetFile))
    
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
        
    snodes=subjectObj.datasets[0].getNodes()
    tnodes=targetObj.datasets[0].getNodes()
    
    sizeratio=float(tnodes.n())/snodes.n()
    sizepercent=str(100*(1-sizeratio))[:6] # percent to decimate by
    
    # decimate the mesh most of the way towards having the same number of nodes as the atlas
    ret,output=eidolon.execBatchProgram(decimate,subjectObj.getObjFiles()[0],dpath,'-reduceby',sizepercent,'-ascii',logcmd=True)
    assert ret==0,output
    
#    dobj=VTK.loadObject(dpath)
#    assert dobj.datasets[0].getNodes().n()>0
    
    
    ret,output=eidolon.execBatchProgram(deformExe,"registration", "3D", modelFile, datasetFile, optimFile, "--output-dir=.",cwd=outdir,logcmd=True)
    assert ret==0,output
    
    return output


def transferLandmarks(archFilename,fieldname,sourceObj,subjectObj,outdir,VTK):
    '''
    Register the landmarks defined as node indices on `sourceObj' to equivalent node indices on `subjectObj' via the
    decimated and registered intermediary stored in `outdir'. The result is a list of index pairs associating a node
    index in `subjectObj' for every landmark index in `sourceObj'.
    '''
#    target=os.path.join(outdir,targetFile)
    decimated=os.path.join(outdir,decimatedFile)
    registered=os.path.join(outdir,registeredFile)
    
    lmarks=loadArchitecture(archFilename,fieldname)[0]
    
#    targ=VTK.loadObject(target) # target
    reg=VTK.loadObject(registered) # mesh registered to target
    dec=VTK.loadObject(decimated) # decimated unregistered mesh
    #subj=VTK.loadObject(subject) # original mesh which was decimated then registered
    
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


class AtrialFibrePropWidget(ui.QtWidgets.QWidget,ui.Ui_AtrialFibre):
    def __init__(self,parent=None):
        super(AtrialFibrePropWidget,self).__init__(parent)
        self.setupUi(self)
        

class AtrialFibreProject(Project):
    def __init__(self,name,parentdir,mgr):
        Project.__init__(self,name,parentdir,mgr)
        self.header='AtrialFibre.createProject(%r,scriptdir+"/..")\n' %(self.name)
#        self.architecture=None
        
        self.AtrialFibre=mgr.getPlugin('AtrialFibre')
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
        self.afprop.epiReg.clicked.connect(lambda:self._registerLandmarks(objNames._epimesh,regTypes._epi))
        self.afprop.endoEdit.clicked.connect(lambda:self._editLandmarks(objNames._endomesh,regTypes._endo))
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

        if not isinstance(obj,eidolon.MeshSceneObject) or obj in self.memberObjs or obj.plugin.getObjFiles(obj) is None:
            return

        def _copy():
            self.mgr.removeSceneObject(obj)
            filename=self.getProjectFile(obj.getName())
            
            VTK=self.mgr.getPlugin('VTK')
            VTK.saveObject(obj,filename,setFilenames=True)

            self.mgr.addSceneObject(obj)
            Project.addObject(self,obj)

            self.save()

        pdir=self.getProjectDir()
        files=list(map(os.path.abspath,obj.plugin.getObjFiles(obj) or []))

        if not files or any(not f.startswith(pdir) for f in files):
            msg="Do you want to add %r to the project? This requires saving/copying the object's file data into the project directory."%(obj.getName())
            self.mgr.win.chooseYesNoDialog(msg,'Adding Object',_copy)
            
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
            
        result=self.AtrialFibre.registerLandmarks(subj,atlas,regtype,tempdir)
        
        @eidolon.taskroutine('Add points')
        def _add(task):
            obj=eidolon.Future.get(result)
            obj.setName(regtype)
            self.addObject(obj)
            self.mgr.addSceneObject(obj)
            self.save()
            
        self.mgr.runTasks(_add())
        
    def _editLandmarks(self,meshname,regtype):
        pass
            

class AtrialFibrePlugin(ScenePlugin):
    def __init__(self):
        ScenePlugin.__init__(self,'AtrialFibre')
        self.project=None

    def init(self,plugid,win,mgr):
        ScenePlugin.init(self,plugid,win,mgr)
        
        if self.win!=None:
            self.win.addMenuItem('Project','AtrialFibreProj'+str(plugid),'&Atrial Fibre Project',self._newProjDialog)
        
    def _newProjDialog(self):
        def chooseProjDir(name):
            newdir=self.win.chooseDirDialog('Choose Project Root Directory')
            if len(newdir)>0:
                self.createProject(name,newdir)

        self.win.chooseStrDialog('Choose Project Name','Project',chooseProjDir)

    def createProject(self,name,parentdir):
        if self.project==None:
            self.mgr.createProjectObj(name,parentdir,AtrialFibreProject)
            
#        self.loadArchitecture(architecture)
        self.project.save()

    def getCWD(self):
        return self.project.getProjectDir()
        
    @eidolon.taskmethod('Registering landmarks')
    def registerLandmarks(self,meshObj,atlasObj,regtype,outdir,task=None):
        
        #output=registerSubjectToTarget(meshobj,atlasobj,outdir)
        
        #eidolon.printFlush(output)
        
        points=transferLandmarks(architecture,regtype,atlasObj,meshObj,outdir,self.mgr.getPlugin('VTK'))
        
        subjnodes=meshObj.datasets[0].getNodes()
        ptds=eidolon.PyDataSet('pts',[subjnodes[n[0]] for n in points],[('LMMap','',points)])
        
        return eidolon.MeshSceneObject('LM',ptds)
        
    @eidolon.taskmethod('Generate mesh')  
    def generateMesh(self,task=None):
        pass
    
        
#    def loadArchitecture(self,filename):
#        if self.project.architecture is not None:
#            self.mgr.removeSceneObject(self.project.architecture)
#            
#        self.project.architecture=eidolon.DatafileSceneObject(objNames._architecture,filename,{},self)
#        self.project.architecture.load()
#        self.project.addObject(self.project.architecture)
#        self.mgr.addSceneObject(self.project.architecture)


eidolon.addPlugin(AtrialFibrePlugin())
