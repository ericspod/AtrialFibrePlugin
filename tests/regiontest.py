
mgr=mgr # pylint:disable=invalid-name,used-before-assignment
scriptdir=scriptdir # pylint:disable=invalid-name,used-before-assignment
CardiacMotion=CardiacMotion # pylint:disable=invalid-name,used-before-assignment
VTK=VTK # pylint:disable=invalid-name,used-before-assignment

from eidolon import *

from AtrialFibrePlugin import loadArchitecture,TriMeshGraph, assignRegion, generateRegionField, findTrisBetweenNodes


def showLines(nodes,lines,name='Lines',matname='Default'):
    mgr=getSceneMgr()
    lineds=LineDataSet(name+'DS',nodes,lines)
    obj=MeshSceneObject(name,lineds)
    mgr.addSceneObject(obj)
    
    rep=obj.createRepr(ReprType._line,matname=matname)
    mgr.addSceneObjectRepr(rep)
    
    return obj,rep


atlas=scriptdir+'../meshdata/03finalAtlas.vtk'
architecture=scriptdir+'architecture.ini'


landmarks,lmlines,lmregions,lmstim,lmground=loadArchitecture(architecture,'endo')

#mesh=VTK.loadObject(atlas)
#mgr.addSceneObject(mesh)

mesh=mgr.objs[0]
nodes=mesh.datasets[0].getNodes()
tris=mesh.datasets[0].getIndexSet('tris')

#lmnodes=[nodes[lm] for lm in landmarks] # landmark node indices are 0-based?

movednodes=mgr.objs[1].datasets[0].getNodes()

movedlandmarks=[nodes.indexOf(lm)[0] for lm in movednodes]


allregions=[]
for r in lmregions:
    lr=[(a,b) for a,b in lmlines if a in r and b in r]
    
    if len(lr)>2:
        allregions.append(lr)


goodregions=[a for a in allregions if len(a)>6]


def testLines(obj,landmarks,lmnodes,regions):
    ds=obj.datasets[0]
    nodes=ds.getNodes()
#    tris=ds.getIndexSet('tris')
    tris=first(ind for ind in ds.enumIndexSets() if ind.m()==3 and bool(ind.meta(StdProps._isspatial)))
    
    graph=TriMeshGraph(nodes,tris)
    
    filledregions=RealMatrix('regions',tris.n(),2)
    filledregions.fill(-5)
    ds.setDataField(filledregions)
    
    for region in regions:
        for i,lml in enumerate(region):
            for tind in findTrisBetweenNodes(lml[0],lml[1],landmarks,graph):
                filledregions[tind,0]=i

    return filledregions
        

def testRegions(obj,landmarks,lmnodes,regions):
    ds=obj.datasets[0]
    nodes=ds.getNodes()
#    tris=ds.getIndexSet('tris')
    tris=first(ind for ind in ds.enumIndexSets() if ind.m()==3 and bool(ind.meta(StdProps._isspatial)))
    
    graph=TriMeshGraph(nodes,tris)
    
    filledregions=RealMatrix('regions',tris.n(),2)
    filledregions.fill(-5)
    ds.setDataField(filledregions)

    for rindex in range(len(regions)):
        region=regions[rindex]
        robj,rrep=showLines(lmnodes,region,'Region','Red')
        assignRegion(region,rindex+1,filledregions,landmarks,graph)  
        
    return filledregions
    

filledregions=testLines(mesh,movedlandmarks,movednodes,goodregions[0:1])

#filledregions=testRegions(mesh,movedlandmarks,movednodes,goodregions[0:1])

#filledregions=generateRegionField(mesh,landmarks,goodregions)

rep=mesh.createRepr(ReprType._volume,matname='Rainbow',field='regions')
mgr.addSceneObjectRepr(rep)


#lobj,lrep=showLines(lmnodes,lmlines,'AllLines','Green')
#robj,rrep=showLines(lmnodes,listSum(allregions),'AllRegions','Blue')


mgr.setCameraSeeAll()