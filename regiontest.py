
mgr=mgr # pylint:disable=invalid-name,used-before-assignment
scriptdir=scriptdir # pylint:disable=invalid-name,used-before-assignment
CardiacMotion=CardiacMotion # pylint:disable=invalid-name,used-before-assignment
VTK=VTK # pylint:disable=invalid-name,used-before-assignment

from eidolon import *

from AtrialFibrePlugin import loadArchitecture, assignRegion, generateRegionField


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

#atlasobj=VTK.loadObject(atlas)
#mgr.addSceneObject(atlasobj)

atlasobj=mgr.objs[0]
nodes=atlasobj.datasets[0].getNodes()
tris=atlasobj.datasets[0].getIndexSet('tris')

lmnodes=[nodes[lm] for lm in landmarks] # landmark node indices are 0-based?

#allregions=[list(successive(r,2,True)) for r in lmregions]


allregions=[]
for r in lmregions:
    lr=[(a,b) for a,b in lmlines if a in r and b in r]
    
    if set(r)!=set(listSum(lr)):
        printFlush(r,lr)
            
    if len(lr)>2:
        allregions.append(lr)


## bad lines: 40, 
#linessubset=lmlines
##linessubset=lmlines[:40]
#for l,line in enumerate(linessubset):
#    for i in findTrisBetweenNodes(line[0],line[1],landmarks,graph):
#        filledregions[i]=l

#for r,region in enumerate(allregions[:1]):
#    robj,rrep=showLines(lmnodes,region,'Region','Blue')
#
#    for lml in region:
#    #lml=allregions[5][-1]
#        for i in findTrisBetweenNodes(lml[0],lml[1],landmarks,graph):
#            filledregions[i]=r+1#max(r+5+i,filledregions[i])
 
       

# endo region 10-12 are vessel and appendage
# endo region 0,14, 19, 22, 24, 25 bad, 11 weird
# epi region 15, 20, 23, 25, 26 bad, 37 weird

#badendo=[0,14,19,22,24,25]
#badepi=[15, 20, 23, 25, 26]
#goodregions=[r for i,r in enumerate(allregions) if i not in badepi]
      
goodregions=allregions


def testRegions(obj,landmarks,lmnodes,regions):
    ds=obj.datasets[0]
    nodes=ds.getNodes()
#    tris=ds.getIndexSet('tris')
    tris=first(ind for ind in ds.enumIndexSets() if ind.m()==3 and bool(ind.meta(StdProps._isspatial)))
    
    graph=TriMeshGraph(nodes,tris)
    
    filledregions=RealMatrix('regions',tris.n(),1)
    filledregions.fill(0)
    ds.setDataField(filledregions)

    for rindex in range(len(regions)):
        region=regions[rindex]
        robj,rrep=showLines(lmnodes,region,'Region','Red')
        #assignRegion(region,rindex+1,filledregions,landmarks,graph)  
        
    return filledregions


#filledregions=testRegions(atlasobj,landmarks,lmnodes,goodregions)

#filledregions=generateRegionField(atlasobj,landmarks,goodregions)

rep=atlasobj.createRepr(ReprType._volume,matname='Rainbow',field='regions')
mgr.addSceneObjectRepr(rep)


lobj,lrep=showLines(lmnodes,lmlines,'AllLines','Green')
#robj,rrep=showLines(lmnodes,listSum(allregions),'AllRegions','Blue')


mgr.setCameraSeeAll()