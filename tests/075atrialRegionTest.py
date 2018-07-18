from eidolon import *
from AtrialFibrePlugin import TriMeshGraph,calculateGradientDirs

obj=VTK.loadObject('075atrialRegion.vtk')
mgr.addSceneObject(obj)

ds=obj.datasets[0]

graph=TriMeshGraph(ds.getNodes(),ds.getIndexSet('tris'))

dirs=calculateGradientDirs(graph,ds.getDataField('laplaceTM'))

printFlush(dirs)

ds.setDataField(dirs)

rep=obj.createRepr(ReprType._glyph,0,drawInternal=False,externalOnly=True,glyphname= 'arrow', dfield='dirs',vecfunc=VecFunc._Linear, glyphscale= (20, 20, 40))
mgr.addSceneObjectRepr(rep)

mgr.setCameraSeeAll()
