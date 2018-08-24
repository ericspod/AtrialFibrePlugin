
from eidolon import *
from AtrialFibrePlugin import *

for rt in regTypes:
    rt=rt[0]
    landmarks,lmlines,allregions,lmstim,lmground,appendageregion,appendagenode=loadArchitecture(architecture,rt)
    
    print('max region:',max(range(len(allregions)),key=lambda i:len(allregions[i])))
    
    print(rt,'lines extra')
    for i,line in enumerate(lmlines):
        if max(line)>=len(landmarks):
            print(i,line,[l for l in line if l>=len(landmarks)])
            
     
