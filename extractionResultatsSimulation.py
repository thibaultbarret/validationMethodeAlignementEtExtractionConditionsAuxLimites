from abaqusConstants import *
from symbolicConstants import *
from odbAccess import *

import numpy as np
import sys
import os
from textRepr import *

odbName     = sys.argv[1] 
# roiName     = sys.argv[2]
pathResults = sys.argv[2]

if not pathResults.endswith('/') : 
    pathResults += '/)'

if not os.path.exists(pathResults) : 
    os.mkdir(pathResults)


#========================================================================
#                              OPEN ODB : 
#========================================================================
if '.odb' in odbName : 
    if not odbName.endswith('.odb') : 
        exitProgram("File %s doesn't finished with .odb" %odbName)

else : 
    exitProgram("File %s doesn't finished with .odb" %odbName)

if not os.path.exists(odbName) : 
    exitProgram("File %s doesn't exist" %odbName)

else : 
    try : 
        odb = openOdb(odbName, readOnly=True)

    except : 
        exitProgram("Impossible to open %s" %odbName)

myInstance = odb.rootAssembly.instances[odb.rootAssembly.instances.keys()[0]]
# print(odb.rootAssembly.nodeSets)
# print(myInstance.nodeSets.keys())

stepName = odb.steps.keys()[0]

for i, incr in enumerate(odb.steps[stepName].frames) : 
    print(i)

    if i > 0 : 
        # print(incr.fieldOutputs.keys())
        C = incr.fieldOutputs['COORD'].getSubset(region = odb.rootAssembly.nodeSets[' ALL NODES'])
        U = incr.fieldOutputs['U']
        LE = incr.fieldOutputs['LE'].getSubset(region = odb.rootAssembly.elementSets[' ALL ELEMENTS'], position = CENTROID)

        PEEQ = incr.fieldOutputs['SDV_EQUIVALENTPLASTICSTRAIN']

        with open(pathResults + "Frame-%i-nodes.res" %i, 'w') as nodes, open(pathResults + "Frame-%i-elements.res" %i, "w") as elements : 
            nodes.write("Label;X;Y;Z;U;V;W\n")

            elements.write("Label;LE11;LE22;LE33;LE12;LE13;LE23;PEEQ\n")

            # Loop over coordinates and displacements
            for c, u in zip(C.values, U.values) : 
                nodes.write("%i;%f;%f;%f;%f;%f;%f\n" %(c.nodeLabel,
                                                          c.data[0],
                                                          c.data[1],
                                                          c.data[2],
                                                          u.data[0],
                                                          u.data[1],
                                                          u.data[2])
                            )

            for le, peeq in zip(LE.values, PEEQ.values) : 
                elements.write("%i;%f;%f;%f;%f;%f;%f;%f\n" %(le.elementLabel,
                                                             le.data[0],
                                                             le.data[1],
                                                             le.data[2],
                                                             le.data[3],
                                                             le.data[4],
                                                             le.data[5],
                                                             peeq.data)
                               )



    else : 
        ptCoord = incr.fieldOutputs['COORD'].getSubset(region = odb.rootAssembly.elementSets[' ALL ELEMENTS'], position = CENTROID)
        with open(pathResults + "ptIntegrationCoordinate.res", 'w') as file : 
            file.write("Label;X;Y;Z\n")

            print(len(ptCoord.values))
            for pt in ptCoord.values : 
                file.write("%i;%f;%f;%f\n" %(pt.elementLabel,
                                             pt.data[0],
                                             pt.data[1],
                                             pt.data[2],
                                             )
                           )

