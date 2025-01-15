import pandas as pd 
import numpy as np 
import os, sys 

sys.path.append("./../programmeExtractionConditionsAuxLimites/_python/")
sys.path.append('./../FEMU-Dshape/preparationModeleEF/')

from classDicData    import dicData
from classAbaqusMesh import classAbaqusMesh


dicoInitialisation = {'strPathDicData'     : './DIC-0-0-meilleurPlanIdentifie//',
                      'strFileName'        : 'imageSynthetique-0-0_0_0.synthetic.tif.csv',
                      'strXnamePixel'      : 'Image X[Pixel]',
                      'strYnamePixel'      : 'Image Y[Pixel]',
                      'strUnamePixel'      : 'Horizontal: u[Pixel]',
                      'strVnamePixel'      : 'Vertical: v[Pixel]',
                      'strXname'           : 'X[mm]',
                      'strYname'           : 'Y[mm]',
                      'strZname'           : 'Z[mm]',
                      'strUname'           : 'Horizontal Displacement U[mm]',
                      'strVname'           : 'Vertical Displacement V[mm]',
                      'strWname'           : 'Out-Of-Plane: W[mm]',
                      'strFieldsDelimiter' : ';',
                      'intStepSize'        : 5,
                      }

# Initialisation de la classe :
DIC_notchedReference = dicData(dicoInitialisation)

# Chargement du maillage :
FE_notchedReduit = classAbaqusMesh("Notched-reduit.inp")

# Determiner la table de connectivite des elements en surface :
FE_notchedReduit.extract2DconnectivityTable("ELT_SURFACE", "NODE_SURFACE", twoDimensionAssumption=True)

# Determiner les sets de noeuds qui partage la meme valeur en X pour appliquer les conditions aux limites :
FE_notchedReduit.determineNSET("NODE_BOT", "X", "nSet_BOT.inp")
quit()

# Inverser l'axe y :
FE_notchedReduit.nodes[:, 1] *= -1

# Selectionner les noeuds en surface : 
nodes = FE_notchedReduit.nodes[FE_notchedReduit.nodeSets['NODE_SURFACE']][:, [0,1,3]]


DIC_notchedReference.projectFiniteElementMesh_simpleMethod(nodes,
                                                           5.95, # Conversion factor
                                                           600., # uTranslation
                                                           600., # vTranslation
                                                           0.,   # rotationAngle
                                                           )

PLOT = False 
# PLOT = True 
if PLOT is True : 
    DIC_notchedReference.plotMeshOnImage_simpleMethod("PLOT-superpositionImageSynthetiqueMaillage",
                                                      "DIC-0-0-meilleurPlanIdentifie/imageSynthetique-0-0_0_0.synthetic.tif",
                                                      connectivityTable = FE_notchedReduit.twoDimensionnalConnectivityTable
                                                      )



