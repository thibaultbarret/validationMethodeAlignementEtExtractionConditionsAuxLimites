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

dictDisplacementFileOpenning = {'strFolderPath'      : './DIC-0-0-meilleurPlanIdentifie/',
                                'strFilePrefix'      : 'imageSynthetique-0-0_',
                                'strFileSuffix'      : '_0.synthetic.tif.csv',
                                'intZeroNumber'      : 0,
                                'loadStep'           : np.linspace(1, 10, 10, dtype = int),
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
                                }

# Initialisation de la classe :
DIC_notchedReference = dicData(dicoInitialisation)

# Charger les fichiers de resulats de MatchID : 
DIC_notchedReference.loadDisplacementResults(dictDisplacementFileOpenning)

# Ajout du tableau correspondance numero_fichier/temps :
DIC_notchedReference.loadStepTimeFromArray(np.linspace((1, 0.1), (10, 1), 10))

# Chargement du maillage :
FE_notchedReduit = classAbaqusMesh("Notched-reduit.inp")

# Determiner la table de connectivite des elements en surface :
FE_notchedReduit.extract2DconnectivityTable("ELT_SURFACE", "NODE_SURFACE", twoDimensionAssumption=True)

# Determiner les sets de noeuds qui partage la meme valeur en X pour appliquer les conditions aux limites :
createNset = False
if createNset is True : 
    FE_notchedReduit.determineNSET("NODE_BOT", "X", "BOT", "nSet_BOT.inp", )
    FE_notchedReduit.determineNSET("NODE_TOP", "X", "TOP", "nSet_TOP.inp", )

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


# Extraction des conditions aux limites :
botNodes = FE_notchedReduit.nodes[FE_notchedReduit.nodeSets['NODE_BOT']]
botNodes = botNodes[botNodes[:, 2] == 0.4]
botNodes = botNodes[botNodes[:, 0].argsort()]

dicoExtractionConditionsAuxLimites = {'nodes'                      : botNodes,
                                      'nSetPrefix'                 : "BOT-",
                                      'amplitudePrefix'            : "Amplitude-BOT-",
                                      'amplitudeFileName'          : "amplitudeNsetBOT.inp",
                                      'boundaryConditionsFileName' : "boundaryConditionsNSetBot.inp",
                                      }

DIC_notchedReference.boundaryConditionsExtractionUsing_simpleMethod(dicoExtractionConditionsAuxLimites)


topNodes = FE_notchedReduit.nodes[FE_notchedReduit.nodeSets['NODE_TOP']]
topNodes = topNodes[topNodes[:, 2] == 0.4]
topNodes = topNodes[topNodes[:, 0].argsort()]

dicoExtractionConditionsAuxLimites = {'nodes'                      : topNodes,
                                      'nSetPrefix'                 : "TOP-",
                                      'amplitudePrefix'            : "Amplitude-TOP-",
                                      'boundaryConditionsFileName' : "boundaryConditionsNSetTop.inp",
                                      'amplitudeFileName'          : "amplitudeNsetTOP.inp",
                                      }

DIC_notchedReference.boundaryConditionsExtractionUsing_simpleMethod(dicoExtractionConditionsAuxLimites)
