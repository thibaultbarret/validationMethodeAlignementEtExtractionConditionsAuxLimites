import os, sys 
import numpy as np 
import subprocess

def lancementCorrelation(MatchIDcommand: str, inputFile: str) : 
    """ 
        Fonction pour lancer la correlation
    """ 

    subprocess.run([MatchIDcommand, inputFile])
            

def ecrireFichierInstructions(inputFile, parametresCorrelation: dict) : 
    """
        Fonction pour ecrire le fichier d'instruction utilise par MatchID pour lancer la correlation
    """
    with open("%s.m3inp" %(inputFile), 'w') as MatchIDfile : 
        # Parametres des cameras :
        MatchIDfile.write("<Camera$0$index>=<_0.>\n")
        MatchIDfile.write("<Camera$1$index>=<_1.>\n")

        MatchIDfile.write("<Camera$0$intrinsic>=<1188.40579710145;1188.40579710145;0;0;0;0;0;0;600;600;0;0>\n")
        MatchIDfile.write("<Camera$1$intrinsic>=<1188.40579710145;1188.40579710145;0;0;0;0;0;0;600;600;0;0>\n")
        MatchIDfile.write("<Distortions>=<5>\n")

        MatchIDfile.write("<Extrinsic$parameters>=<-68.4040286651337;0;24.8970303380007;0;20;0>\n")

        # Images de references, images deformees :
        if parametresCorrelation['dossierCourant'] is True : 
            dossierCourant = os.getcwd()

        else : 
            dossierCourant = ''
        
        
        MatchIDfile.write("<Deformed$image>=<" +
                          os.path.join(dossierCourant, parametresCorrelation["nomDossier"],
                          "%s%i%s" %(parametresCorrelation['prefixNomImage'], parametresCorrelation['numeroImageNonDeformee'], parametresCorrelation['suffixNomImage'] ) +
                          ";True;1;0;0;SAME_None;CROSS_None>\n")
                          )

        for number in parametresCorrelation['listeNumeroImageDeformee'] : 
            MatchIDfile.write("<Deformed$image>=<" +
                              os.path.join(dossierCourant, parametresCorrelation["nomDossier"],
                              "%s%i%s" %(parametresCorrelation['prefixNomImage'], number, parametresCorrelation['suffixNomImage'] ) +
                              ";False;1;0;0;SAME_None;CROSS_None>\n")
                              )
        

        # Parametres de correlation : Subset, step, correlation ...
        MatchIDfile.write("<Subset$size>=<%s>\n" %parametresCorrelation['subsetSize'])
        MatchIDfile.write("<Step$size>=<%s>\n" %parametresCorrelation['stepSize'])

        # Critere de correlation :
        match parametresCorrelation['critereCorrelation'] :
            case 'NSSD' : 
                MatchIDfile.write("<Correlation>=<0>\n")

            case 'ANSSD' :
                MatchIDfile.write("<Correlation>=<1>\n")

            case 'CC' : 
                MatchIDfile.write("<Correlation>=<2>\n")

            case 'ZNCC' : 
                MatchIDfile.write("<Correlation>=<4>\n")

            case 'ZNSSD' : 
                MatchIDfile.write("<Correlation>=<5>\n")

        # Poids des subsets :
        match parametresCorrelation['poidsSubset'] : 
            case 'uniforme' : 
                MatchIDfile.write("<SubsetWeight>=<0>\n")

            case 'gaussien' : 
                MatchIDfile.write("<SubsetWeight>=<1>\n")

        # Remplissage partiel :
        if parametresCorrelation['remplissagePartiel']['condition'] is True : 
            MatchIDfile.write("<AllowPartialSubsets>=<True>\n")
            MatchIDfile.write("<SubsetPointTolerance>=<%i>\n" %parametresCorrelation['remplissagePartiel']['valeur'])

        else : 
            MatchIDfile.write("<AllowPartialSubsets>=<False>\n")


        # Type de mouchetis :
        match parametresCorrelation['typeMouchetis'] : 
            case "attache" : 
                MatchIDfile.write("<SpeckleType>=<0>\n")

            case "projectif" : 
                MatchIDfile.write("<SpeckleType>=<1>\n")
        
        # Interpolation niveau de gris : 
        match parametresCorrelation['interpolationGris'] : 
            case "bilineaire" : 
                MatchIDfile.write("<Interpolation>=<3>\n")

            case "bicubic" : 
                MatchIDfile.write("<Interpolation>=<4>\n")

            case "splineBicubicGlobal" : 
                MatchIDfile.write("<Interpolation>=<5>\n")

            case "splineBicubicLocal" : 
                MatchIDfile.write("<Interpolation>=<6>\n")


        match parametresCorrelation['fonctionFormeSubset'] : 
            case 'rigide' :
                MatchIDfile.write("<Transformation>=<0>\n")

            case 'affine' :
                MatchIDfile.write("<Transformation>=<1>\n")

            case 'irregulier' :
                MatchIDfile.write("<Transformation>=<2>\n")

            case 'quadratique' :
                MatchIDfile.write("<Transformation>=<3>\n")


        # Transformation stereo :
        match parametresCorrelation['transformationStereo'] : 
            case 'affine' : 
                MatchIDfile.write("<Stereo$Transformation>=<0>\n")

            case 'qaudratique' : 
                MatchIDfile.write("<Stereo$Transformation>=<1>\n")

            case 'homographique' : 
                MatchIDfile.write("<Stereo$Transformation>=<2>\n")


        # Critere precision :
        MatchIDfile.write("<Precision>=<%f>\n" %parametresCorrelation['criterePrecision'])

        # Nombre iterations maximal :
        MatchIDfile.write("<Iterations>=<%i>\n" %parametresCorrelation['nombreIterationsMaximum'])

        # Niveau de correlation :
        MatchIDfile.write("<Correlation$treshold>=<%f>\n" %parametresCorrelation['seuilConvergence'])

        # Compensation manque de donnees : 
        if parametresCorrelation['compensationManqueDonnees'] is True : 
            MatchIDfile.write("<Missing$data>=<True>\n")

        else : 
            MatchIDfile.write("<Missing$data>=<False>\n")

        # Estimation mouvement corps rigide :
        MatchIDfile.write("<Maximal$rigid>=<%i>\n" %parametresCorrelation['estimationMouvementRigide'])
        
        # Estimation niveau deformation : 
        match parametresCorrelation['estimationNiveauDeformation'] : 
            case 'petit' : 
                MatchIDfile.write("<Estimated$strain>=<small>\n")

            case 'moyen' : 
                MatchIDfile.write("<Estimated$strain>=<medium>\n")

            case 'grand' : 
                MatchIDfile.write("<Estimated$strain>=<large>\n")

        # Angle entre les cameras : 
        if parametresCorrelation['grandAngleCameras'] is True : 
            MatchIDfile.write("<Large$angles>=<True>\n")

        else : 
            MatchIDfile.write("<Large$angles>=<False>\n")

        # Progression : 
        match parametresCorrelation['progression']['critere'] : 
            case 'spatial' : 
                MatchIDfile.write("<History>=<1>\n")

            case 'spatialAndUpdateReference' : 
                MatchIDfile.write("<History>=<2>\n")
                if parametresCorrelation['progression']['incrementation'] == "excesSeuil" :
                    MatchIDfile.write("<IterationBasedIncremental>=<True>\n")
                    MatchIDfile.write("<IncrementIterationThreshold>=<%f>\n" %parametresCorrelation['progression']['seuilIteration'])

                else : 
                    MatchIDfile.write("<IterationBasedIncremental>=<False>\n")
                    MatchIDfile.write("<IncrementIterationThreshold>=<3.5>\n")

        # Traitement du bruit : 
        match parametresCorrelation['traitementBruit'] : 
            case "None" : 
                MatchIDfile.write("<Noise>=<0>\n")

            case "Moyenne" : 
                MatchIDfile.write("<Noise>=<1>\n")

            case "gaussien" : 
                MatchIDfile.write("<Noise>=<2>\n")
                MatchIDfile.write("<Kernel$size>=<%i>\n" %parametresCorrelation['tailleNoyau'])


        # Distance epipolaire : 
        MatchIDfile.write('<Epipolar$tolerance>=<%i>\n' %parametresCorrelation['distanceEpipolaire'])

        # Erreur epipolaire : 
        MatchIDfile.write('<Epipolar$error>=<%i>\n' %parametresCorrelation['erreurEpipolaire'])

        # Triangulation de l'optimisation : 
        if parametresCorrelation['triangulationOptimisation'] is True : 
            MatchIDfile.write("<Triangulation$optimization>=<True>\n")
        
        else :
            MatchIDfile.write("<Triangulation$optimization>=<False>\n")
        
        # Nombre de processeur : 
        MatchIDfile.write('<Processors>=<%i>\n' %parametresCorrelation['nombreProcesseur'])

        # Repere pour exprimer les resultats : 
        match parametresCorrelation['repereExpressionResultats'] : 
            case 'cameraCentrale' : 
                MatchIDfile.write("<Coordinate$transformation>=<0>")

            case 'meilleurPlanIdentifie' : 
                MatchIDfile.write("<Coordinate$transformation>=<1>")

            case "camera0" : 
                MatchIDfile.write("<Coordinate$transformation>=<2>")

            case "camera1" : 
                MatchIDfile.write("<Coordinate$transformation>=<3>")

        # Calcul automatique des deformations :
        if parametresCorrelation['calculDeformations']['condition'] is True : 
            MatchIDfile.write("<Automatic$strain>=<True>\n")
            MatchIDfile.write("<Strain$window>=<%i>\n" %parametresCorrelation['calculDeformations']['tailleFenetre'])
            MatchIDfile.write("<Point$tolerance>=<%i>\n" %parametresCorrelation['calculDeformations']['tolerancePoints'])
            match parametresCorrelation['calculDeformations']['conventionDeformation'] : 
                case 'GreenLagrange' : 
                    MatchIDfile.write("<Strain$convention>=<0>\n")
                    
                case 'LogEulerAlmansi' : 
                    MatchIDfile.write("<Strain$convention>=<1>\n")
                    
                case 'Euler' : 
                    MatchIDfile.write("<Strain$convention>=<2>\n")
                    
                case 'Hencky' : 
                    MatchIDfile.write("<Strain$convention>=<3>\n")
                    
                case 'BiotReference' : 
                    MatchIDfile.write("<Strain$convention>=<4>\n")

                case 'BiotCourant' : 
                    MatchIDfile.write("<Strain$convention>=<5>\n")

            match parametresCorrelation['calculDeformations']['degrePolynome'] : 
                case 'Q4' : 
                    MatchIDfile.write("<Strain$interpolation>=<0>\n")

                case 'Q8' : 
                    MatchIDfile.write("<Strain$interpolation>=<1>\n")

                case 'Q9' : 
                    MatchIDfile.write("<Strain$interpolation>=<2>\n")

        else : 
            MatchIDfile.write("<Automatic$strain>=<True>\n")
            MatchIDfile.write("<Strain$window>=<5>\n")
            MatchIDfile.write("<Point$tolerance>=<0>\n")
            MatchIDfile.write("<Strain$convention>=<1>\n")
            MatchIDfile.write("<Strain$interpolation>=<0>\n")

        # Export des resultats : 
        if parametresCorrelation['exportResultats']['condition'] is True : 
            MatchIDfile.write("<Automatic$export>=<True>\n")
            if parametresCorrelation['exportResultats']['formatExport'] == 'standard'  : 
                MatchIDfile.write("<Export$format>=<0>\n")

            else : 
                MatchIDfile.write("<Export$format>=<1>\n")

        else : 
            MatchIDfile.write("<Automatic$export>=<False>\n")

        # Dossier d'export :
        MatchIDfile.write("<Output$path>=<%s>\n" %os.path.join(dossierCourant, parametresCorrelation['exportResultats']['dossier'] ))

        # Stockage des donnees :
        match parametresCorrelation['exportResultats']['stockageDonnees'] : 
            case 'essentiel' : 
                MatchIDfile.write("<DataStorage>=<0>\n")

            case 'default' : 
                MatchIDfile.write("<DataStorage>=<1>\n")

            case 'etendue' : 
                MatchIDfile.write("<DataStorage>=<2>\n")

        # Delimiter :
        MatchIDfile.write("<Delimiter>=<%s>\n" %parametresCorrelation["exportResultats"]['delimiter'])

        # Sauvegarder les statistiques :
        if parametresCorrelation['exportResultats']['stockageStats'] is True : 
            MatchIDfile.write("<StoreStats>=<True>\n")

        else : 
            MatchIDfile.write("<StoreStats>=<False>\n")

        # Region d'interet :
        MatchIDfile.write("<Shape>=<%s%i;%i>\n" %(parametresCorrelation['ROI']['forme'],
                                                   *parametresCorrelation['ROI']['initialGuess'],
                                                   ),
                          )

        # Automask :
        MatchIDfile.write("<AutoMask>=<0;5;%s>\n" %os.path.join(dossierCourant,
                                                                parametresCorrelation['ROI']['automask']),
                          )

        
def creationDossierResultats(nomDossier: str) : 
    """ 
    """

    if not os.path.exists(nomDossier) : 
        os.makedirs(nomDossier)

    else : 
        if len(os.listdir(nomDossier)) != 0 : 
            print('pb')


parametresCorrelation = {"dossierCourant"              : True,
                         "nomDossier"                  : "FEDEF-0-0",
                         "prefixNomImage"              : "image_",
                         "suffixNomImage"              : "_0.tiff",
                         "numeroImageNonDeformee"      : 0 ,
                         "listeNumeroImageDeformee"    : np.arange(1, 11, 1, dtype = int),
                         "subsetSize"                  : 21,
                         "stepSize"                    : 5,
                         "critereCorrelation"          : "ZNSSD", 
                         "poidsSubset"                 : "uniforme",
                         "remplissagePartiel"          : {"condition" : True,
                                                          "valeur"    : 75,
                                                          },
                         "typeMouchetis"               : "attache",
                         "interpolationGris"           : "splineBicubicLocal",
                         "fonctionFormeSubset"         : "quadratique",
                         "transformationStereo"        : "quadratique",
                         "criterePrecision"            : 0.001,
                         "nombreIterationsMaximum"     : 1000,
                         "seuilConvergence"            : 0.9, 
                         "compensationManqueDonnees"   : False,
                         "estimationMouvementRigide"   : 100,
                         "estimationNiveauDeformation" : "small",
                         "grandAngleCameras"           : False,
                         "progression"                 : {"critere"        :"spatialAndUpdateReference",
                                                          "incrementation" : "excesSeuil",
                                                          "seuilIteration" : 3.5,
                                                          },
                         "traitementBruit"             : "gaussien",
                         "tailleNoyau"                 : 5,
                         "distanceEpipolaire"          : 5,
                         "erreurEpipolaire"            : 10,
                         "triangulationOptimisation"   : "False",
                         "nombreProcesseur"            : 24,
                         "repereExpressionResultats"   : "camera0",
                         "calculDeformations"          : {"condition"             : False,
                                                          "tailleFenetre"         : 5,
                                                          "tolerancePoints"       : 0,
                                                          "conventionDeformation" : "Hencky",
                                                          "degrePolynome"         : "Q4",
                                                          },
                         "exportResultats"             : {"condition"      : True,
                                                          "dossier"        : "DIC-0-0",
                                                          "formatExport"   : "standard",
                                                          "stockageDonnees" : "default",
                                                          "delimiter"      : ";",
                                                          "stockageStats"  : False,
                                                          },
                         "ROI"                         : {"forme"        : "0;0;False;0;0;1200;1200;",
                                                          "initialGuess" : [540, 575],
                                                          "automask"     : "image_0_0_mask.tiff",
                                                          },
                         } 


ecrireFichierInstructions("DIC-0-0", parametresCorrelation)
lancementCorrelation("MatchIDStereo.exe", "DIC-0-0.m3inp")

