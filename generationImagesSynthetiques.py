import numpy as np 
import os 


def verificationExistanceFichier(cheminFichier: str) : 
    """ 
        Fonction pour verifier que le fichier existe
        Met fin au programme si le fichier n'existe pas
    """
    if not os.path.exists(cheminFichier):
        print("Le fichier %s n'existe pas" %cheminFichier)
        print("Fin du programme")
        # quit()


def lectureFichierCalibrationMatchID(cheminFichier: str) : 
    """ 
        Fonction pour lire le fichier de calibration au format .caldat de MatchID
    """

    # Verification que le fichier existe : 
    verificationExistanceFichier(cheminFichier)

    dicoCalibration = {'c0_fx' : 'Cam0_Fx',
                       'c0_fy' : 'Cam0_Fy',
                       'c0_fs' : 'Cam0_Fs',
                       'c0_k1' : 'Cam0_Kappa1',
                       'c0_k2' : 'Cam0_Kappa2',
                       'c0_k3' : 'Cam0_Kappa3',
                       'c0_p1' : 'Cam0_P1',
                       'c0_p2' : 'Cam0_P2',
                       'c0_cx' : 'Cam0_Cx',
                       'c0_cy' : 'Cam0_Cy', 
                       # Camera 1 :
                       'c1_fx' : 'Cam1_Fy',
                       'c1_fy' : 'Cam1_Fy',
                       'c1_fs' : 'Cam1_Fs',
                       'c1_k1' : 'Cam1_Kappa1',
                       'c1_k2' : 'Cam1_Kappa2',
                       'c1_k3' : 'Cam1_Kappa3',
                       'c1_p1' : 'Cam1_P1',
                       'c1_p2' : 'Cam1_P2',
                       'c1_cx' : 'Cam1_Cx',
                       'c1_cy' : 'Cam1_Cy', 
                       # Parametres extrinseques :
                       'tx'    : 'Tx',
                       'ty'    : 'Ty',
                       'tz'    : 'Tz',
                       'theta' : 'Theta',
                       'phi'   : 'Phi',
                       'psi'   : 'Psi',
                       }

    with open(cheminFichier, 'r') as calibFile  :
        lines = calibFile.readlines()
        for key in dicoCalibration.keys()  : 
            i = 0
            cond = False
            while cond == False and i < len(lines) :
                if lines[i].split(';')[0].replace(' ', '').replace('[pixels]', '').replace('[deg]', '') == dicoCalibration[key] : 

                    dicoCalibration.update({key :  float(lines[i].split(';')[1].replace('\n', '').replace(' ', ''))})
                    cond = True

                else : 
                    i += 1

    return dicoCalibration


def ecrireFichierInstructions(nomFichier: str, dictionnaireInstructions: dict) : 
    """
    """ 

    with open('%s.mtind' %nomFichier, 'w') as fedefFile : 
        fedefFile.write("<application>=<Stereo>\n")

        # Image de reference : motif initial
        verificationExistanceFichier(dictionnaireInstructions['imageReference'])
        fedefFile.write("<Reference$image>=<%s>\n" %dictionnaireInstructions['imageReference'])

        # Fichier contenant le maillage :
        verificationExistanceFichier(dictionnaireInstructions['nomFichierMaillage'])
        fedefFile.write("<Mesh$file>=<%s>\n" %dictionnaireInstructions['nomFichierMaillage'])

        # Unite du maillage :
        match dictionnaireInstructions['nomFichierMaillage'] : 
            case 'mm' : 
                fedefFile.write("<Mesh$unit>=<mm>\n")

            case 'm' : 
                fedefFile.write("<Mesh$unit>=<m>\n")

            case 'inch' : 
                fedefFile.write("<Mesh$unit>=<inch>\n")

            case _ : 
                print("Mauvaise unité renseignee")
                # quit()

        # Orientation du maillage par rapport a la camera 0 :
        if len(dictionnaireInstructions['orientationMaillage']) != 6 :
            print("Le nombre de valeurs pour indiquer l'orientation du maillage par rapport à la camera 0 n'est pas egal à 6")
            print("Fin du programme")
            quit()

        fedefFile.write("<Mesh$position>=<{:f};{:f};{:f};{:f};{:f};{:f}>\n".format(*dictionnaireInstructions['orientationMaillage']))

        # Fichier de deplacements nodaux :
        chaineCaractere =  "<Nodal$file>=<" 
        for fichier in dictionnaireInstructions['listeFichiersDeplacementsNodaux'] :
            verificationExistanceFichier(fichier)
            chaineCaractere += fichier  + ";"

        chaineCaractere = chaineCaractere[:-1] + ">\n"
        fedefFile.write(chaineCaractere)

        # Fichier de calibration des cameras : 
        dicoCalibration = lectureFichierCalibrationMatchID(dictionnaireInstructions["nomFichierCalibration"])

        fedefFile.write("<Camera$1$calibrationfile>=<%s>\n" %dictionnaireInstructions["nomFichierCalibration"])

        # Parametres intrinseques camera 0 :
        fedefFile.write("<Camera$0$intrinsic>=<%f;%f;%f;%f;%f;%f;%f;%f;%i;%i;0;0>\n" %(dicoCalibration['c0_fx'],
                                                                                     dicoCalibration['c0_fy'],
                                                                                     dicoCalibration['c0_fs'],
                                                                                     dicoCalibration['c0_k1'],
                                                                                     dicoCalibration['c0_k2'],
                                                                                     dicoCalibration['c0_k3'],
                                                                                     dicoCalibration['c0_p1'],
                                                                                     dicoCalibration['c0_p2'],
                                                                                     dicoCalibration['c0_cx'],
                                                                                     dicoCalibration['c0_cy'],
                                                                                     )
                        )

        # Bruit camera 0 : 
        match dictionnaireInstructions['bruitCamera']['camera1'] :
            case None :
                fedefFile.write("<Camera$0$noise>=<0>\n")

        # Parametres intrinseques camera 1 :
        fedefFile.write("<Camera$1$intrinsic>=<%f;%f;%f;%f;%f;%f;%f;%f;%i;%i;0;0>\n" %(dicoCalibration['c1_fx'],
                                                                                     dicoCalibration['c1_fy'],
                                                                                     dicoCalibration['c1_fs'],
                                                                                     dicoCalibration['c1_k1'],
                                                                                     dicoCalibration['c1_k2'],
                                                                                     dicoCalibration['c1_k3'],
                                                                                     dicoCalibration['c1_p1'],
                                                                                     dicoCalibration['c1_p2'],
                                                                                     dicoCalibration['c1_cx'],
                                                                                     dicoCalibration['c1_cy'],
                                                                                     )
                        )

        # Bruit camera 1 : 
        match dictionnaireInstructions['bruitCamera']['camera1'] :
            case None :
                fedefFile.write("<Camera$1$noise>=<0>\n")


        # Parametres extrinseques :
        fedefFile.write("<Camera$1$extrinsic>=<%f;%f;%f;%f;%f;%f>\n" %(dicoCalibration['tx'],
                                                                     dicoCalibration['ty'],
                                                                     dicoCalibration['tz'],
                                                                     dicoCalibration['theta'],
                                                                     dicoCalibration['phi'],
                                                                     dicoCalibration['psi'],
                                                                     )
                        )
        
        # Parametres de sauvegarde :
        fedefFile.write("<Base$filename>=<%s>\n" %dictionnaireInstructions['prefixFichier'])
        fedefFile.write("<Output$folder>=<%s>\n" %dictionnaireInstructions['dossierSauvegardeResultats'])
        
        if dictionnaireInstructions['sauvegardeFichierMappingElements'] is True : 
            fedefFile.write("<Save$Mapping$File>=<True>\n")

        else : 
            fedefFile.write("<Save$Mapping$File>=<False>\n")

        if dictionnaireInstructions['sauvegardeFichierMappingNoeuds'] is True : 
            fedefFile.write("<Save$Nodal$Mapping$File>=<True>\n")

        else : 
            fedefFile.write("<Save$Nodal$Mapping$File>=<False>\n")

        if dictionnaireInstructions['sauvegardeFichierMasquage'] is True : 
            fedefFile.write("<Save$Masking$File>=<True>\n")

        else : 
            fedefFile.write("<Save$Masking$File>=<False>\n")

        if dictionnaireInstructions['sauvegardeFichierDef'] is True : 
            fedefFile.write("<Save$As$def>=<True>\n")

        else : 
            fedefFile.write("<Save$As$def>=<False>\n")

        if dictionnaireInstructions['sauvegardeImageSynthetique'] is True : 
            fedefFile.write("<Save$As$Synthetic>=<True>\n")

        else : 
            fedefFile.write("<Save$As$Synthetic>=<False>\n")

        if dictionnaireInstructions['sauvegardeImageTiff'] is True : 
            fedefFile.write("<Save$As$tif>=<True>\n")

        else : 
            fedefFile.write("<Save$As$tif>=<False>\n")

        if dictionnaireInstructions['sauvegardeCsv'] is True : 
            fedefFile.write("<Save$As$csv>=<True>\n")

        else : 
            fedefFile.write("<Save$As$csv>=<False>\n")

        if dictionnaireInstructions['sauvegardeDat'] is True : 
            fedefFile.write("<Save$As$dat>=<True>\n")

        else : 
            fedefFile.write("<Save$As$dat>=<False;5>\n")

        # Interpolation niveau de gris : 
        match dictionnaireInstructions['interpolationNiveauGris'] : 
            case "biCubic" : 
                fedefFile.write("<Interpolation>=<0>\n")
        
            case "splineBiCubic" : 
                fedefFile.write("<Interpolation>=<1>\n")

        # Niveau de gris du fond : 
        fedefFile.write("<Background$color>=<%i>\n" %dictionnaireInstructions['niveauGrisFond'])

        # Conservation du fond initial de la camera 0 : 
        if dictionnaireInstructions['conserverFondOriginalImageCamera0'] is True : 
            fedefFile.write("<KeepOriginalBack$Reference$camera0>=<True>\n")

        else : 
            fedefFile.write("<KeepOriginalBack$Reference$camera0>=<False>\n")

        # Options de rendu : 
        # Nombre de processeurs : 
        fedefFile.write("<Render$number$processors>=<%i>" %dictionnaireInstructions['nombreProcesseur'])

        # Variation de lumiere
        if dictionnaireInstructions['variationLumiere'] is True : 
            fedefFile.write("<Render$light$changes>=<True>\n")

        else :
            fedefFile.write("<Render$light$changes>=<False>\n")

        # Integration des pixels : 
        if dictionnaireInstructions['integrationPixel'] is True : 
            fedefFile.write("<Render$pixel$integration>=<True>\n")

        else :
            fedefFile.write("<Render$pixel$integration>=<False>\n")

        # Profondeur de champs :
        if dictionnaireInstructions['profondeurChamps'] is True : 
            fedefFile.write("<Render$depth$of$field>=<True>\n")

        else : 
            fedefFile.write("<Render$depth$of$field>=<False>\n")

listeFichiersDeplacementsNodaux = []
for i in range(1, 11, 1) : 
    listeFichiersDeplacementsNodaux.append('./Deplacements/FEDEF-nodesDisplacement-DIC-frame-increment-%i.csv' %i)

dictionnaireInstructions = {"imageReference"                    : "./imageReference.bmp",
                            "nomFichierMaillage"                : "./Deplacements/Notched-Validation.mesh",
                            "uniteMaillage"                     :  "mm",
                            "orientationMaillage"               : [0, 0, 200.4, 0, 0, 0],
                            "listeFichiersDeplacementsNodaux"   : listeFichiersDeplacementsNodaux,
                            "nomFichierCalibration"             : 'calibration.caldat',
                            "bruitCamera"                       : {'camera0' : None, 
                                                                   'camera1' : None,
                                                                  },
                            "prefixFichier"                     : 'imageSynthetique',
                            "dossierSauvegardeResultats"        : 'FEDEF-0-0',
                            "sauvegardeFichierMappingElements"  : False,
                            "sauvegardeFichierMappingNoeuds"    : False,
                            "sauvegardeFichierMasquage"         : True,
                            "sauvegardeFichierDef"              : False,
                            "sauvegardeImageSynthetique"        : False,
                            "sauvegardeImageTiff"               : True,
                            "sauvegardeCsv"                     : False,
                            "sauvegardeDat"                     : False,
                            "interpolationNiveauGris"           : "splineBiCubic",
                            "niveauGrisFond"                    : 117,
                            "conserverFondOriginalImageCamera0" : False,
                            "nombreProcesseur"                  : 23,
                            "integrationPixel"                  : True,
                            "variationLumiere"                  : False,
                            "profondeurChamps"                  : False,
                            }


ecrireFichierInstructions("testFEDEF", dictionnaireInstructions)
