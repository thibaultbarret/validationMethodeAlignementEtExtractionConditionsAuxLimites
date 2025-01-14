# Etude paramétrique sur l'influence du désalignement entre la caméra et l'éprouvette 

## Méthodologie
1. Simulation EF avec un géométrie de référence et des conditions aux limites hétérogènes en vue d'obtenir des champs cinématiques hétérogènes 
    - Maillage généré avec `Cast3M` 
    - Loi de comportement elastoplastique généré avec `MFront`
    - Génération des conditions aux limites avec le script `generationConditionsAuxLimites.py`

2. Génération d'images synthétiques avec le module `FEDEF` de `MatchID 2025.1.2`
    - Image de référence `imageReference.bmp` et du fichier de calibration `calibration.caldat`
    - Mise en forme du maillage en utilisant le script `generationConditionsAuxLimites.py` : détermination de table de connectivité 2D des noeuds en surface
    - Utilisation d'un script pour générer l'ensemble des configurations étudiées : `generationImagesSynthetiques.py`

3. Application de la corrélation d'image avec le module `Stereo` de `MatchID 2025.1.2`
    - Utilisation des fichiers `_mask.tiff` pour selectionner les pixels qui ont été modifés par `MatchID`
    - Export des résultats dans le repère de la Caméra\_0 et dans le meilleur plan
    - Utilisation d'un script pour traiter l'ensemble des cas

## Simulation numérique de référence

## Génération des images 
