import pandas as pd 
import numpy as np 
import os
from copy import deepcopy
from skspatial.objects import Plane

# Scipy : 
from scipy.interpolate import griddata

# Plotly : 
from plotly import graph_objects as go 

# Matplotlib : 
import matplotlib.pyplot as plt
import matplotlib.tri as mtri

from skimage import io

# Custom class : 
from classPlotlyObjects import coordinateSystemPlotly

class dicData():
    """
    """

    #--------------------------------------------------------------------------------------------------------
    #                                        Initialisation : 
    #--------------------------------------------------------------------------------------------------------
    def __init__(self, dictDicData: dict) : 
        """
        """

        self.centimetre = 1./2.54

        #================================================================================
        #                         Initialisation : 
        #================================================================================
        self.boolDataReduced                    = False
        self.boolBestPlaneFitted                = False 
        self.boolTriangulationComputed          = False
        self.boolFullTriangulationComputed      = True
        self.boolAlphaBetaAnglesComputed        = False


        # Bool for projection :
        self.boolFiniteElementBorderProjected               = False
        self.boolFiniteElementBorderProjectedInTheBestPlane = False 
        self.boolSubsetCoordsProjected                      = False

        # Bool for plolty traces : 
        self.boolTraceBestPlaneSurface                                              = False
        self.boolTraceSubsetMesh3D                                                  = False

        self.boolTraceSubsetCoordinatesUsedForDisplacementsExtraction               = False
        self.boolTraceSubsetCoordinatesUsedForDisplacementsExtractionInTheBestPlane = False

        self.boolTraceFiniteElementBorderProjected                                  = False
        self.boolTraceFiniteElementBorderProjectedInTheBestPlane                    = False
        

        # Dict to store results per stage : 
        self.displacementsData                                           = {}
        self.tracesSubsetDisplacement                                    = {}
        self.displacementsDataExpressProjectedInThePlane                 = {}
        self.displacementsDataExpressProjectedInThePlaneExpressInFeFrame = {}


        self.strXname_DIC_ref                   = 'Xref[mm]'
        self.strYname_DIC_ref                   = 'Yref[mm]'
        self.strZname_DIC_ref                   = 'Zref[mm]'

        #================================================================================
        # 1. Load dic data : 
        #================================================================================
        dicDataPath = os.path.join(dictDicData['strPathDicData'], dictDicData['strFileName'])

        if not os.path.exists(dicDataPath) : 
            self.leftProgramme("File %s doesn't exist" %dicDataPath)

        try : 
            self.dicData = pd.read_csv(dicDataPath,
                                      header = 0,
                                      delimiter = dictDicData['strFieldsDelimiter'],
                                      usecols = [dictDicData['strXnamePixel'],
                                                 dictDicData['strYnamePixel'],
                                                 dictDicData['strUnamePixel'],
                                                 dictDicData['strVnamePixel'],
                                                 dictDicData['strXname'], 
                                                 dictDicData['strYname'],
                                                 dictDicData['strZname'],
                                                 dictDicData['strUname'],
                                                 dictDicData['strVname'],
                                                 dictDicData['strWname'],
                                                 ]
                                       )
         
        except  Exception as error:
            print("Error :", error) 
            self.leftProgramme("Error during file : %s openning" %dicDataPath)
             

        self.strXnamePixel_DIC = dictDicData['strXnamePixel'] 
        self.strYnamePixel_DIC = dictDicData['strYnamePixel'] 

        self.strUnamePixel_DIC = dictDicData['strUnamePixel'] 
        self.strVnamePixel_DIC = dictDicData['strVnamePixel'] 

        self.strXname_DIC = dictDicData['strXname'] 
        self.strYname_DIC = dictDicData['strYname'] 
        self.strZname_DIC = dictDicData['strZname'] 
         
        self.strUname_DIC = dictDicData['strUname'] 
        self.strVname_DIC = dictDicData['strVname'] 
        self.strWname_DIC = dictDicData['strWname'] 

        self.intStepSize = dictDicData['intStepSize']


        # Sort data regarding x_pic and y_pic : 
        self.dicData = self.dicData.dropna(axis = 0)
        self.dicData = self.dicData.sort_values(by = [self.strXnamePixel_DIC, self.strYnamePixel_DIC], ascending = [True, True])
        self.dicData = self.dicData.reset_index(drop = True)


    #--------------------------------------------------------------------------------------------------------
    #                                        Load displacements data : 
    #--------------------------------------------------------------------------------------------------------
    def loadDisplacementResults(self, dictDisplacementFileOpenning: dict) : 
        """
        """

        # 1. Check if the folder exists : 
        if not os.path.exists(dictDisplacementFileOpenning['strFolderPath']) : 
            self.leftProgramme("The folder '%s' wasn't found")


        if isinstance(dictDisplacementFileOpenning['loadStep'], int) :
            loadStep = [dictDisplacementFileOpenning['loadStep']]

        elif isinstance(dictDisplacementFileOpenning['loadStep'], list) :
            loadStep = dictDisplacementFileOpenning['loadStep']

        elif isinstance(dictDisplacementFileOpenning['loadStep'], np.ndarray) :
            loadStep = dictDisplacementFileOpenning['loadStep']

        else : 
            print("Error, fileNumber passed to supported")
            return None

        self.firstTimeStep = min(loadStep)

        #2. Loop over loadStep :
        for number in loadStep : 
            strFileName = dictDisplacementFileOpenning['strFilePrefix'] + str(int(number)).zfill(dictDisplacementFileOpenning['intZeroNumber']) + dictDisplacementFileOpenning['strFileSuffix']
            dicDataPath = os.path.join(dictDisplacementFileOpenning['strFolderPath'], strFileName)

            if not os.path.exists(dicDataPath) : 
                self.leftProgramme("File %s doesn't exist" %dicDataPath)

            try : 
                data = pd.read_csv(dicDataPath,
                                   header    = 0,
                                   delimiter =  dictDisplacementFileOpenning['strFieldsDelimiter'],
                                   usecols   = [dictDisplacementFileOpenning['strXnamePixel'],
                                                dictDisplacementFileOpenning['strYnamePixel'],
                                                # dictDisplacementFileOpenning['strUnamePixel'],
                                                # dictDisplacementFileOpenning['strVnamePixel'],
                                                dictDisplacementFileOpenning['strXname'], 
                                                dictDisplacementFileOpenning['strYname'],
                                                dictDisplacementFileOpenning['strZname'],
                                                dictDisplacementFileOpenning['strUname'],
                                                dictDisplacementFileOpenning['strVname'],
                                                dictDisplacementFileOpenning['strWname'],
                                                ],
                                   )

            except  Exception as error:
                print("Error :", error) 
                self.leftProgramme("Error during file : %s openning" %dicDataPath)
            
            # Compute reference coordinates : 
            # // NOTE: X = x - U
            data[self.strXname_DIC_ref] = data[self.strXname_DIC] - data[self.strUname_DIC]
            data[self.strYname_DIC_ref] = data[self.strYname_DIC] - data[self.strVname_DIC]
            data[self.strZname_DIC_ref] = data[self.strZname_DIC] - data[self.strWname_DIC]

            self.displacementsData.update({'%i' %number : data} )

            del data


    def loadStepTimeFile(self, fileName: str, fileType: str = 'MatchID', extractFileNumber: dict = None) : 
        """ 
            Method to load the file from MatchID Grabber to associate load step and time
        """

        if not os.path.exists(fileName) : 
            self.leftProgramme("The file %s wasn't found")

        if fileType == 'MatchID' : 
            try : 
                self.stepTimeFile = pd.read_csv(fileName,
                                                header = 0,
                                                delimiter = ';')

            except  Exception as error:
                print("Error :", error) 
                self.leftProgramme("Error during file : %s openning" %fileName)
        
        else : 
            self.leftProgramme("Other file type than MatchID are supported yet")

        if extractFileNumber is not None: 
            print("• File name extraction: ")
            try : 
                index = self.stepTimeFile['File'].iloc[0].find(extractFileNumber['triggerString'])
                print("\t• Application to the first row: %s" %(self.stepTimeFile['File'].iloc[0][index: -1]))

                listTemp = []
                listTempTwo = []
                for i in range(len(self.stepTimeFile)) :
                    listTemp.append(self.stepTimeFile['File'].iloc[i][index: -1])
                    listTempTwo.append(int(listTemp[-1].replace(extractFileNumber['preNumber'], '').replace(extractFileNumber['postNumber'], '')))

                self.stepTimeFile['fileName'] = listTemp
                self.stepTimeFile['loadStep'] = listTempTwo

            except  Exception as error:
                print("Error :", error) 
                self.leftProgramme("Error during file : %s openning" %fileName)
        
        print()


    def loadStepTimeFromArray(self, timeStep: list | np.ndarray) : 
        """
            Simple method to load time step corresponding to displacement results loaded
            timeStep must containts : [[fileNumber1, timeStep1], 
                                       [fileNumber2, timeStep2],
                                                ... 
                                       [fileNumberN, timeStepN]]
                                                            
        """

        if not isinstance(timeStep, (list, np.ndarray)) : 
            self.leftProgramme("The timeStep arg isn't a list or array")

        match timeStep : 
            case list() :
                if len(timeStep[0]) != 2 :
                    self.leftProgramme("The timeStep hasn't two columns")

                else : 
                    self.listCorrespondingTimeOfDisplacementsInterpolated = np.array(timeStep)

            case np.ndarray() : 
                if timeStep.shape[1] != 2 : 
                    self.leftProgramme("The timeStep hasn't two columns")

                else : 
                    self.listCorrespondingTimeOfDisplacementsInterpolated = timeStep


    #--------------------------------------------------------------------------------------------------------
    #                                           Data reduction : 
    #--------------------------------------------------------------------------------------------------------
    def reduceNumberOfDataPoints(self, divider: int) :
        """
            Method to reduce the ammount of points contains at initial state

            divider : Keep one point over divider

            Create a copy of dicData
        """

        print('• Data reduction:')
        print('\tInitial number of points: %i' %len(self.dicData) )

        self.boolDataReduced = True
        self.intDividerToReduceData = divider

        self.dicDataRaw = self.dicData.copy()

        xPixelUnique = np.sort(self.dicData[self.strXnamePixel_DIC].unique())
        yPixelUnique = np.sort(self.dicData[self.strYnamePixel_DIC].unique())

        xPixelUniqueToKeep = []
        yPixelUniqueToKeep = []

        for i in range(0, len(xPixelUnique), divider) : 
            xPixelUniqueToKeep.append(xPixelUnique[i])

        for i in range(0, len(yPixelUnique), divider) : 
            yPixelUniqueToKeep.append(yPixelUnique[i])


        self.dicData = self.dicData[(self.dicData[self.strXnamePixel_DIC].isin(xPixelUniqueToKeep)) & (self.dicData[self.strYnamePixel_DIC].isin(yPixelUniqueToKeep))]
        self.dicData = self.dicData.reset_index(drop = True)

        print('\tNumber of points after reduction: %i' %len(self.dicData) )
        print('') 

    
    #--------------------------------------------------------------------------------------------------------
    #                                        Geometrical operations : 
    #--------------------------------------------------------------------------------------------------------
    def fitBestPlane(self) :
        """
            Method to compute the best plane

            $\vec{z}$ is the normal vector
            $\vec{x}$ is obtain by the projection of \vec{x}_{global} in the plane
            $\vec{y}$ is obtain by the cross product of \vec{n} ⊗ \vec{t}
        """

        # Put X|Y|Z in np.ndarray : 
        dataPoints = self.dicData[[self.strXname_DIC, self.strYname_DIC, self.strZname_DIC]].to_numpy()

        # Determine the plane : 
        self.bestPlane = Plane.best_fit(dataPoints)
        self.fitBestPlaneCartesianEquation = self.bestPlane.cartesian()
        self.centralPointBestPlane = self.bestPlane.point

        self.zVectorBestPlane = self.bestPlane.normal

        # Project the global xVector : 
        self.xVectorBestPlane = self.bestPlane.project_vector([1., 0., 0.])
        self.xVectorBestPlane = self.xVectorBestPlane/np.linalg.norm(self.xVectorBestPlane)

        # Determine the yVector : 
        self.yVectorBestPlane = np.cross(self.zVectorBestPlane, self.xVectorBestPlane)

        # Assemble the passage matrix : 
        self.passageMatrix = np.array([self.xVectorBestPlane, self.yVectorBestPlane, self.zVectorBestPlane]).T

        self.boolBestPlaneFitted = True
    

    def computeAlphaBetaAngles(self) : 
        """
            Method to determine the angle alpha and beta
        """
        
        print("• Alpha/beta computation: ")
        if self.boolBestPlaneFitted is False : 
            print("The best plane wasn't fitted")
            return None

        # // NOTE: This method seems to work :
        self.beta  = np.arcsin(self.zVectorBestPlane[0])
        self.alpha = np.arcsin(-(self.zVectorBestPlane[1]/
                                 np.cos(self.beta))
                               )

        print("\t• n_z vector   :", self.zVectorBestPlane[2])
        print("\t• n_z computed :", np.cos(self.alpha)*np.cos(self.beta) )
        print('') 

        self.boolAlphaBetaAnglesComputed = True

    
    def computeLocalFrame(self, gamma: float) : 
        """
            Method to compute the coordinate frame attached to the bestPlane
            This local frame will be used to express the displacements    
            gamma : rotation angle arround z in degrees 
        """

        if self.boolBestPlaneFitted is False : 
            print("The best plane must be fit to compute the local frame")
            return None

        # 1. rotate the xVector : 
        if self.boolAlphaBetaAnglesComputed is False : 
            print("alpha and beta angle must be fit to compute the local coordinate frame")
            return None
        
        rotationMatrix = self.computeRotationMatrix(np.radians(gamma))

        self.xVectorBestPlaneRotated = np.matmul(rotationMatrix, np.array([1., 0., 0.]).T)

        self.yVectorBestPlaneRotated = np.matmul(rotationMatrix, np.array([0., 1., 0.]).T)
        
        self.zVectorBestPlaneRotated = np.cross(self.xVectorBestPlaneRotated, self.yVectorBestPlaneRotated)

        self.passageMatrixToExpressInTheBestPlane = np.array([self.xVectorBestPlaneRotated, self.yVectorBestPlaneRotated, self.zVectorBestPlaneRotated]).T

    
    #---------------------------------------------------------------------------------------------------------
    #                                              FE projection :
    #---------------------------------------------------------------------------------------------------------
    def projectFiniteElementMesh(self, finiteElementMesh: list, xShift: float, yShift: float, gamma: float) : 
        """
            Method to project FE mesh using alpha & beta computed

        """
        if self.boolAlphaBetaAnglesComputed is False : 
            print("The angles alpha and beta were not computed")
            return None
        
        # Check shape : 
        for i, part in enumerate(finiteElementMesh) : 
            if part.shape[1] != 3 : 
                print("Object n°%i in finitElementMesh part hasn't the right dimension" %i)
                return None

        # Compute the rotation matrix :
        rotationMatrix = self.computeRotationMatrix(np.radians(gamma))

        # Determine the zShift =
        a, b, c, d = self.fitBestPlaneCartesianEquation

        zShift = (-a*xShift -b*yShift - d)/c


        finiteElementOutline = []

        # Apply rotation and shift : 
        for part in finiteElementMesh : 
            partRotatedShift        = np.matmul(part, rotationMatrix.T)
            partRotatedShift[:, 0] += xShift
            partRotatedShift[:, 1] += yShift
            partRotatedShift[:, 2] += zShift

            finiteElementOutline.append(partRotatedShift)

        self.finiteElementMeshProjected = finiteElementOutline

    
    def projectFiniteElementBorder(self, finiteElementMesh: list, xShift: float, yShift: float, gamma: float) : 
        """
            Method to project finite element border to do the interpolation of displacement
            The FE data must be expressed in the same coordinate frame as Cam0
        """

        if self.boolAlphaBetaAnglesComputed is False : 
            print("The angles alpha and beta were not computed")
            return None
        
        # Check shape : 
        for i, part in enumerate(finiteElementMesh) : 
            if part.shape[1] != 3 : 
                print("Object n°%i in finitElementMesh part hasn't the right dimension" %i)
                return None

        # Compute the rotation matrix :
        rotationMatrix = self.computeRotationMatrix(np.radians(gamma))

        # Determine the zShift =
        a, b, c, d = self.fitBestPlaneCartesianEquation

        zShift = (-a*xShift -b*yShift - d)/c


        self.listFiniteElementBorderProjected = []

        # Apply rotation and shift : 
        for part in finiteElementMesh : 
            nodes = []
            for node in part : 
                nodes.append(np.matmul(rotationMatrix, node.T))

            partRotatedShift        = np.array(nodes)
            partRotatedShift[:, 0] += xShift
            partRotatedShift[:, 1] += yShift
            partRotatedShift[:, 2] += zShift

            self.listFiniteElementBorderProjected.append(partRotatedShift)
        
        self.boolFiniteElementBorderProjected = True


    def projectFiniteElementMeshBorderInTheBestPlane(self)  :
        """
            Method to project in the best plane coordinate frame
            This displacements can be used to compute the virtual extensometer or to be impose in FE simulation
        """

        if self.boolFiniteElementBorderProjected is False : 
            print("The finite element mesh border weren't projected, please to it using projectFiniteElementBorder method")
            return None

        self.listFiniteElementBorderProjectedInTheBestPlane = []
        inverseMatricePassage = np.linalg.inv(self.passageMatrixToExpressInTheBestPlane)

        for part in self.listFiniteElementBorderProjected : 
            nodes = []
            for node in part :
                nodes.append(np.matmul(inverseMatricePassage, node.T))

            self.listFiniteElementBorderProjectedInTheBestPlane.append(np.array(nodes))

        self.boolFiniteElementBorderProjectedInTheBestPlane = True


    #---------------------------------------------------------------------------------------------------------
    #                                            Subset projection :
    #---------------------------------------------------------------------------------------------------------
    def projectSubsetCoordinatesInThePlane(self) : 
        """
            Method to project the subset coordinates in the best plane fitted
            The results are still expressed in the global coordinates frame
        """

        if self.boolBestPlaneFitted is False : 
            print("The best plane wasn't fitted")
            return None

        subsetCoords = self.dicData[[self.strXname_DIC, self.strYname_DIC, self.strZname_DIC]].to_numpy()
        listTemp = []

        for point in subsetCoords : 
            listTemp.append(self.bestPlane.project_point(point))

        self.subsetCoordsProjected = np.array(listTemp)

        self.boolSubsetCoordsProjected = True


    def projectSubsetDisplacementsInThePlane(self, intStep: int) : 
        """
            Method to project the displacements computed by the DIC in the best plane fitted
            Express in the global coordinates frame
        """

        if self.boolBestPlaneFitted is False : 
            print("The best plane wasn't fitted")
            return None

        if str(intStep) not in self.displacementsData.keys() : 
            print("The load step %i wasn't load" %intStep)
            return None

        displacements = self.displacementsData['%i' %intStep][[self.strUname_DIC, self.strVname_DIC, self.strWname_DIC]].to_numpy()
        

    def projectSubsetCoordinatesAndDisplacementsInTheBestPlane(self, intStep: int)  :
        """
            Method to express subset coordinates and displacements in the bestPlane
            1. Project subset coordinates in the best plane and then express in the bestPlaneCoordinateSystem
            2. Project subset displacements in the best plane and then express in the bestPlaneCoordinateSystem
            3. Add in the dictionnary displacementsDataExpressProjectedInThePlaneExpressInFeFrame
        """
        
        if self.boolBestPlaneFitted is False : 
            print("The best plane wasn't fitted")
            return None

        if str(intStep) not in self.displacementsData.keys() : 
            print("The load step %i wasn't load" %intStep)
            return None

        # 1. Project subset coordinates in the best plane fitted : 
        data = self.displacementsData['%i' %intStep]

        subsetCoord = data[[self.strXname_DIC_ref, self.strYname_DIC_ref, self.strZname_DIC_ref]].to_numpy()
        
        inverseMatricePassage = np.linalg.inv(self.passageMatrixToExpressInTheBestPlane)
        listTemp = []
        for subset in subsetCoord : 
            subset = self.bestPlane.project_point(subset)
            subset = np.matmul(inverseMatricePassage, subset.T)
            
            listTemp.append(subset)

        npArraySubsetCoords = np.array(listTemp)

        subsetDisplacement = data[[self.strUname_DIC, self.strVname_DIC, self.strWname_DIC]].to_numpy()

        listTemp = []
        for subset in subsetDisplacement : 
            subset = self.bestPlane.project_vector(subset)
            subset = np.matmul(inverseMatricePassage, subset.T)
            listTemp.append(subset)

        npArraySubsetDisplacements = np.array(listTemp)
        
        self.displacementsDataExpressProjectedInThePlaneExpressInFeFrame.update({'%i' %intStep : [npArraySubsetCoords, npArraySubsetDisplacements]})


    #--------------------------------------------------------------------------------------------------------
    #                                           Triangulation methods : 
    #--------------------------------------------------------------------------------------------------------
    def computeTriangulation(self) :
        """
            Method to compute the triangulation 
        """

        if self.boolDataReduced is True : 
            threeshold = self.intDividerToReduceData * self.intStepSize

        else : 
            threeshold = self.intStepSize

        xCoord = self.dicData[self.strXnamePixel_DIC].to_numpy()
        yCoord = self.dicData[self.strYnamePixel_DIC].to_numpy()

        triangulation = mtri.Triangulation(xCoord, yCoord)
        triangles = triangulation.triangles

        xtri = xCoord[triangles]  - np.roll(xCoord[triangles], 1, axis = 1)
        ytri = yCoord[triangles]  - np.roll(yCoord[triangles], 1, axis = 1)

        maxi = np.max(np.sqrt(xtri**2 + ytri**2), axis = 1)
        triangulation.set_mask(maxi > np.sqrt(2.*threeshold**2))

        self.triangulation = triangulation
        del triangulation

        self.boolTriangulationComputed = True


    def computeFullTriangulation(self, trianglesThreeshold: float = 1) : 
        """
            Method to compute the full triangulation of data in case of the dicData were reduced
        """

        # Check if the data were reduced :
        if self.boolDataReduced is False : 
            print("Data were not reduced, compute standard triangulation")
            self.computeTriangulation()
            self.boolFullTriangulationComputed = True
            return None

        # Compute full triangulation :
        xCoord = self.dicDataRaw[self.strXnamePixel_DIC].to_numpy()
        yCoord = self.dicDataRaw[self.strYnamePixel_DIC].to_numpy()

        triangulation = mtri.Triangulation(xCoord, yCoord)
        triangles = triangulation.triangles

        xtri = xCoord[triangles]  - np.roll(xCoord[triangles], 1, axis = 1)
        ytri = yCoord[triangles]  - np.roll(yCoord[triangles], 1, axis = 1)

        maxi = np.max(np.sqrt(xtri**2 + ytri**2), axis = 1)
        triangulation.set_mask(maxi > trianglesThreeshold * np.sqrt(2.*self.intStepSize**2))

        self.fullTriangulation = triangulation
        self.boolFullTriangulationComputed = True

        del triangulation

    
    #--------------------------------------------------------------------------------------------------------
    #                                           Interpolation : 
    #--------------------------------------------------------------------------------------------------------
    def displacementsInterpolationAtFiniteElementNodes(self, loadStep: int = None) : 
        """
            This method is used to inteprolate DIC displacements express in the best plane fit at finite element nodes
        """
        
        print('• Displacements interpolation at finite element nodes:')

        if loadStep is None : 
            print('\t• All loadStep will be considered')
        
        
        strLoadStep = ''
        listLoadStep = [] 
        for key in self.displacementsDataExpressProjectedInThePlaneExpressInFeFrame.keys() :  
            listLoadStep.append(int(key))

        listLoadStep.sort()
        lenListLoadStep = len(listLoadStep)

        for i, num in enumerate(listLoadStep, start = 1) : 
            if i != lenListLoadStep :
                strLoadStep += '%i, ' %num
            else : 
                strLoadStep += '%i' %num

        print('\t•The loadStep %s are considered' %strLoadStep)

        if self.boolFiniteElementBorderProjectedInTheBestPlane is False : 
            print("The finite element mesh border weren't express in the best coordinate frame")
            return None

        temp = self.stepTimeFile[self.stepTimeFile['loadStep'].isin(listLoadStep)].copy()

        temp['TimeStamp'] -= temp['TimeStamp'].min()

        self.listCorrespondingTimeOfDisplacementsInterpolated = temp['TimeStamp'].to_list()


        listDisplacementInterpolated = []

        # Loop over load step :
        for i, loadstep in enumerate(listLoadStep) : 
            coord, displ = self.displacementsDataExpressProjectedInThePlaneExpressInFeFrame['%i' %loadstep]

            # Loop over part : 
            for ii, part in enumerate(self.listFiniteElementBorderProjectedInTheBestPlane) :

                # Interpolation of the two fields :
                interpolation = griddata(coord[:, 0:2], displ[:, 0:2], part[:, 0:2], method = 'linear')
            
                # Reverse displacement in direction 2 :
                interpolation[:, 1] *= -1

                # Special method to complete interpolation when there is np.nan :
                interpolation = self.completeInterpolation(interpolation)

                if i == 0 : 
                    listDisplacementInterpolated.append([interpolation])

                else :  
                    listDisplacementInterpolated[ii].append(interpolation)


        # Transform to 3d array and store them in a list
        self.listDisplacementInterpolated = []

        for i in range(len(listDisplacementInterpolated)) :  
            self.listDisplacementInterpolated.append(np.dstack(listDisplacementInterpolated[i]))

        print()
        

    def completeInterpolation(self, interpolationResults: np.ndarray) : 
        """
            Special method to replace np.nan at extremities in interpolation results from griddata
        """

        nRow, nCol = interpolationResults.shape 

        # Loop over fields interpolate :
        for col in range(nCol) : 

            indexList = []
            for i, val in enumerate(interpolationResults[:, col]) : 
                if np.isnan(val): 
                    indexList.append(i)

            for index in indexList : 
                if index < (nRow//2) :

                    i = index
                    while np.isnan(interpolationResults[i, col] ) :
                        i += 1

                    interpolationResults[index, col] = interpolationResults[i, col]

                else : 

                    i = index
                    while np.isnan(interpolationResults[i, col] ) :
                        i -= 1

                    interpolationResults[index, col] = interpolationResults[i, col]
            
        return interpolationResults

    #--------------------------------------------------------------------------------------------------------
    #                                           Plot methods MPL : 
    #--------------------------------------------------------------------------------------------------------
    def plotSubsetCoordinatesPixel(self, saveName: str = 'subsetCoordinatesPixel-withMpl', pixelToCentimeter: float = 40) : 
        """
            Method to plot the subset coordinates
        """

        if self.boolDataReduced is True : 
            df = self.dicDataRaw.copy()

        else : 
            df = self.dicData.copy()

        # Determine the dimensions of the plot : 
        xMin = df[self.strXnamePixel_DIC].min()
        xMax = df[self.strXnamePixel_DIC].max()
        xDelta = abs(xMax - xMin)

        yMin = df[self.strYnamePixel_DIC].min()
        yMax = df[self.strYnamePixel_DIC].max()
        yDelta = abs(yMax - yMin)

        if xDelta < yDelta : 
            length = xDelta 
        else : 
            length = yDelta


        plotWidth = self.centimetre * (length/pixelToCentimeter)

        fig = plt.figure(figsize = (plotWidth,
                                    plotWidth),
                         )

        ax = fig.add_subplot(111,
                             aspect = 'equal')

        ax.plot(df[self.strXnamePixel_DIC],
                df[self.strYnamePixel_DIC],
                '+',
                ms = 1.5,
                mew = 0.5
                )

        ax.invert_yaxis()


        ax.set_xlabel('X [px]')
        ax.set_ylabel('Y [px]')


        fig.savefig(saveName + '.pdf',
                    format = 'pdf',
                    # dpi = 600,
                    orientation = "portrait",
                    bbox_inches = "tight",
                    pad_inches = 0)


    def plotSubsetCoordinatesPixelRawAndReduced(self, saveName: str = 'subsetCoordinatesPixelRawAndReduced-withMpl', pixelToCentimeter: float = 40) : 
        """
            Method to plot the subset coordinates
        """

        if self.boolDataReduced is True : 
            df = self.dicDataRaw.copy()
            dfReduced = self.dicData.copy()

        else : 
            print("Data were not reduced, no plot")
            return None

        # Determine the dimensions of the plot : 
        xMin = df[self.strXnamePixel_DIC].min()
        xMax = df[self.strXnamePixel_DIC].max()
        xDelta = abs(xMax - xMin)

        yMin = df[self.strYnamePixel_DIC].min()
        yMax = df[self.strYnamePixel_DIC].max()
        yDelta = abs(yMax - yMin)

        if xDelta < yDelta : 
            length = xDelta 
        else : 
            length = yDelta


        plotWidth = self.centimetre * (length/pixelToCentimeter)

        fig = plt.figure(figsize = (plotWidth,
                                    plotWidth),
                         )

        ax = fig.add_subplot(111,
                             aspect = 'equal')

        # Raw data : 
        ax.plot(df[self.strXnamePixel_DIC],
                df[self.strYnamePixel_DIC],
                '.',
                ms = 1.5,
                mew = 0.0,
                )

        # Reduced data : 
        ax.plot(dfReduced[self.strXnamePixel_DIC],
                dfReduced[self.strYnamePixel_DIC],
                '+',
                ms = 1.5,
                mew = 0.5
                )

        ax.invert_yaxis()


        ax.set_xlabel('X [px]')
        ax.set_ylabel('Y [px]')


        fig.savefig(saveName + '.pdf',
                    format = 'pdf',
                    # dpi = 600,
                    orientation = "portrait",
                    bbox_inches = "tight",
                    pad_inches = 0)


    def plotSubsetCoordinatesPixelAndTriangulation(self, saveName: str = 'subsetCoordinatesPixelAndTriangulation-withMpl', pixelToCentimeter: float = 40) :
        """
        """

        if self.boolTriangulationComputed is False : 
            self.computeTriangulation()

        df = self.dicData.copy()
        # Determine the dimensions of the plot : 
        xMin = df[self.strXnamePixel_DIC].min()
        xMax = df[self.strXnamePixel_DIC].max()
        xDelta = abs(xMax - xMin)

        yMin = df[self.strYnamePixel_DIC].min()
        yMax = df[self.strYnamePixel_DIC].max()

        yDelta = abs(yMax - yMin)

        if xDelta < yDelta : 
            length = xDelta 
        else : 
            length = yDelta
        
    
        plotWidth = self.centimetre * (length/pixelToCentimeter)

        fig = plt.figure(figsize = (plotWidth,
                                    plotWidth),
                         )

        ax = fig.add_subplot(111,
                             aspect = 'equal')

        ax.plot(df[self.strXnamePixel_DIC],
                df[self.strYnamePixel_DIC],
                'o',
                ms = 1.5,
                mew = 0.
                )

        ax.triplot(self.triangulation,
                   '-', 
                   lw = 0.5)
        
        ax.invert_yaxis()


        ax.set_xlabel('X [px]')
        ax.set_ylabel('Y [px]')


        fig.savefig(saveName + '.pdf',
                    format = 'pdf',
                    # dpi = 600,
                    orientation = "portrait",
                    bbox_inches = "tight",
                    pad_inches = 0)


    def plotSubsetCoordinatesPixelAndFullTriangulation(self, saveName: str = 'subsetCoordinatesPixelAndFullTriangulation-withMpl', pixelToCentimeter: float = 40) :
        """
        """

        if self.boolFullTriangulationComputed is False : 
            self.computeFullTriangulation()

        df = self.dicDataRaw.copy()
        # Determine the dimensions of the plot : 
        xMin = df[self.strXnamePixel_DIC].min()
        xMax = df[self.strXnamePixel_DIC].max()
        xDelta = abs(xMax - xMin)

        yMin = df[self.strYnamePixel_DIC].min()
        yMax = df[self.strYnamePixel_DIC].max()

        yDelta = abs(yMax - yMin)

        if xDelta < yDelta : 
            length = xDelta 
        else : 
            length = yDelta
        
    
        plotWidth = self.centimetre * (length/pixelToCentimeter)

        fig = plt.figure(figsize = (plotWidth,
                                    plotWidth),
                         )

        ax = fig.add_subplot(111,
                             aspect = 'equal')

        # ax.plot(df[self.strXnamePixel_DIC],
                # df[self.strYnamePixel_DIC],
                # 'o',
                # ms = 1.,
                # mew = 0.
                # )

        ax.triplot(self.fullTriangulation,
                   'o-', 
                   ms = 0.5,
                   mew = 0.,
                   lw = 0.3)
        
        ax.invert_yaxis()


        ax.set_xlabel('X [px]')
        ax.set_ylabel('Y [px]')


        fig.savefig(saveName + '.pdf',
                    format = 'pdf',
                    # dpi = 600,
                    orientation = "portrait",
                    bbox_inches = "tight",
                    pad_inches = 0)
        del df


    #--------------------------------------------------------------------------------------------------------
    #                                          Trace objects Plotly : 
    #--------------------------------------------------------------------------------------------------------
    def goSurface(self, XX: np.ndarray, YY: np.ndarray, ZZ: np.ndarray, dictOptions: dict ) :
        """
        """
        # // TODO: code goSurface function
        pass
    

    def goScatter3d(self, x: np.ndarray, y: np.ndarray, z: np.ndarray) : 
        """
        """
        # // TODO: code goScatter3d function

        pass


    def goMesh3d(self, x: np.ndarray, y: np.ndarray, z: np.ndarray, triangulation: np.ndarray) : 
        """
        """
        # // TODO: code goMesh3d function

        pass


    def goTraceSurfaceBestPlane(self, floatPlaneMargin: float = 5.) : 
        """
            Method to create the trace of best plane

        """

        if self.boolBestPlaneFitted is False : 
            print("Best plane wasn't fitted")
            return None

        xCoord = self.dicData[self.strXname_DIC].to_numpy()
        yCoord = self.dicData[self.strYname_DIC].to_numpy()

        # 1. Get data for the plane : 
        xCoordMin = xCoord.min() 
        xCoordMax = xCoord.max() 
        yCoordMin = yCoord.min() 
        yCoordMax = yCoord.max() 
        
        # 2 Create the grid : 
        xSurf = np.linspace(xCoordMin - floatPlaneMargin, xCoordMax + floatPlaneMargin, 2)
        ySurf = np.linspace(yCoordMin - floatPlaneMargin, yCoordMax + floatPlaneMargin, 2)
        xSurf, ySurf = np.meshgrid(xSurf, ySurf)
        
        # 4.3 Compute surface :
        a, b, c, d = self.fitBestPlaneCartesianEquation
        zSurf = (-a*xSurf -b*ySurf - d)/c
        
        self.traceBestPlaneSurface = go.Surface(x          = xSurf,
                                                y          = ySurf,
                                                z          = zSurf,
                                                colorscale = [[0, 'orange'],
                                                              [1, 'orange']],
                                                opacity    = 0.5,
                                                showscale  = False,
                                                showlegend = True,
                                                name       = 'Best plane fitted',
                                                )

        self.boolTraceBestPlaneSurface = True
        
    
    def goTraceSubsetMesh3D(self) : 
        """
        """

        # Check if the triangulation was computed :
        if self.boolTriangulationComputed is False : 
            self.computeTriangulation()

        # 3. Get data of DIC :  
        rightTriangulation = self.triangulation.get_masked_triangles() 
        xCoord = self.dicData[self.strXname_DIC].to_numpy()
        yCoord = self.dicData[self.strYname_DIC].to_numpy()
        zCoord = self.dicData[self.strZname_DIC].to_numpy()
        
        self.traceSubsetMesh3D = go.Mesh3d(x          = xCoord,
                                           y          = yCoord,
                                           z          = zCoord,
                                           color      = 'black', 
                                           opacity    = 1.0,
                                           # alphahull = 3, 
                                           i          = rightTriangulation[:, 0], 
                                           j          = rightTriangulation[:, 1], 
                                           k          = rightTriangulation[:, 2],
                                           name       = 'DIC data',
                                           showlegend = True,
                                           )

        self.boolTraceSubsetMesh3D = True
     

    def goTraceFeMeshLine(self) : 
        """
        """

        pass


    def goTraceSubsetDisplacements(self, timeStep) : 
        """
        """
        # trace = go.Scatter3d(x = ,
                             # y = ,
                             # )

        # self.tracesSubsetDisplacement.update({'%i' %timeStep : trace})

        pass

    
    def goTraceSubsetCoordinates(self) : 
        """
        """
        self.traceSubsetCoordinatesScatter = go.Scatter3d(x      = self.dicData[self.strXname_DIC],
                                                          y      = self.dicData[self.strYname_DIC],
                                                          z      = self.dicData[self.strZname_DIC],
                                                          mode   = 'markers',
                                                          marker = dict(size = 2,
                                                                        color = 'green'),
                                                          name   = 'DIC Subsets',
                                                          )


    def goTraceSubsetCoordinatesProjected(self) : 
        """
        """

        self.traceSubsetCoordinatesProjectedScatter = go.Scatter3d(x = self.subsetCoordsProjected[:, 0],
                                                                   y = self.subsetCoordsProjected[:, 1],
                                                                   z = self.subsetCoordsProjected[:, 2],
                                                                   mode = 'markers',
                                                                   marker = dict(size = 2,
                                                                                 color = 'red'),
                                                                   name = 'DIC Subsets Projected'
                                                                   )
        self.boolTrace = True


    def goTraceSubsetCoordinatesUsedForDisplacementsExtraction(self) : 
        """
        """

        data = self.displacementsData['%i' %self.firstTimeStep][[self.strXname_DIC_ref, self.strYname_DIC_ref, self.strZname_DIC_ref]].to_numpy()
        self.traceSubsetCoordinatesUsedForDisplacementsExtraction = go.Scatter3d(x = data[:, 0],
                                                                                 y = data[:, 1],
                                                                                 z = data[:, 2],
                                                                                 mode = 'markers',
                                                                                 marker = dict(size = 1,
                                                                                               color = 'black'),
                                                                                 name = 'Subset used for displacements extraction',
                                                                                 # legendgroup = 'Finite element border',
                                                                                 )
        
        self.boolTraceSubsetCoordinatesUsedForDisplacementsExtraction = True

    
    def goTraceSubsetCoordinatesUsedForDisplacementsExtractionInTheBestPlane(self) : 
        """
        """

        # Project point in the bestPlane :
        data = []
        for subset in self.displacementsData['%i' %self.firstTimeStep][[self.strXname_DIC_ref, self.strYname_DIC_ref, self.strZname_DIC_ref]].to_numpy() :
            subset = np.array(self.bestPlane.project_point(subset))
             
            data.append(np.matmul(np.linalg.inv(self.passageMatrixToExpressInTheBestPlane), subset))

        data = np.array(data)
        self.traceSubsetCoordinatesUsedForDisplacementsExtractionInTheBestPlane = go.Scatter3d(x = data[:, 0],
                                                                                               y = data[:, 1],
                                                                                               z = data[:, 2],
                                                                                               mode = 'markers',
                                                                                               marker = dict(size = 1,
                                                                                                             color = 'black',
                                                                                                             ),
                                                                                               name = 'Subset used for displacements extraction',
                                                                                               )

        self.boolTraceSubsetCoordinatesUsedForDisplacementsExtractionInTheBestPlane = True
        
    
    def goTraceFiniteElementBorderProjected(self) : 
        """
        """

        if self.boolFiniteElementBorderProjected is False : 
            print("The finite element border weren't projected, please refeer to method projectFiniteElementBorder")
            return None
        
        self.traceFiniteElementBorderProjected = []
        for i, part in enumerate(self.listFiniteElementBorderProjected) : 
            if i == 0 : 
                statementShowLegend  = True
            else : 
                statementShowLegend  = False 

            self.traceFiniteElementBorderProjected.append(go.Scatter3d(x = part[:, 0],
                                                                       y = part[:, 1],
                                                                       z = part[:, 2],
                                                                       mode = 'markers',
                                                                       marker = dict(size = 4,
                                                                                     color = 'red'),
                                                                       name = 'FE borders Projected',
                                                                       legendgroup = 'Finite element border',
                                                                       showlegend = statementShowLegend,
                                                                       )
                                                          )

        self.boolTraceFiniteElementBorderProjected = True

    
    def goTraceFiniteElementBorderProjectedInTheBestPlane(self) : 
        """
        """

        if self.boolFiniteElementBorderProjectedInTheBestPlane is False : 
            print("The FE mesh wasn't express in the best plane, please do it using projectFiniteElementMeshBorderInTheBestPlane")
            return None

        self.traceFiniteElementBorderProjectedInTheBestPlane = []
        for i, part in enumerate(self.listFiniteElementBorderProjectedInTheBestPlane) : 
            if i == 0 : 
                statementShowLegend  = True
            else : 
                statementShowLegend  = False 

            self.traceFiniteElementBorderProjectedInTheBestPlane.append(go.Scatter3d(x = part[:, 0],
                                                                                     y = part[:, 1],
                                                                                     z = part[:, 2],
                                                                                     mode = 'markers',
                                                                                     marker = dict(size = 4,
                                                                                                   color = 'red'),
                                                                                     name = 'FE borders Projected',
                                                                                     legendgroup = 'Finite element border',
                                                                                     showlegend = statementShowLegend,
                                                                                     )
                                                                        )

        self.boolTraceFiniteElementBorderProjectedInTheBestPlane = True


    #--------------------------------------------------------------------------------------------------------
    #                                         3D Plolty figures : 
    #--------------------------------------------------------------------------------------------------------
    def plot3dShapeAndBestPlane(self, saveName: str = 'bestPlanePlot-withPlotly', camera: np.ndarray = None) : 
        """
            Method to plot the best planeFit and the original data with plolty
        """

        # 1. Add coordinate system :
        if camera is not None : 
            # Check if input is correct : 
            if camera.shape != (3, 3) : 
                print("np.ndarray passed for coordinateSystem hasn't the right shape")
                return None

            else : 
                coordinateSystem = coordinateSystemPlotly(camera)


        # 2. Plot data in 3d : 
        fig = go.Figure()

        # Subet coordinate / Mesh3D
        if self.boolTraceSubsetMesh3D is False : 
            self.goTraceSubsetMesh3D()

        fig.add_trace(self.traceSubsetMesh3D)
        
        # Best plane : 
        if self.boolTraceBestPlaneSurface is False : 
            self.goTraceSurfaceBestPlane()

        fig.add_trace(self.traceBestPlaneSurface)

        if camera is not None : 
            for trace in coordinateSystem.traces : 
                fig.add_trace(trace)


        bestPlaneCoordinateSystem = coordinateSystemPlotly(np.array([self.centralPointBestPlane,
                                                                     self.centralPointBestPlane + self.xVectorBestPlaneRotated,
                                                                     self.centralPointBestPlane + self.yVectorBestPlaneRotated])
                                                           )
        for trace in bestPlaneCoordinateSystem.traces : 
                fig.add_trace(trace)


        fig.update_layout(scene = dict(aspectmode = 'data'
                                       # // NOTE: Use this option to obtain plot 3d data triangulated with aspect ratio :
                                       ),
                          yaxis = dict(autorange = "reversed",
                                       ),
                          scene_camera = dict(eye = dict(x=0.0, y=-1., z=-5) ,
                                              ),
                          )

        fig.write_html("%s.html" %saveName)
    

    def plot3dShapeBestPlaneAndFiniteElementMesh(self, saveName: str = 'bestShapeBestPlaneFittedAndFiniteElementMeshPlot-withPlotly', coordinateSystem: np.ndarray = None, floatPlaneMargin: float = 10.) : 
        """
            Method to plot the finite element mesh
        """

        # 1. Add coordinate system :
        if coordinateSystem is not None : 
            # Check if input is correct : 
            if coordinateSystem.shape != (3, 3) : 
                print("np.ndarray passed for coordinateSystem hasn't the right shape")
                return None

            else : 
                coordinateSystem = coordinateSystemPlotly(coordinateSystem)


        # 2. Plot data in 3d : 
        fig = go.Figure()

        # Subet coordinate / Mesh3D
        if self.boolTraceSubsetMesh3D is False : 
            self.goTraceSubsetMesh3D()

        fig.add_trace(self.traceSubsetMesh3D)
        
        # Best plane : 
        if self.boolTraceBestPlaneSurface is False : 
            self.goTraceSurfaceBestPlane()
        
        fig.add_trace(self.traceBestPlaneSurface)

        # Finite element mesh : 
        for i, part in enumerate(self.finiteElementMeshProjected) : 
            fig.add_trace(go.Scatter3d(x          = part[:, 0],
                                       y          = part[:, 1],
                                       z          = part[:, 2],
                                       mode       = 'lines',
                                       line       = dict(color = 'red',
                                                         width = 2,
                                                         ),
                                       name       = 'FE part n°%i' %i,
                                       showlegend = True,
                                     ),
                          )

        if coordinateSystem is not None : 
            for trace in coordinateSystem.traces : 
                fig.add_trace(trace)

        fig.update_layout(scene = dict(aspectmode = 'data'
                                       # // NOTE: Use this option to obtain plot 3d data triangulated with aspect ratio :
                                       ),
                          yaxis = dict(autorange = "reversed",
                                       ),
                          scene_camera = dict(eye = dict(x=0.0, y=-1., z=-5) ,
                                              ),
                          )

        fig.write_html("%s.html" %saveName)
    

    def plot3dShapeBestPlaneSubsetCoordinatesScatter(self, saveName: str = 'bestShapeBestPlaneFittedAndSubsetCoordinatesScatter-withPlotly') : 
        """ 
        """

        fig = go.Figure()

        # Subet coordinate / Mesh3D
        if self.boolTraceSubsetMesh3D is False : 
            self.goTraceSubsetMesh3D()

        fig.add_trace(self.traceSubsetMesh3D)
        
        # Best plane : 
        if self.boolTraceBestPlaneSurface is False : 
            self.goTraceSurfaceBestPlane()

        # Scatter : 
        self.goTraceSubsetCoordinates()
        self.goTraceSubsetCoordinatesProjected()

        fig.add_trace(self.traceSubsetCoordinatesScatter)
        fig.add_trace(self.traceSubsetCoordinatesProjectedScatter)

        fig.add_trace(self.traceBestPlaneSurface)

        fig.update_layout(scene = dict(aspectmode = 'data'
                                       # // NOTE: Use this option to obtain plot 3d data triangulated with aspect ratio :
                                       ),
                          yaxis = dict(autorange = "reversed",
                                       ),
                          scene_camera = dict(eye = dict(x=0.0, y=-1., z=-5) ,
                                              ),
                          )

        fig.write_html("%s.html" %saveName)


    def plot3dShapeBestPlaneSubsetCoordinatesAndDisplacements(self, timeStep: int, saveName: str = "bestPlanePlotSubsetCoordinatesAndDisplacements-withPlotly") :
        """
            Method to plot with plotly : 
                - the best plane
                - the subset mesh
                - the subset coordinate
                - the displacement associate
        """

        if timeStep not in self.displacementsData.keys() : 
            print("The time step %i not loaded" %timeStep)
            return None

        self.goTraceSubsetDisplacements(timeStep)

        # 3d figure : 
        fig = go.Figure()
        
        # Subet coordinate / Mesh3D
        if self.boolTraceSubsetMesh3D is False : 
            self.goTraceSubsetMesh3D()

        fig.add_trace(self.traceSubsetMesh3D)
        
        # Best plane : 
        if self.boolTraceBestPlaneSurface is False : 
            self.goTraceSurfaceBestPlane()
         
        fig.add_trace(self.traceBestPlaneSurface)
         
        #  Update layout to match image axis : 
        fig.update_layout(scene = dict(aspectmode = 'data'
                                       # // NOTE: Use this option to obtain plot 3d data triangulated with aspect ratio :
                                       ),
                          yaxis = dict(autorange = "reversed",
                                       ),
                          scene_camera = dict(eye = dict(x=0.0, y=-1., z=-5) ,
                                              ),
                          )
        
        saveName = saveName.replace('-', '-timeStep-%i-' %timeStep)
        fig.write_html("%s.html" %saveName)


    def plot3dBestPlaneFiniteElementBorderProjectedAndSubsetCoordinatesUsedForDisplacementsExtraction(self, saveName: str = "bestPlaneFiniteElementBorderProjectedAndSubsetCoordinatesUsedForDisplacementsExtraction-withPlotly") :
        """
            Method to compute
        """

        # 3d figure : 
        fig = go.Figure()

        # Best plane : 
        if self.boolTraceBestPlaneSurface is False : 
            self.goTraceSurfaceBestPlane()

        fig.add_trace(self.traceBestPlaneSurface)

        # Finite element border
        if  self.boolTraceFiniteElementBorderProjected is False : 
            self.goTraceFiniteElementBorderProjected()

        fig.add_traces(self.traceFiniteElementBorderProjected)

        if self.boolTraceSubsetCoordinatesUsedForDisplacementsExtraction is False : 
            self.goTraceSubsetCoordinatesUsedForDisplacementsExtraction()

        fig.add_trace(self.traceSubsetCoordinatesUsedForDisplacementsExtraction)

        #  Update layout to match image axis : 
        fig.update_layout(scene = dict(aspectmode = 'data'
                                       # // NOTE: Use this option to obtain plot 3d data triangulated with aspect ratio :
                                       ),
                          yaxis = dict(autorange = "reversed",
                                       ),
                          scene_camera = dict(eye = dict(x=0.0, y=-1., z=-5) ,
                                              ),
                          )
        
        fig.write_html("%s.html" %saveName)


    def plot3dFiniteElementMeshBorderAndSubsetCoordinatesUsedForDisplacementsExtractionInTheBestPlane(self, saveName: str = "finiteElementBorderAndSubsetCoordinatesProjectedInTheBestPlane-withPlotly") : 
        """
        """

        # 3d figure : 
        fig = go.Figure()
        
        if self.boolTraceFiniteElementBorderProjectedInTheBestPlane is False : 
            self.goTraceFiniteElementBorderProjectedInTheBestPlane()

        fig.add_traces(self.traceFiniteElementBorderProjectedInTheBestPlane)

        if self.boolTraceSubsetCoordinatesUsedForDisplacementsExtractionInTheBestPlane is False : 
            self.goTraceSubsetCoordinatesUsedForDisplacementsExtractionInTheBestPlane()

        fig.add_trace(self.traceSubsetCoordinatesUsedForDisplacementsExtractionInTheBestPlane)
        
        fig.add_trace(go.Scatter3d(x = [ 0 ],
                                   y = [ 0 ], 
                                   z = [ 0 ]),
                      )

        #  Update layout to match image axis : 
        fig.update_layout(scene = dict(aspectmode = 'data'
                                       # // NOTE: Use this option to obtain plot 3d data triangulated with aspect ratio :
                                       ),
                          yaxis = dict(autorange = "reversed",
                                       ),
                          scene_camera = dict(eye = dict(x=0.0, y=-1., z=-5) ,
                                              ),
                          )

        fig.write_html("%s.html" %saveName)


    #--------------------------------------------------------------------------------------------------------
    #                                            Save methods : 
    #--------------------------------------------------------------------------------------------------------
    def saveAnglesAndBestPlaneEquation(self, saveName: str = 'anglesAndBestPlaneEquation.txt') :
        """
            Method to save angles alpha & beta and the plane coefficient
        """
        sp = ' '
        # Check if the angles were computed and the plane fitted :
        if self.boolAlphaBetaAnglesComputed is True and self.boolBestPlaneFitted is True :
            a, b, c, d = self.fitBestPlaneCartesianEquation
            stringToWrite = sp.join([str(self.alpha), str(self.beta), str(a), str(b), str(c), str(d)])
            # print(stringToWrite)
            with open(saveName, 'w') as target:
                target.write(stringToWrite )
    

    def saveTriangulation(self, saveName: str = 'triangulation.txt') :
        """
        """
        if self.boolTriangulationComputed is True : 
            print("Save reduced data triangulation")
            triangulation = self.triangulation.get_masked_triangles()
            np.savetxt(saveName, triangulation)
            del triangulation

        if self.boolFullTriangulationComputed is True : 
            print("Save Full triangulation")
            triangulation = self.fullTriangulation.get_masked_triangles()
            np.savetxt('fullTriangulation.txt', triangulation)
            del triangulation

    
    def saveDisplacementsAsAmplitudeForAbaqus(self, listPart: list, saveName: str) :
        """ 
            Method to save displacement interpolate over the time as amplitude for Abaqus
            Each amplitude will be name as : 
                Amp-%partName-%fieldNumber-%nodeNumber
        """

        print("• Save displacements interpolated as amplitude for Abaqus:")

        if len(listPart) != len(self.listDisplacementInterpolated) :
            print("\t• The number of part name in listPart isn't correct")
            print("\t• Expected %i, gave %i" %(len(self.listDisplacementInterpolated, len(listPart))))
            return None

        else : 
            print("\t• The following set will be used:")
            for part in listPart : 
                print('\t\t• %s' %part)
        

        timeList = self.listCorrespondingTimeOfDisplacementsInterpolated

        with open(saveName, 'w') as file : 
            # Loop over set :
            for part, partName in zip(self.listDisplacementInterpolated, listPart) :

                # Loop over field :
                for j in range(part.shape[1]) :

                    # Loop over node :
                    for i in range(part.shape[0]) :
                        
                        file.write("*Amplitude, name=Amp-%s-%i-%i" %(partName, (j + 1), (i + 1)) + "\n")

                        data = part[i, j, :]

                        # Write data :
                        for k, val in enumerate(data) : 
                            file.write('%f,%f\n' %(timeList[k], val))


        for part, displacementInterpolated in zip(listPart, self.listDisplacementInterpolated) : 
            np.save(part + '.npy', displacementInterpolated)
        print()


    def saveSubsetCoordinatesProjectedInTheBestPlane(self, saveName) : 
        """ 
            Method to save the subset coordinates projected in the best plane and save them
        """

        if self.boolSubsetCoordsProjected is False :
            print("The subset coordinates weren't projected to the best plane")
            return None
        

        data = self.dicData.copy()

        data[self.strXname_DIC] = self.subsetCoordsProjected[:, 0]
        data[self.strYname_DIC] = self.subsetCoordsProjected[:, 1]
        data[self.strZname_DIC] = self.subsetCoordsProjected[:, 2]
        

        data[[self.strXnamePixel_DIC, self.strYnamePixel_DIC, self.strXname_DIC, self.strYname_DIC, self.strZname_DIC]].to_csv(saveName,
                    header = True,
                    index = False, 
                    sep = ";",
                    )


    #--------------------------------------------------------------------------------------------------------
    #                                         Projection methods : 
    #--------------------------------------------------------------------------------------------------------
    def computeRotationMatrix(self, gamma: float) :
        """
            Method to compute the rotation matrix : 
            Successive rotations arround : 
                1. alpha (x) roll
                2. beta  (y) pitch
                3. gamma (z) yaw

            All angles are expresed in radians
        """

        if self.boolAlphaBetaAnglesComputed is False : 
            return None

        Rx = np.array([[1.,           0.,                  0.],
                       [0.,  np.cos(self.alpha), -np.sin(self.alpha)],
                       [0.,  np.sin(self.alpha),  np.cos(self.alpha)],
                       ])

        Ry = np.array([[ np.cos(self.beta), 0., np.sin(self.beta)],
                       [         0.,        1.,        0.        ],
                       [-np.sin(self.beta), 0., np.cos(self.beta)],
                       ])

        Rz = self.computeInPlaneRotationMatrix(gamma)

        rotationMatrix = np.matmul(Rx, Ry) 
        rotationMatrix = np.matmul(rotationMatrix, Rz)

        return rotationMatrix


    def computeInPlaneRotationMatrix(self, gamma: float) : 
        """ 
            Method to compute the rotation matrix arround z 
        """

        if self.boolAlphaBetaAnglesComputed is False : 
            return None


        Rz = np.array([[np.cos(gamma), -np.sin(gamma), 0.],
                       [np.sin(gamma),  np.cos(gamma), 0.],
                       [      0.,            0.,       1.],
                       ])

        return Rz 
    

    def compute2DrotationMatrix_simpleMethod(self, angle: float) :
        """
            Method to compute the rotation matrix 
            2x2 matrix
        """

        return np.array([[np.cos(angle), -np.sin(angle)],
                         [np.sin(angle),  np.cos(angle)]])
    

    #--------------------------------------------------------------------------------------------------------
    #                                         Simple projection methods : 
    #--------------------------------------------------------------------------------------------------------
    def projectFiniteElementMesh_simpleMethod(self, feNodes: np.ndarray, conversionFactor_mm_to_px: float, uTranslation: float, vTranslation: float, gammaRotation:float) :
        """
            Method to project FiniteElementMesh on image using :
                - feNodes : np.ndarray, it will use the two first column as x_cam0 and y_cam0, the third column is the node Label
                - conversionFactor_mm_to_px : factor to convert node coordinate in CAM_0 coordinate frame 
                - uTranslation : translation along u vector in sensor coordinate frame, in pixel
                - vTranslation : translation along v vector in sensor coordinate frame, in pixel
                - gammaRotation : rotation angle in the plane, in degrees
        """
        
        self.nodesProjected_simpleMethod = []

        if feNodes.shape[1] != 3 : 
            self.leftProgramme("The finite element mesh hasn't almost 3 columns")
            
        else : 
            # Copy :
            self.nodesProjected_simpleMethod = deepcopy(feNodes)
            
            # Apply conversion mm to px :
            self.nodesProjected_simpleMethod[:, 0] *= conversionFactor_mm_to_px
            self.nodesProjected_simpleMethod[:, 1] *= conversionFactor_mm_to_px

            # Compute 2d rotation matrix :
            rotationMatrix = self.compute2DrotationMatrix_simpleMethod(np.radians(gammaRotation))

            # Apply rotation :
            self.nodesProjected_simpleMethod[:, [0,1]] = np.matmul(self.nodesProjected_simpleMethod[:, [0,1]], rotationMatrix.T)

            # Apply translation
            self.nodesProjected_simpleMethod[:, 0] = self.nodesProjected_simpleMethod[:, 0] + uTranslation
            self.nodesProjected_simpleMethod[:, 1] = self.nodesProjected_simpleMethod[:, 1] + vTranslation

        
    def plotMeshOnImage_simpleMethod(self, saveName: str, imagePath: str, connectivityTable: np.ndarray = None) :
        """ 
            Method to plot the mesh projected on the image with matploltib
            Need to give image path to open it and use it for the plot
            If connectivityTable is passed, it will plot the element
        """

        try : 
            image = io.imread(imagePath)

        except Exception as error : 
            print("Error :", error) 
            self.leftProgramme("Error during file : %s openning" %imagePath)


        fig = plt.figure()
        ax = fig.add_subplot(111,
                             aspect = 'equal',
                             )

        ax.imshow(image,
                  vmin = 0,
                  vmax = 255, 
                  cmap = 'gray',
                  interpolation = None,
                  )
        
        # Case if connectivityTable is giving 
        if not connectivityTable is None : 
            if isinstance(connectivityTable, np.ndarray) : 
                for element in connectivityTable : # Loop over element :
                    u, v = [], []
                    for node in element[1:] : # Loop over nodes who composed element
                        n = self.nodesProjected_simpleMethod[np.where(self.nodesProjected_simpleMethod[:, 2] == node)]
                        u.append(n[0, 0])
                        v.append(n[0, 1])

                    ax.plot(u,
                            v,
                            lw = 0.1,
                            color = 'red')
                    
                    del u, v

            else : 
                self.leftProgramme("Error, connectivityTable isn't a np.array")

        # Plot nodes
        ax.plot(self.nodesProjected_simpleMethod[:, 0],
                self.nodesProjected_simpleMethod[:, 1],
                'o',
                ms = 0.5,
                mec = 'Gold',
                mew = 0.,
                color = 'Gold')
        
        ax.set_xlabel(r"$u$ [Pixel]")
        ax.set_ylabel(r"$v$ [Pixel]")

        fig.savefig("./" + saveName + '.pdf',
                    format = 'pdf',
                    dpi = 500,
                    orientation = "portrait",
                    bbox_inches="tight",
                    pad_inches = 0.)


    def boundaryConditionsExtractionUsing_simpleMethod(self, boundaryNSet: dict | list) : 
        """
            Method to extract boundary conditions and export them for MatchID
            boundaryNset : dict {xyNodesCoord    : np.ndarray,
                                 nSetPrefix      : str,
                                 amplitudePrefix : str,
            } 
        """

        dictKeys = ['nodes', 'nSetPrefix', 'amplitudePrefix', 'amplitudeFileName', 'boundaryConditionsFileName']

        match boundaryNSet :
            case list() : 
                boundaryNSet = []
                for i, instance in enumerate(boundaryNSet, start = 1) : 
                    if not isinstance(instance, dict) : 
                        self.leftProgramme("The instance n°{:g} is'nt a dict".format(i))

                    else : 
                        if not boundaryNSet.keys() in dictKeys : 
                            self.leftProgramme('The following keys are missing : %s' %list(set(dictKeys).difference(list(boundaryNSet.keys()))))

                    boundaryNSet.append(boundaryNSet['nodes'])

            case dict() : 
                if len(list(set(dictKeys) - set(boundaryNSet.keys()))) != 0 :
                    self.leftProgramme('The following keys are missing : %s' %list(set(dictKeys).difference(list(boundaryNSet.keys()))))


                displacementInterpolated = []
                for index in self.listCorrespondingTimeOfDisplacementsInterpolated[:, 0] :
                    displacementInterpolated.append(self.displacementsInterpolation_simpleMethod(boundaryNSet['nodes'], index))

                displacementInterpolatedOverTime = np.dstack((displacementInterpolated))

                nRows, nCols, nDeep =  displacementInterpolatedOverTime.shape

                with open(boundaryNSet['amplitudeFileName'], 'w') as amplitudeFile, open(boundaryNSet['boundaryConditionsFileName'], "w") as boundaryFile :
                    # Loop over unique value :
                    for i in range(nRows) : 
                        # Loop over displacement field 
                        for j in range(nCols) : 
                            amplitudeFile.write("*Amplitude, name={:}{:n}-{:n}\n".format(boundaryNSet['amplitudePrefix'], i+1, j+1))
                            amplitudeFile.write("0.,0.\n")

                            # Loop over instant :
                            for k in range(nDeep) : 
                                amplitudeFile.write("{:f},{:f}\n".format(self.listCorrespondingTimeOfDisplacementsInterpolated[k, 1], displacementInterpolatedOverTime[i, j, k]))


                            boundaryFile.write("*Boundary, amplitude={:}{:n}-{:n}\n".format(boundaryNSet['amplitudePrefix'], i+1, j+1))
                            boundaryFile.write("Part-1-1.{:}{:n}, {:n}, {:n}, 1\n".format(boundaryNSet['nSetPrefix'], i+1, j+1, j+1))


    def displacementsInterpolation_simpleMethod(self, nodes: list | np.ndarray, timeStep: int, interpolationMethod: str = 'linear' ) : 
        """     
            Method to interpolate the displacement of a giving timeStep at xyCoord points
        """

        # Load coordinates and displacements subsets :
        coord = self.displacementsData['%i' %timeStep][[self.strXnamePixel_DIC, self.strYnamePixel_DIC]].to_numpy()
        displ = self.displacementsData['%i' %timeStep][[self.strUname_DIC, self.strVname_DIC]].to_numpy()
        

        # Interpolation :
        if isinstance(nodes, np.ndarray) : 
            nodesCoord = self.nodesProjected_simpleMethod[np.isin(element = self.nodesProjected_simpleMethod[:, 2], test_elements = nodes[:, 3])]
            nodesCoord = nodesCoord[nodesCoord[:, 0].argsort()]
            interpolation = griddata(coord[:, 0:2], displ[:, 0:2], nodesCoord[:, 0:2], method = interpolationMethod)

            # Reverse displacement in direction Y to match FE coordinate system
            interpolation[:, 1] *= -1

            return self.completeInterpolation(interpolation)

        elif isinstance(nodes, list) :
            interpolation = []
            for instance in nodes : # Loop over instance
                nodesCoord = self.nodesProjected_simpleMethod[np.isin(element = self.nodesProjected_simpleMethod[:, 2], test_elements = instance[:, 3])]
                nodesCoord = nodesCoord[nodesCoord[:, 0].argsort()]
                interpolation_ = griddata(coord[:, 0:2], displ[:, 0:2], nodesCoord[:, 0:2], method = interpolationMethod)
                
                # Reverse displacement in direction Y to match FE coordinate system
                interpolation_[:, 1] *= -1

                interpolation.append(self.completeInterpolation(interpolation_))
            
            return interpolation


    #--------------------------------------------------------------------------------------------------------
    #                                         Other methods : 
    #--------------------------------------------------------------------------------------------------------
    def leftProgramme(self, message = None) :
        """
        """
        if isinstance(message, str) :
            print(message)
            quit()

        # if message
        if isinstance(message, list) :
            for mess in message :
                if isinstance(mess, str) :
                    print(mess)

                else :
                    continue
            quit()
