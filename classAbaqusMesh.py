import pandas as pd 
import numpy as np 
import meshio 
import os
from shapely.geometry import LineString
from concave_hull import concave_hull, concave_hull_indexes
import matplotlib.pyplot as plt

from copy import deepcopy

class classAbaqusMesh():
    """
        Classe pour exploiter les maillages d'abaqus au format .inp
    """

    def __init__(self, meshFileName: str) : 
        """  
        """

        self.boolExtract2DconnectivityTable = False 

        self.listOfAbaqusKeywords = ['*NSET',
                                     '*ELSET', 
                                     '*NODE',
                                     ]
        
        if not os.path.exists(meshFileName) : 
            self.leftClass()
            
        self.readInpFileToExtractElements(meshFileName) 

        try : 
            mesh = meshio.read(meshFileName, file_format = 'abaqus')

        except Exception as error : 
            print("Error :", error)
            self.leftClass("Error during file : %s openning" %meshFileName)
        
        self.nodes = np.column_stack((mesh.points, np.arange(1, len(mesh.points) + 1, 1, dtype = int) ))
        self.nodeSets = mesh.point_sets
        self.elementSets = mesh.cell_sets

        del mesh

        
    def extract2DconnectivityTable(self, elementSetName: str, nodeSetName: str = None, twoDimensionAssumption: bool = False):
        """ 
            Method to compute connectivity table from 3D mesh
            If the *ELSET and *NSET have the same name, you can just give the *ELSET
        """
        
        # 1. Check if elementSetName is in mesh.cell_sets : 
        if elementSetName not in self.elementSets.keys() : 
            self.leftClass("The *ELSET %s doesn't exist" %elementSetName)

        
        if nodeSetName is not None : 
            if nodeSetName not in self.nodeSets.keys() : 
                self.leftClass("The *NSET %s doesn't exist" %elementSetName)
        else : 
            nodeSetName = elementSetName
        
        # 2. Keep nodes from NSET :
        nodes = self.nodes[self.nodeSets[nodeSetName]]

        # 3. Keep elements from ELSET :
        elements = self.elements[self.elementSets[elementSetName][0]]
        
        # 4 List to store connectivity determined
        connectivity = []

        # 5. Loop over elements :
        for element in elements : 
            elementNodes = nodes[np.isin(nodes[:, 3], element[1:])]
            connectivity.append(self.constructElementFromNodes(elementNodes, twoDimensionAssumption = twoDimensionAssumption))
        
        self.twoDimensionnalConnectivityTable = np.array(connectivity, dtype = int)

        self.boolExtract2DconnectivityTable = True

        del connectivity


    def constructElementFromNodes(self, nodes: np.ndarray, twoDimensionAssumption: bool = False)  : 
        """ 
            Method to determine the sequence of vertices to form quadrangle
        """

        nodeNumbers = nodes [:, -1]

        if twoDimensionAssumption is True : 
            nodes = np.delete(nodes, [2, 3], axis = 1)

        else : 
            nodes = np.delete(nodes, [3], axis = 1)

        # list for the order of nodes  :
        nodesIndex = [0]

        # Intersections of diagonals :
        if LineString(nodes[[0, 1]]).intersects(LineString(nodes[[2, 3]])) : 
            nodesIndex = [0, 2, 1, 3]
            connectivity = [nodeNumbers[0], nodeNumbers[2], nodeNumbers[1], nodeNumbers[3]]
            self.exportTikZelementReconstruction(nodes, nodesIndex, saveName = "exportTikz-Intersection-diagonales.tex")

        else : 
            if LineString(nodes[[0, 2]]).intersects(LineString(nodes[[1, 3]])) : 
                nodesIndex = [0, 1, 2, 3]
                connectivity = [nodeNumbers[0], nodeNumbers[1], nodeNumbers[2], nodeNumbers[3]]
                self.exportTikZelementReconstruction(nodes, nodesIndex, saveName = "exportTikz-cas-1.tex")

            else : 
                nodesIndex =  [0, 1, 3, 2]
                connectivity = [nodeNumbers[0], nodeNumbers[1], nodeNumbers[3], nodeNumbers[2]]
                self.exportTikZelementReconstruction(nodes, nodesIndex, saveName = "exportTikz-cas-2.tex")
        
        return connectivity


    def computeDistanceBetweenNodes(self, n1, n2) :
        """ 
        """
        if len(n1) == 3 : 
            return np.sqrt((n1[0] - n2[0])**2 + (n1[1] - n2[1])**2 + (n1[2] - n2[2])**2)
        
        elif len(n1) == 2 : 
            return np.sqrt((n1[0] - n2[0])**2 + (n1[1] - n2[1])**2)

        else :
            self.leftClass("The dimension of nodes coordinates aren't 2 or 3")


    def readInpFileToExtractElements(self, meshFileName: str) : 
        """
            Method to 
        """

        with open(meshFileName, 'r') as file : 
            fileLines = file.readlines()

        numberOfLines = len(fileLines)
        firstCondition = False 
        secondCondition = False
        intEndElementSection = -1 
        intStartElementSection = -1


        i = 0 
        while "*ELEMENT" not in fileLines[i].upper() and i <= numberOfLines : 
            i += 1
        else : 
            intStartElementSection = i+1
            firstCondition = True

        while secondCondition is False :
            if fileLines[i].startswith("*") and fileLines[i][1] != "*"  and i <= numberOfLines and i > intStartElementSection :
                intEndElementSection = i - 1
                secondCondition = True

            i += 1


        elements = []
        for line in fileLines[intStartElementSection: intEndElementSection+1] : 
            if line.startswith('**') : continue

            line = line.replace(' ', '').replace('\n', '')
            if line.endswith(',') : 
                line = line[:-1]

            line = line.split(',')
            elements.append([int(i) for i in line])
       
        self.elements = np.array(elements)
    

    def addAreaOfInterest(self, dictData) : 
        """ 
        """

        if dictData["type"] == 'contour' : 
            if dictData['region'] in self.nodeSets.keys() : 
                nodes = self.nodes[self.nodeSets[dictData["region"]]]
                self.determineConvexConcaveHull(nodes)

        elif "elements" in dictData.keys() :
            print("Not support yet") 

        # elif "" in dictData.keys() :
        # elif "" in dictData.keys() :
        
        else : 
            self.leftClass("")


    def determineConvexConcaveHull(self, nodes: np.ndarray, lengthThreshold: float = 1, plotToCheck: bool = False ) : 
        """ 
            Method to determine the convexe/concave hull
        """

        self.indexConvexeConcaveHull = concave_hull_indexes(nodes[:, :2],
                                                            length_threshold = lengthThreshold,
                                                            )
        
        self.meshOutline = nodes[self.indexConvexeConcaveHull]

        if plotToCheck is True : 
            fig = plt.figure()
            ax = fig.add_subplot(111)

            ax.plot(nodes[:, 0],
                    nodes[:, 1],
                    'o',
                    color = 'black',
                    mec  = 'black',
                    ms = 0.5,
                    mew = 0.1,
                    )


            ax.plot(self.meshOutline[:, 0],
                    self.meshOutline[:, 1],
                    '-',
                    color = 'red',
                    lw = 0.5)

            fig.savefig("concaveConvexeHull.pdf")

    
    def computeAngleDistanceBetweenFourPoints(self, n1, n2, n3, n4) : 
        """ 
        """


        pass
    

    def determineNSET(self, nSetName: str, commonCoordinate: str, nSetPrefix: str, saveName: str) : 
        """ 
        """

        # 1. select nodes : 
        nodes = self.nodes[self.nodeSets[nSetName]]

        # Select commonCoordinate :
        match commonCoordinate : 
            case "X" : 
                index = 0

            case "Y" :
                index = 1 

            case "Z" : 
                index = 2

            case _ : 
                print("Please select, X or Y or Z for commonCoordinate")
                return None

        with open(saveName, 'w') as file : 
            for i, val in enumerate(np.unique(nodes[:, index])) :  # Loop over unique value of specified commonCoordinate :
                file.write(f"*NSET, NSET={nSetPrefix}-{i+1}\n")

                selectedNodes = nodes[np.where(nodes[:, index] == val)][:, 3]
                # print(selectedNodes)
                for nodeNumber in selectedNodes : 
                    file.write("{:g}\n".format(int(nodeNumber)))
                

    #----------------------------------------------------------------
    #-------------------------Tikz Figures---------------------------
    #----------------------------------------------------------------
    def exportTikZelementReconstruction(self, nodes: np.ndarray, connectivity: list, saveName: str = "Tikz-export-elementReconstruction.tex") : 
        """ 
        """
        with open(saveName, 'w') as file : 
            # Header :
            file.write(r"\documentclass[border = 1cm]{standalone}" + '\n')
            file.write(r"\usepackage[usenames,dvipsnames,svgnames]{xcolor}" + '\n')
            file.write(r"\usepackage{tikz}" + '\n')

            # Node style :
            # file.write(r"" + '\n')

            file.write(r"\begin{document}" + '\n')
            file.write("\t" + r"\begin{tikzpicture}" + '\n')
    
            # Core :
            i = 1
            # for node, color in zip(nodes, ["red!50", "ForestGreen!50", "blue!50", "orange!50"]) : 
            for node, color in zip(nodes, ["red!50", "ForestGreen!50", "blue!50", "orange!50"]) : 
                file.write("\t" + r"\fill [%s] (%f, %f) circle (%f) ;" %(color, node[0], node[1], 0.05) + '\n')
                file.write("\t" + r"\node at (%f, %f) [font = \tiny] {N%i} ;" %(node[0], node[1], i) + "\n")
                i += 1
            
            n1 = nodes[0]
            n2 = nodes[1]
            n3 = nodes[2]
            n4 = nodes[3]

            file.write("\t" + r"\draw [-, red, line cap = round, line width = 1] (%f, %f) -- (%f, %f) ;" %(n1[0], n1[1], n2[0], n2[1]) + '\n')
            # file.write("\t" + r"\draw [->, red, line cap = round, line width = 1] (%f, %f) -- (%f, %f) ;" %(n1[0], n1[1], n2[0], n2[1]) + '\n')
            # file.write("\t" + r"\draw [->, ForestGreen, line cap = round, line width = 1] (%f, %f) -- (%f, %f) ;" %(n1[0], n1[1], n3[0], n3[1]) + '\n')
            # file.write("\t" + r"\draw [->, blue, line cap = round, line width = 1] (%f, %f) -- (%f, %f) ;" %(n1[0], n1[1], n4[0], n4[1]) + '\n')
            # file.write("\t" + r"\draw [ForestGreen, line cap = round, line width = 1] (%f, %f) -- (%f, %f) ;" %(n2[0], n2[1], n3[0], n3[1]) + '\n')
            file.write("\t" + r"\draw [-, blue, line cap = round, line width = 1] (%f, %f) -- (%f, %f) ;" %(n3[0], n3[1], n4[0], n4[1]) + '\n')
            # file.write("\t" + r"\draw [orange, line cap = round, line width = 1] (%f, %f) -- (%f, %f) ;" %(n4[0], n4[1], n1[0], n1[1]) + '\n')

            # Draw first vertices :
            if len(connectivity) == 4 : 
                n1 = nodes[connectivity[0]]
                n2 = nodes[connectivity[1]]
                n3 = nodes[connectivity[2]]
                n4 = nodes[connectivity[3]]

                file.write("\t" + r"\draw [->, black, line cap = round, line width = 0.5] (%f, %f) -- (%f, %f) ;" %(n1[0], n1[1], n2[0], n2[1]) + '\n')
                file.write("\t" + r"\draw [->, black, line cap = round, line width = 0.5] (%f, %f) -- (%f, %f) ;" %(n2[0], n2[1], n3[0], n3[1]) + '\n')
                file.write("\t" + r"\draw [->, black, line cap = round, line width = 0.5] (%f, %f) -- (%f, %f) ;" %(n3[0], n3[1], n4[0], n4[1]) + '\n')
                file.write("\t" + r"\draw [->, black, line cap = round, line width = 0.5] (%f, %f) -- (%f, %f) ;" %(n4[0], n4[1], n1[0], n1[1]) + '\n')


            # Footer :
            file.write("\t" + r"\end{tikzpicture}" + '\n')
            file.write(r"\end{document}" + '\n')

    
    def exportTikZmeshReconstruction(self, saveName: str = "Tikz-export-2Dmesh.tex", addAreaOfInterest: list | dict = None, useMatchIdNodesCoordinates = False) :
        """ 
        """

        dataAddAreaOfInteret = []

        # Check instance :
        # if isinstance(addAreaOfInterest, list) :
            # for i, el in enumerate(addAreaOfInterest) : 
                # if not isinstance(el, dict) :
                    # print("element no %i of addAreaOfInterest is not a string" %i)
            
                # else : 
                    # dataAddAreaOfInteret.append(self.addAreaOfInterest(el))
                        
        # elif isinstance(addAreaOfInterest, dict) : 
            # dataAddAreaOfInteret.append(self.addAreaOfInterest(addAreaOfInterest))

        # else : 


        if useMatchIdNodesCoordinates is False : 
            with open(saveName, 'w') as file : 
                # Header :
                file.write(r"\documentclass[border = 1cm]{standalone}" + '\n')
                file.write(r"\usepackage[usenames,dvipsnames,svgnames]{xcolor}" + '\n')
                file.write(r"\usepackage{tikz}" + '\n')
                file.write(r"\tikzset{feNodeStyle/.style={red}}" + "\n")
                file.write(r"\tikzset{feElementStyle/.style={line width = 0.1, line cap = round, Gold}}" + "\n")
                file.write(r"\begin{document}" + '\n')
                file.write("\t" + r"\begin{tikzpicture}" + '\n')
                file.write(r"\pgfmathsetmacro{\nodesCircleRayon}{0.1pt}" + "\n")
                file.write("\t\t" + r"\input{./nodesCoordinatesExpressInFeFrame}" + '\n')
                file.write("\t\t" + r"\input{./elementsExpressInFeFrame}" + '\n')
                file.write("\t" + r"\end{tikzpicture}" + '\n')
                file.write(r"\end{document}" + '\n')

            with open("./nodesCoordinatesExpressInFeFrame.tex", 'w') as nodeFile, open("./elementsExpressInFeFrame.tex", 'w') as elementFile :
                for element in self.twoDimensionnalConnectivityTable : # Loop over elements :
                    nodes = self.nodes[element - 1, 0:2]
                    for i, node in enumerate(nodes) : # Loop over nodes :
                        nodeFile.write(r"\fill [feNodeStyle] " + "(%f, %f) " %(node[0], node[1]) + r"circle (\nodeRadius)" + ";\n")

                        if i == 0 : 
                            elementFile.write("\t" + r"\draw [feElementStyle] (%f, %f) -- " %(node[0], node[1]) )


                        else : 
                            elementFile.write("(%f, %f) -- " %(node[0], node[1]) )
                    
                    elementFile.write("cycle ; \n")

        else : 
            with open(saveName, 'w') as file : 
                # Header :
                file.write(r"\documentclass[border = 1cm]{standalone}" + '\n')
                file.write(r"\usepackage[usenames,dvipsnames,svgnames]{xcolor}" + '\n')
                file.write(r"\usepackage{tikz}" + '\n')
                file.write(r"\tikzset{feNodeStyle/.style={red}}" + "\n")
                file.write(r"\tikzset{feElementStyle/.style={line width = 0.1, line cap = round, Gold}}" + "\n")
                file.write(r"\begin{document}" + '\n')
                file.write("\t" + r"\begin{tikzpicture}" + '\n')
                file.write("\t\t" + r"\pgfmathsetmacro{\nodeRadius}{0.1pt}" + "\n")
                file.write("\t\t" + r"\input{./nodesCoordinatesExpressInDicFrame}" + '\n')
                file.write("\t\t" + r"\input{./elementsCoordinatesExpressInDicFrame}" + '\n')
                file.write("\t" + r"\end{tikzpicture}" + '\n')
                file.write(r"\end{document}" + '\n')

            with open("./nodesCoordinatesExpressInDicFrame.tex", 'w') as nodeFile, open("./elementsCoordinatesExpressInDicFrame.tex", 'w') as elementFile :
                for element in self.twoDimensionnalConnectivityTable : # Loop over elements :
                    for i, node in enumerate(element) : # Loop over nodes :
                        n = self.nodesCoordinatesFromMatchID[self.nodesCoordinatesFromMatchID['Label'] == node ][["X_SENSOR", "Y_SENSOR"]].to_numpy().ravel()

                        nodeFile.write(r"\fill [feNodeStyle] " + "(axis cs:%f, %f) " %(n[0], n[1]) + r"circle (\nodeRadius)" + ";\n")

                        if i == 0 : 
                            elementFile.write("\t" + r"\draw [feElementStyle] (axis cs:%f, %f) -- " %(n[0], n[1]) )


                        else : 
                            elementFile.write("(axis cs:%f, %f) -- " %(n[0], n[1]) )
                    
                    elementFile.write("cycle ; \n")


    def exportTikzMeshOutline(self, saveName: str = "Tikz-exportOutline", coordinnateFrameToUse = "FE") : 
        """ 
            Method to export mesh outline for Tikz
            Different coordinate Frame available
        """

        match coordinnateFrameToUse :
            case 'FE' : 
                with open(f"{saveName}-FEcoordinateSystem.tex", 'w') as file : 
                    file.write(r'\draw [feMeshOutlineFEcoordinateSystemStyle] ')
                    for node in self.meshOutline :
                        file.write("({:f}, {:f}) -- \n".format(*node[0: 2]))

                    file.write("cycle ;\n")

            case "DIC" :
                with open(f"{saveName}-DICcoordinateSystem.tex", 'w') as file : 

                    file.write(r'\draw [feMeshOutlineDICcoordinateSystemStyle] ')

                    for node in self.meshOutline : # Loop over node who composed the outline
                        n = self.nodesCoordinatesFromMatchID[self.nodesCoordinatesFromMatchID['Label'] == node[3]][["X_CAM0", "Y_CAM0"]].to_numpy().ravel()

                        file.write("({:f}, {:f}) -- \n".format(*n))

                    file.write("cycle ;\n")

            case "IMG" :
                with open(f"{saveName}-IMGcoordinateSystem.tex", 'w') as file : 

                    file.write(r'\draw [feMeshOutlineIMGcoordinateSystemStyle] ')

                    for node in self.meshOutline : # Loop over node who composed the outline
                        n = self.nodesCoordinatesFromMatchID[self.nodesCoordinatesFromMatchID['Label'] == node[3]][["X_SENSOR", "Y_SENSOR"]].to_numpy().ravel()

                        file.write("({:f}, {:f}) -- \n".format(*n))

                    file.write("cycle ;\n")


    #----------------------------------------------------------------
    #------------------------------Export----------------------------
    #----------------------------------------------------------------
    def export2DConnectivityTable(self, saveName: str) :
        """ 
            Method to export 2D connectivity table
        """

        if self.boolExtract2DconnectivityTable is False : 
            print("The 2d connectivity table wasn't computed")
            return None

         
        np.savetxt(saveName,
                   np.column_stack((np.arange(1, len(self.twoDimensionnalConnectivityTable) + 1, 1, dtype = int),
                                    self.twoDimensionnalConnectivityTable),
                                   ),
                   delimiter = ";",
                   header = "Elt;N1;N2;N3;N4",
                   comments = '',
                   fmt = "%i",
                   )


    def exportMeshForMatchID(self, nodeSet: str, saveName: str, plotToCheck: bool = False) : 
        """ 
            Method to export 2D connectivity table for MatchID FEDEF/FEVAL module
        """

        if self.boolExtract2DconnectivityTable is False : 
            print("The 2d connectivity table wasn't computed")
            return None

        if nodeSet not in self.nodeSets.keys() : 
            print("The *NSET gave wasn't found")
            return None

        # nodes
        nodes = self.nodes[self.nodeSets[nodeSet]]

        # Reset X, Y, Z : 
        nodes[:, 0] -= nodes[:, 0].min()

        nodes[:, 1] *= -1
        nodes[:, 1] -= nodes[:, 1].min()

        nodes[:, 2] = 0.

        if plotToCheck is True : 
            fig = plt.figure()
            ax = fig.add_subplot(111,
                                 aspect = "equal")

            ax.plot(nodes[:, 0],
                    nodes[:, 1],
                    'o',
                    ms = 0.5,
                    mew = 0.1,
                    mec = 'blue',
                    color = 'blue',
                    )

            ax.set_xlabel('X_cam [mm]')
            ax.set_ylabel('Y_cam [mm]')
            ax.invert_yaxis()

            fig.savefig("PLOT-nodesCoordinatesInDicFrame"+ '.pdf',
                        format = 'pdf',
                        orientation = "portrait",
                        bbox_inches = "tight",
                        pad_inches = 0)

            
        with open(saveName, 'w') as exportFile : 
            exportFile.write("*Part, name=Part-1\n")
            exportFile.write("*Nodes\n")

            for node in nodes : # Loop over nodes :
                exportFile.write("{3:d};{0:f};{1:f};{2:f}\n".format(node[0], node[1], node[2], int(node[3])))

            exportFile.write("*Elements\n")

            for i in range(len(self.twoDimensionnalConnectivityTable))  : # Loop over elements :
                exportFile.write('{0};{1};{2};{3};{4}\n'.format(i + 1, *self.twoDimensionnalConnectivityTable[i]))
            
            exportFile.write("*End Part")
            

    def exportNodesCoordinatesInDicFrame(self, nset: str, saveName: str, PLOT = False) : 
        """ 
        """
        
        if nset not in self.nodeSets.keys() : 
            print("The *NSET %s doesn't exist" %nset)
        
        nodes = self.nodes[self.nodeSets[nset]]
        

        # Reset X, Y, Z : 
        nodes[:, 0] -= nodes[:, 0].min()

        nodes[:, 1] *= -1
        nodes[:, 1] -= nodes[:, 1].min()

        nodes[:, 2] = 0.

        # nodes[:, 2] *= -1
        # nodes[:, 2] -= nodes[:, 2].min()
        
        if PLOT is True : 
            self.plotNodesCoordinatesInDicFrame(nodes)

        
        np.savetxt(saveName, 
                   nodes,
                   delimiter = ";",
                   comments = '',
                   header = "X;Y;Z;Label",
                   )
    
    #----------------------------------------------------------------
    #------------------------------Import----------------------------
    #----------------------------------------------------------------
    def importNodemapFileFromMatchID(self, fileName: str) :
        """ 
            Method to load .nodemap file from MatchID
        """
        
        if not isinstance(fileName, str) : 
            self.leftClass("The argument %s isn't a string" %fileName)

        else : 
            if not os.path.exists(fileName) :
                self.leftClass("The file %s doesn't exists" %fileName)


        try : 
            self.nodesCoordinatesFromMatchID = pd.read_csv(fileName, 
                                                           header = None, 
                                                           skiprows=1,
                                                           names=['Label',
                                                                  'X_FEM',
                                                                  'Y_FEM',
                                                                  'Z_FEM',
                                                                  'X_CAM0',
                                                                  'Y_CAM0',
                                                                  'Z_CAM0',
                                                                  'X_SENSOR',
                                                                  'Y_SENSOR',
                                                                  ],
                                                           delimiter = ";",
                                                           )

            self.nodesCoordinatesFromMatchID_Sensor_frame = self.nodesCoordinatesFromMatchID[['X_SENSOR', 'Y_SENSOR']].to_numpy()
            self.nodesCoordinatesFromMatchID_Cam_frame = self.nodesCoordinatesFromMatchID[['X_CAM0', 'Y_CAM0']].to_numpy()


        except Exception as error : 
            print("Error :", error)
            self.leftClass("Error during file : %s openning" %fileName)

    
    #----------------------------------------------------------------
    #-------------------------------Plot-----------------------------
    #----------------------------------------------------------------
    def plotNodesCoordinatesInDicFrame(self, nodes: np.ndarray) : 
        """ 
        """ 
        
        fig = plt.figure()
        ax = fig.add_subplot(111,
                             aspect = 'equal')

        ax.plot(nodes[:, 0],
                nodes[:, 1],
                'o',
                ms = 0.5,
                mew = 0.1,
                mec = "black",
                color = "black")

        ax.invert_yaxis()
        
        ax.set_xlabel("X [mm]")
        ax.set_ylabel("Y [mm]")

        fig.savefig("PLOT-nodesCoordinatesInDicFrame"+ '.pdf',
                    format = 'pdf',
                    # dpi = 600,
                    orientation = "portrait",
                    bbox_inches = "tight",
                    pad_inches = 0)


    def leftClass(self, message: str = None) : 
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
    

class vector :
    """ 
    """
    def __init__(self, n1, n2) : 
        """ 
            Initialization :
        """ 

        self.u = n2[0] - n1[0]
        self.v = n2[1] - n1[1]

        self.norme = np.sqrt(self.u**2 + self.v**2)
 

