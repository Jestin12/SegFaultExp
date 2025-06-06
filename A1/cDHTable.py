import numpy as np
import math


# *************************** DHTable.py ***************************************

# Filename:       cDHTable.py
# Author:         Alfie

# Description:  This file defines a class that constructs a Denavit-Hartenberg
#               Table, which describes the joints of a robotic limb using the
#               Denavit-Hartenberg notation. 

#               The cDHTable class receives joint angle values, Theta, of the robotic 
#               limb whilst having pre-existing knowledge of the other dimensional 
#               values of the limb required for the DH Table such as Di, Ai, and Alpha
# 
#               The cDHTable class returns the transform matrices of each of the frames
#               corresponding to each robotic link through the mConstructHT function. 

# Dependencies: numpy   math

# ************************************************************************************


class cDHTable:

    # Constant definitions
    TABLE_COLUMNS = 4
    D = 0
    THETA = 1
    A = 2
    ALPHA = 3

    #Define link lengths
    L2 = 600
    L3 = 450

    # Initialise DH Table 
    def __init__(self, JointAngles, S1, S4):
        self.JointAngles = JointAngles
        self.NumJoints = len(JointAngles)
        self.DHTable = np.zeros((self.NumJoints, self.TABLE_COLUMNS))

        if self.NumJoints == 4:
            Di = [S1, 0, 0, self.L3 + S4]
            Ai = [0, self.L2, 0, 0]
            AlphaI = [math.pi/2,0,-(math.pi/2),0]
        elif self.NumJoints == 6:
            Di = [163, 0, 0, 127, 100, 100]
            Ai = [0, -425, -392, 0, 0, 0]
            AlphaI = [(math.pi)/2, 0, 0, (math.pi)/2, -(math.pi/2), 0]
        else:
            print("Incompatible Joint Angle Vector. Incorrect number of angles?")

        for row in range(self.NumJoints):
            self.DHTable[row][self.D] = Di[row]
            self.DHTable[row][self.THETA] = self.JointAngles[row]
            self.DHTable[row][self.A] = Ai[row]
            self.DHTable[row][self.ALPHA] = AlphaI[row]

    #Helper function to get DH parameters
    def mGetDHParameters(self, FrameNum):
        # Extract parameters
        Theta = self.DHTable[FrameNum][self.THETA]
        Di = self.DHTable[FrameNum][self.D]
        Ai = self.DHTable[FrameNum][self.A]
        Alpha = self.DHTable[FrameNum][self.ALPHA]

        # Compute trigonometric values
        Ct = math.cos(Theta)
        St = math.sin(Theta)
        Ca = math.cos(Alpha)
        Sa = math.sin(Alpha)

        return Ct, St, Ca, Sa, Di, Ai

    # Construct homogeneous transform matrix (either Standard or modified)
    def mConstructHT(self, FrameNum, Standard=True):
        Ct, St, Ca, Sa, Di, Ai = self.mGetDHParameters(FrameNum)

        if Standard:
            SHT = np.array([
                [Ct, -St * Ca, St * Sa, Ai * Ct],
                [St, Ct * Ca, -Ct * Sa, Ai * St],
                [0, Sa, Ca, Di],
                [0, 0, 0, 1]
            ])
            return SHT
        else:
            MHT = np.array([
                [Ct, -St, 0, Ai],
                [St * Ca, Ct * Ca, -Sa, -Sa * Di],
                [St * Sa, Ct * Sa, Ca, Ca * Di],
                [0, 0, 0, 1]
            ])
            return MHT

